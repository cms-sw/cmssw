#include "EventFilter/ESRawToDigi/interface/ESUnpackerV4.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include <fstream>

ESUnpackerV4::ESUnpackerV4(const ParameterSet& ps) 
  : pset_(ps), fedId_(0), run_number_(0), orbit_number_(0), bx_(0), lv1_(0), trgtype_(0)
{

  debug_ = pset_.getUntrackedParameter<bool>("debugMode", false);
  lookup_ = ps.getUntrackedParameter<FileInPath>("LookupTable");

  m2  = ~(~Word64(0) << 2);
  m4  = ~(~Word64(0) << 4);
  m5  = ~(~Word64(0) << 5);
  m8  = ~(~Word64(0) << 8);
  m16 = ~(~Word64(0) << 16);
  m32 = ~(~Word64(0) << 32);

  // read in look-up table
  int nLines, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
  ifstream file;
  file.open(lookup_.fullPath().c_str());
  if( file.is_open() ) {

    file >> nLines;

    for (int i=0; i<nLines; ++i) {
      file>> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;

      zside_[kchip-1][pace-1] = iz;
      pl_[kchip-1][pace-1] = ip;
      x_[kchip-1][pace-1] = ix;
      y_[kchip-1][pace-1] = iy;      
    }
    
  } else {
    cout<<"ESUnpackerV4::ESUnpackerV4 : Look up table file can not be found in "<<lookup_.fullPath().c_str()<<endl;
  }

}

ESUnpackerV4::~ESUnpackerV4() {
}

void ESUnpackerV4::interpretRawData(int fedId, const FEDRawData & rawData, ESDigiCollection & digis) {
  
  int nWords = rawData.size()/sizeof(Word64);
  if (nWords==0) return;
  int dccWords = 6;
  int head, kid, kFlag1, kFlag2, kBC, kEC, optoBC, optoEC, ttcEC;
  
  // Event header
  const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); --header;
  bool moreHeaders = true;
  while (moreHeaders) {
    ++header;
    FEDHeader ESHeader( reinterpret_cast<const unsigned char*>(header) );
    if ( !ESHeader.check() ) break; // throw exception?
    if ( ESHeader.sourceID() != fedId) throw cms::Exception("PROBLEM in ESUnpackerV4 !");

    fedId_ = ESHeader.sourceID();
    lv1_   = ESHeader.lvl1ID();
    bx_    = ESHeader.bxID();

    if (debug_) {
      cout<<"[ESUnpackerV4]: FED Header candidate. Is header? "<< ESHeader.check();
      if (ESHeader.check())
	cout <<". BXID: "<<bx_<<" SourceID : "<<fedId_<<" L1ID: "<<lv1_<<endl;
      else cout<<" WARNING!, this is not a ES Header"<<endl;
    }
    
    moreHeaders = ESHeader.moreHeaders();
  }

  // Event trailer
  const Word64* trailer = reinterpret_cast<const Word64* >(rawData.data())+(nWords-1); ++trailer;
  bool moreTrailers = true;
  while (moreTrailers) {
    --trailer;
    FEDTrailer ESTrailer(reinterpret_cast<const unsigned char*>(trailer));
    if ( !ESTrailer.check()) { ++trailer; break; } // throw exception?
    if ( ESTrailer.lenght()!= nWords) throw cms::Exception("PROBLEM in ESUnpackerV4 !!");

    if (debug_)  {
      cout<<"[ESUnpackerV4]: FED Trailer candidate. Is trailer? "<<ESTrailer.check();
      if (ESTrailer.check())
        cout<<". Length of the ES event: "<<ESTrailer.lenght()<<endl;
      else cout<<" WARNING!, this is not a ES Trailer"<<endl;
    }

    moreTrailers = ESTrailer.moreTrailers();
  }

  // DCC data
  int count = 0;
  for (const Word64* word=(header+1); word!=(header+dccWords+1); ++word) {
    if (debug_) cout<<"DCC   : "<<print(*word)<<endl;

    head = (*word >> 60) & m4;
    if (head == 3) {
      count++;
    } else {
      cout<<"WARNING ! ES-DCC data are not correct !"<<endl;
    }

  } 
  
  // Event data
  for (const Word64* word=(header+dccWords+1); word!=trailer; ++word) {
    if (debug_) cout<<"Event : "<<print(*word)<<endl;

    head = (*word >> 60) & m4;

    if (head == 12) {
      word2digi(kid, *word, digis);
    } else if (head == 9) {
      kid    = (*word >> 0) & m16;
      kFlag2 = (*word >> 16) & m8;
      kFlag1 = (*word >> 24) & m8;
      kBC    = (*word >> 32) & m16;
      kEC    = (*word >> 48) & m8;
    } else if (head == 6) {
      ttcEC  = (*word >> 0) & m32;
      optoBC = (*word >> 32) & m16;
      optoEC = (*word >> 48) & m8;      
    }
  }

}

void ESUnpackerV4::word2digi(int kid, const Word64 & word, ESDigiCollection & digis) 
{

  int adc[3];
  adc[0]    = (word >> 0)  & m16;
  adc[1]    = (word >> 16) & m16;
  adc[2]    = (word >> 32) & m16;
  int strip = (word >> 48) & m5;
  int pace  = (word >> 53) & m2;

  if (debug_) cout<<kid<<" "<<strip<<" "<<pace<<" "<<adc[2]<<" "<<adc[1]<<" "<<adc[0]<<endl;

  int zside, plane, ix, iy;
  zside = zside_[kid-1][pace];
  plane = pl_[kid-1][pace];
  ix    = x_[kid-1][pace];
  iy    = y_[kid-1][pace];

  if (debug_) cout<<"DetId : "<<zside<<" "<<plane<<" "<<ix<<" "<<iy<<" "<<strip+1<<endl;

  ESDetId detId(strip+1, ix, iy, plane, zside);
  ESDataFrame df(detId);
  df.setSize(3);

  for (int i=0; i<3; i++) df.setSample(i, adc[i]);  

  digis.push_back(df);

  if (debug_) 
    cout<<"Si : "<<detId.zside()<<" "<<detId.plane()<<" "<<detId.six()<<" "<<detId.siy()<<" "<<detId.strip()<<" ("<<kid<<","<<pace<<") "<<df.sample(0).adc()<<" "<<df.sample(1).adc()<<" "<<df.sample(2).adc()<<endl;

}

string ESUnpackerV4::print(const  Word64 & word) const
{
  ostringstream str;
  str << "Word64:  " << reinterpret_cast<const bitset<64>&> (word);
  return str.str();
}

