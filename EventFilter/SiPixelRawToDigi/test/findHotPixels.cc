/** \class SiPixelRawDumper_H
 * To find hot pixels from raw data
 * Works in  v352
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

namespace {
  const bool printErrors  = true;
  const bool printData    = false;
  const bool printHeaders = false;
  int count1=0, count2=0, count3=0;
}

using namespace std;

// Include the helper decoding class
/////////////////////////////////////////////////////////////////////////////
class MyDecode {
public:
  MyDecode() {}
  ~MyDecode() {}
  static int error(int error, bool print=false);
  static int data(int error, int &channel, int &roc, int &dcol, int &pix, bool print=false);
  static int header(unsigned long long word64, bool print);
  static int trailer(unsigned long long word64, bool print);
private:
};
/////////////////////////////////////////////////////////////////////////////
int MyDecode::header(unsigned long long word64, bool printFlag) {
  int fed_id=(word64>>8)&0xfff;
  int event_id=(word64>>32)&0xffffff;
  unsigned int bx_id=(word64>>20)&0xfff;
//   if(bx_id!=101) {
//     cout<<" Header "<<" for FED "
//   <<fed_id<<" event "<<event_id<<" bx "<<bx_id<<endl;
//     int dummy=0;
//     cout<<" : ";
//     cin>>dummy;
//   }
  if(printFlag) cout<<" Header "<<" for FED "
             <<fed_id<<" event "<<event_id<<" bx "<<bx_id<<endl;
  
  return event_id;
}
//
int MyDecode::trailer(unsigned long long word64, bool printFlag) {
  int slinkLength = int( (word64>>32) & 0xffffff );
  int crc         = int( (word64&0xffff0000)>>16 );
  int tts         = int( (word64&0xf0)>>4);
  int slinkError  = int( (word64&0xf00)>>8);
  if(printFlag) cout<<" Trailer "<<" len "<<slinkLength
             <<" tts "<<tts<<" error "<<slinkError<<" crc "<<hex<<crc<<dec<<endl;
  return slinkLength;
}
//
// Decode error FIFO
// Works for both, the error FIFO and the SLink error words. d.k. 25/04/07
int MyDecode::error(int word, bool printFlag) {
  int status = -1;
  const unsigned int  errorMask      = 0x3e00000;
  const unsigned int  dummyMask      = 0x03600000;
  const unsigned int  gapMask        = 0x03400000;
  const unsigned int  timeOut        = 0x3a00000;
  const unsigned int  eventNumError  = 0x3e00000;
  const unsigned int  trailError     = 0x3c00000;
  const unsigned int  fifoError      = 0x3800000;

//  const unsigned int  timeOutChannelMask = 0x1f;  // channel mask for timeouts
  //const unsigned int  eventNumMask = 0x1fe000; // event number mask
  const unsigned int  channelMask = 0xfc000000; // channel num mask
  const unsigned int  tbmEventMask = 0xff;    // tbm event num mask
  const unsigned int  overflowMask = 0x100;   // data overflow
  const unsigned int  tbmStatusMask = 0xff;   //TBM trailer info
  const unsigned int  BlkNumMask = 0x700;   //pointer to error fifo #
  const unsigned int  FsmErrMask = 0x600;   //pointer to FSM errors
  const unsigned int  RocErrMask = 0x800;   //pointer to #Roc errors
  const unsigned int  ChnFifMask = 0x1f;   //channel mask for fifo error
  const unsigned int  Fif2NFMask = 0x40;   //mask for fifo2 NF
  const unsigned int  TrigNFMask = 0x80;   //mask for trigger fifo NF

 const int offsets[8] = {0,4,9,13,18,22,27,31};

 //cout<<"error word "<<hex<<word<<dec<<endl;

  if( (word&errorMask) == dummyMask ) { // DUMMY WORD
    //cout<<" Dummy word";
    return 0;
  } else if( (word&errorMask) == gapMask ) { // GAP WORD
    //cout<<" Gap word";
    return 0;
  } else if( (word&errorMask)==timeOut ) { // TIMEOUT
     // More than 1 channel within a group can have a timeout error
     unsigned int index = (word & 0x1F);  // index within a group of 4/5
     unsigned int chip = (word& BlkNumMask)>>8;
     int offset = offsets[chip];
     if(printErrors) {
       cout<<"Timeout Error- channels: ";
       for(int i=0;i<5;i++) {
      if( (index & 0x1) != 0) {
        int chan = offset + i + 1;
        cout<<chan<<" ";
      }
      index = index >> 1;
       }
       cout<<endl;
     }
     //end of timeout  chip and channel decoding
     
  } else if( (word&errorMask) == eventNumError ) { // EVENT NUMBER ERROR
    unsigned int channel =  (word & channelMask) >>26;
    unsigned int tbm_event   =  (word & tbmEventMask);
    
    if(printErrors) cout<<"Event Number Error- channel: "<<channel<<" tbm event nr. "
                     <<tbm_event<<endl;
    
  } else if( ((word&errorMask) == trailError)) {
    unsigned int channel =  (word & channelMask) >>26;
    unsigned int tbm_status   =  (word & tbmStatusMask);
    if(word & RocErrMask)
      if(printErrors) cout<<"Number of Rocs Error- "<<"channel: "<<channel<<" "<<endl;
    if(word & FsmErrMask)
      if(printErrors) cout<<"Finite State Machine Error- "<<"channel: "<<channel
                       <<" Error status:0x"<<hex<< ((word & FsmErrMask)>>9)<<dec<<" "<<endl;
    if(word & overflowMask)
      if(printErrors) cout<<"Overflow Error- "<<"channel: "<<channel<<" "<<endl;
    //if(!((word & RocErrMask)|(word & FsmErrMask)|(word & overflowMask)))
    if(tbm_status!=0)
      if(printErrors) cout<<"Trailer Error- "<<"channel: "<<channel<<" TBM status:0x"
                       <<hex<<tbm_status<<dec<<" "<<endl;

  } else if((word&errorMask)==fifoError) {
    if(printErrors) { 
      if(word & Fif2NFMask) cout<<"A fifo 2 is Nearly full- ";
      if(word & TrigNFMask) cout<<"The trigger fifo is nearly Full - ";
      if(word & ChnFifMask) cout<<"fifo-1 is nearly full for channel"<<(word & ChnFifMask);
      cout<<endl;
    }
  } else {
    cout<<" Unknown error?";
  }

  //unsigned int event   =  (word & eventNumMask) >>13;
  //unsigned int tbm_status   =  (word & tbmStatusMask);
  //if(event>0) cout<<":event: "<<event;
  //cout<<endl;

  return status;
}
// ///////////////////////////////////////////////////////////////////////////
int MyDecode::data(int word, int &c, int &r, int &d, int &p, bool printFlag) {
  
  const bool CHECK_PIXELS = true;
  //const bool PRINT_PIXELS = printData;
  const bool PRINT_PIXELS = false;
  
  const int ROCMAX = 24;
  const unsigned int plsmsk = 0xff;   // pulse height
  const unsigned int pxlmsk = 0xff00; // pixel index
  const unsigned int dclmsk = 0x1f0000;
  const unsigned int rocmsk = 0x3e00000;
  const unsigned int chnlmsk = 0xfc000000;
  int status = 0;

  int roc = ((word&rocmsk)>>21);
  // Check for embeded special words
  if(roc>0 && roc<25) {  // valid ROCs go from 1-24
    //if(PRINT_PIXELS) cout<<"data "<<hex<<word<<dec;
    unsigned int channel = ((word&chnlmsk)>>26);
    if(channel>0 && channel<37) {  // valid channels 1-36
      //cout<<hex<<word<<dec;
      int dcol=(word&dclmsk)>>16;
      int pix=(word&pxlmsk)>>8;
      int adc=(word&plsmsk);
      // print the roc number according to the online 0-15 scheme
      if(PRINT_PIXELS) cout<<" Channel- "<<channel<<" ROC- "<<(roc-1)<<" DCOL- "<<dcol<<" Pixel- "
          <<pix<<" ADC- "<<adc<<endl;
      if(CHECK_PIXELS) {
        if(roc>ROCMAX)
          cout<<" wrong roc number "<<channel<<"/"<<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
        if(dcol<0 || dcol>25)
          cout<<" wrong dcol number "<<channel<<"/"<<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
        if(pix<2 || pix>181)
          cout<<" wrong pix number chan/roc/dcol/pix/adc = "<<channel<<"/"<<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
      }
      c = channel;
      r = roc-1; // start roc counting from 0
      d = dcol;
      p = pix;
      status++;
    } else {
      cout<<"Wrong channel "<<channel<<endl;
      return -2;
    }
  } else {  // error word
    //cout<<"error word "<<hex<<word<<dec;
    status=error(word);
  }

  return status;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Class to get names
class MyConvert {
public:
  MyConvert() {}
  ~MyConvert() {}
  static string moduleNameFromFedChan(int fed,int fedChan, string & tbm);
private:
};

// Method returns the module name and the tbm type as strings
// input: int fed, fedChan
// output: string name, tbm ("A" or "B")
string MyConvert::moduleNameFromFedChan(int fed0,int fedChan0, string & tbm0) {
  if(fed0<0 || fed0>31) return " ";
  if(fedChan0<1 || fedChan0>36) return " ";

  ifstream infile; //input file, name data_file uniqe
  infile.open("translation_bpix.dat",ios::in); // open data file

  //cout << infile.eof() << " " << infile.bad() << " "
  //   << infile.fail() << " " << infile.good()<<endl;

  if (infile.fail()) {
    cout << " File not found " << endl;
    return(" "); // signal error
  }

  //string line;
  char line[500];
  infile.getline(line,500,'\n');

  string name, modName=" ";
  int fec,mfec,mfecChan,hub,port,rocId,fed,fedChan,rocOrder;
  string tbm = " ";
  int fedOld=-1, fedChanOld=-1;
  bool found = false;
  for(int i=0;i<100000;++i) {
    //bool print = false;

    infile>>name>>tbm>>fec>>mfec>>mfecChan>>hub>>port>>rocId>>fed>>fedChan>>rocOrder;
    
    if(name==" ")continue;
    
    if ( infile.eof() != 0 ) {
      cout<< " end of file " << endl;
      break;;
    } else if (infile.fail()) { // check for errors
      cout << "Cannot read data file" << endl;
      return(" ");
    }
    
    if(fed==fedOld && fedChanOld==fedChan) continue;
    fedOld = fed;
    fedChanOld = fedChan;


    if(fed==fed0 && fedChan==fedChan0) {  // found
      found = true;
      tbm0=tbm;

      string::size_type idx;
      idx = name.find("_ROC");
      if(idx != string::npos) {
        //      cout<<" ROC0 "<<idx<<endl;
        //name.replace(idx,idx+4,"     ");
        modName = name.substr(0,(idx));
      }


      break;
    }
  }  // end line loop
  
  infile.close();  
  if(!found) cout<<" Module not found "<<fed0<<" "<<fedChan0<<endl;

  return modName;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

const int NumPixels = 100000;
class HotPixels {
public:
  HotPixels() {count=0; for(int i=0;i<NumPixels;++i) {array[i]=0; data[i]=0;}}
  ~HotPixels() {}
  void update(int channel, int roc, int dcol, int pix); 
  int code(int channel, int roc, int dcol, int pix); 
  void decode(int index, int &channel, int &roc, int &dcol, int &pix); 
  void print(int, int);
private:
  int count;
  int array[NumPixels];
  int data[NumPixels];
};

int HotPixels::code(int channel, int roc, int dcol, int pix) {
  // pix 0 - 182, dcol 0 - 26 , roc 0 -15, chan 1-36
  int index = pix + 1000 * dcol + 100000 * roc + 10000000 * channel;
  return index;
}
void HotPixels::decode(int index, int &channel, int &roc, int &dcol, int &pix) {
  //int index = pix + 1000 * dcol + 100000 * roc + 10000000 * channel;
  channel =  index/10000000;
  roc     = (index%10000000)/100000;
  dcol    = (index%100000)/1000;
  pix     = (index%1000);
}
void HotPixels::update(int channel, int roc, int dcol, int pix) {
  int index = code(channel, roc, dcol, pix);
  //cout<<channel<<"   "<<roc<<"    "<<dcol<<"    "<<pix<<"    "<<index<<endl;
  bool found = false;
  for(int i=0;i<count;++i) {
    if(index == array[i]) {
      data[i]++;
      found=true;
      break;
    }
  }
  if(!found) {
    if(count>=NumPixels) {
      cout<<" array too small "<<count<<" "<<endl;
    } else {
      data[count] =1;
      array[count]=index;
      count++;
    }
  }

}
void HotPixels::print(int events, int fed_id) {
  int channel=0, roc=0, dcol=0, pix=0;
  int num =0;
  int cut1 = events/100;
  int cut2 = events/1000;
  int cut3 = events/10000;

  int cut = events/1000;
  //int cut = 2;
  if(cut<2) cut=10;



  if(fed_id==0) {
    cout<<" Threshold of "<<cut<<endl;
    cout<<"fed chan     module                  tbm roc dcol  pix  colR ";   
    cout<<"rowR count  num roc-local"<<endl;   
  }
  for(int i=0;i<count;++i) {

    if(data[i]>cut1) count1++;
    if(data[i]>cut2) count2++;
    if(data[i]>cut3) count3++;

    if(data[i]>cut) {
      num++;
      int index = array[i];
      decode(index, channel, roc, dcol, pix);


      // First find if we are in the first or 2nd col of a dcol.
      int colEvenOdd = pix%2;  // module(2), 0-1st sol, 1-2nd col.
      // Transform
      int colROC = dcol * 2 + colEvenOdd; // col address, starts from 0
      int rowROC = abs( int(pix/2) - 80); // row addres, starts from 0
      //cout<<index<<" ";

      // Get the module name and tbm type 
      string modName = " ",tbm=" ";
      modName = MyConvert::moduleNameFromFedChan(fed_id,channel,tbm);
      int realRocNum = roc;
      if(tbm=="B") realRocNum = roc + 8; // shift for TBM-N
      cout<<setw(3)<<fed_id<<" "<<setw(3)<<channel<<" "<<setw(30)<<modName<<" "
          <<tbm<<" "<<setw(3)
          <<realRocNum<<"  "<<setw(3)<<dcol<<"  "<<setw(3)<<pix<<"   "
          <<setw(3)<<colROC<<"  "<<setw(3)<<rowROC<<"  "<<setw(4)<<data[i]<<"  "
          <<setw(3)<<num<<"  "<<setw(3)<<roc<<endl;
    }
  }
  //cout<<num<<" total 'noisy' pixels above the cut = "<<cut<<endl;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main program
class findHotPixels : public edm::EDAnalyzer {
public:

  /// ctor
  explicit findHotPixels( const edm::ParameterSet& cfg) : theConfig(cfg) {
  consumes<FEDRawDataCollection>(theConfig.getUntrackedParameter<std::string>("InputLabel","source"));

} 
  
  /// dtor
  virtual ~findHotPixels() {}

  void beginJob();

  //void beginRun( const edm::EventSetup& ) {}

  // end of job 
  void endJob();

  /// get data, convert to digis attach againe to Event
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
private:
  edm::ParameterSet theConfig;
  int countEvents, countAllEvents;
  float sumPixels, sumFedPixels[40];
  HotPixels hotPixels[40];
};

void findHotPixels::endJob() {
  if(countEvents>0) {
    sumPixels /= float(countEvents);
    for(int i=0;i<40;++i) sumFedPixels[i] /= float(countEvents);
  }
  
  cout<<" Total/non-empty events " <<countAllEvents<<" / "<<countEvents<<" average number of pixels "<<sumPixels<<endl;

  //for(int i=0;i<40;++i) cout<<sumFedPixels[i]<<" ";
  //cout<<endl;

  for(int i=0;i<40;++i) {
    hotPixels[i].print(countAllEvents,i);
  }
  cout<<" Number of noisy pixels: 1% "<<count1<<" 0.1% "<<count2<<" 0.01% "<<count3<<endl;


}

void findHotPixels::beginJob() {
  countEvents=0;
  countAllEvents=0;
  sumPixels=0.;
  for(int i=0;i<40;++i) sumFedPixels[i]=0;
}

void findHotPixels::analyze(const  edm::Event& ev, const edm::EventSetup& es) {

  edm::Handle<FEDRawDataCollection> buffers;
  static std::string label = theConfig.getUntrackedParameter<std::string>("InputLabel","source");
  static std::string instance = theConfig.getUntrackedParameter<std::string>("InputInstance","");
  
  ev.getByLabel( label, instance, buffers);

  std::pair<int,int> fedIds(FEDNumbering::MINSiPixelFEDID, FEDNumbering::MAXSiPixelFEDID);
  
  //PixelDataFormatter formatter(0); // to get digis
  //bool dummyErrorBool;
  
  typedef uint32_t Word32;
  typedef uint64_t Word64;
  int status=0;
  int countPixels=0;
  int countErrors=0;
  int eventId = -1;
  int channel=-1, roc=-1, dcol=-1, pix=-1;
  
  countAllEvents++;
  if(printHeaders) cout<<" Event = "<<countEvents<<endl;
  
  //edm::DetSetVector<PixelDigi> collection; // for digis only


  // Loop over FEDs
  for (int fedId = fedIds.first; fedId <= fedIds.second; fedId++) {
    LogDebug("findHotPixels")<< " GET DATA FOR FED: " <<  fedId ;
    if(printHeaders) cout<<" For FED = "<<fedId<<endl;

     PixelDataFormatter::Errors errors;
    
    //get event data for this fed
    const FEDRawData& rawData = buffers->FEDData( fedId );
    
    int nWords = rawData.size()/sizeof(Word64);
    //cout<<" size "<<nWords<<endl;
    
    // check headers
    const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); 
    //cout<<hex<<*header<<dec<<endl;
    eventId = MyDecode::header(*header, printHeaders);
    //if(fedId = fedIds.first) 

    const Word64* trailer = reinterpret_cast<const Word64* >(rawData.data())+(nWords-1);
    //cout<<hex<<*trailer<<dec<<endl;
    status = MyDecode::trailer(*trailer,printHeaders);

    int countPixelsInFed=0;
    int countErrorsInFed=0;
    // Loop over payload words
    for (const Word64* word = header+1; word != trailer; word++) {
      static const Word64 WORD32_mask  = 0xffffffff;
      Word32 w1 =  *word       & WORD32_mask;
      status = MyDecode::data(w1, channel, roc, dcol, pix, printData);
      if(status>0) {
	countPixels++;
	countPixelsInFed++;
        hotPixels[fedId].update(channel,roc,dcol,pix);
      } else if(status<0) countErrorsInFed++;
      Word32 w2 =  *word >> 32 & WORD32_mask;
      status = MyDecode::data(w2, channel, roc, dcol, pix, printData);
      if(status>0) {
	countPixels++;
	countPixelsInFed++;
        hotPixels[fedId].update(channel,roc,dcol,pix);
      } else if(status<0) countErrorsInFed++;
      //cout<<hex<<w1<<" "<<w2<<dec<<endl;
    } // loop over words
    
    countErrors += countErrorsInFed;
    
    //convert data to digi (dummy for the moment)
    //formatter.interpretRawData( dummyErrorBool, fedId, rawData,  collection, errors);
    //cout<<dummyErrorBool<<" "<<digis.size()<<" "<<errors.size()<<endl;
    
    if(countPixelsInFed>0)  {
      sumFedPixels[fedId] += countPixelsInFed;
    }

  } // loop over feds

  if(countPixels>0) {
    cout<<"EVENT: "<<countEvents<<" "<<eventId<<" pixels "<<countPixels<<" errors "<<countErrors<<endl;
    sumPixels += countPixels;
    countEvents++;
    //int dummy=0;
    //cout<<" : ";
    //cin>>dummy;
  }
  
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(findHotPixels);
