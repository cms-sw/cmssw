#include "EventFilter/CastorRawToDigi/interface/CastorCtdcPacker.h"
#include "EventFilter/CastorRawToDigi/interface/CastorCORData.h"
#include "EventFilter/CastorRawToDigi/interface/CastorMergerData.h"
#include "EventFilter/CastorRawToDigi/interface/CastorCTDCHeader.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "FWCore/Utilities/interface/CRC16.h"
#include "EventFilter/CastorRawToDigi/interface/CastorCollections.h"
#include <iostream>

using namespace std;

namespace {
template <class Coll, class DetIdClass> 
int process(const Coll* pt, const DetId& did, unsigned short* buffer, int& presamples) {
  if (pt==nullptr) return 0;
  int size=0;
  typename Coll::const_iterator i=pt->find(DetIdClass(did));
  if (i!=pt->end()) {
    presamples=i->presamples();
    size=i->size();
    for (int j=0; j<size; j++) 
      buffer[j]=(*i)[j].raw();
  }
  return size;
}
}

int CastorCtdcPacker::findSamples(const DetId& did, const CastorCollections& inputs,
			    unsigned short* buffer, int &presamples) {

  if (did.det()!=DetId::Calo) return 0;
  int size=0;
  HcalCastorDetId genId(did);

  size=process<CastorDigiCollection,HcalCastorDetId>(inputs.castorCont,did,buffer,presamples);

  return size;
}

void CastorCtdcPacker::pack(int fedid, int dccnumber,
		      int nl1a, int orbitn, int bcn,
		      const CastorCollections& inputs, 
		      const CastorElectronicsMap& emap,
		      FEDRawData& output) {
  std::vector<unsigned short> precdata(CastorCORData::CHANNELS_PER_SPIGOT*CastorCORData::MAXIMUM_SAMPLES_PER_CHANNEL);
  std::vector<unsigned short> trigdata(CastorCORData::CHANNELS_PER_SPIGOT*CastorCORData::MAXIMUM_SAMPLES_PER_CHANNEL);
  std::vector<unsigned char> preclen(CastorCORData::CHANNELS_PER_SPIGOT);
  std::vector<unsigned char> triglen(CastorCORData::CHANNELS_PER_SPIGOT);
  constexpr int CORFormatVersion=1;

//  CastorCORData spigots[CastorCTDCHeader::SPIGOT_COUNT];
  CastorCORData spigots[2];
  // loop over all valid channels in the given ctdc, spigot by spigot.
  for (int spigot=0; spigot<CastorCTDCHeader::SPIGOT_COUNT; spigot++) {
    spigots[spigot].allocate(CORFormatVersion);
    CastorElectronicsId exampleEId;
    int npresent=0;
    int presamples=-1, samples=-1;
    for (int fiber=1; fiber<=12; fiber++) 
      for (int fiberchan=0; fiberchan<3; fiberchan++) {
	int linear=(fiber-1)*3+fiberchan;
//	HcalQIESample chanSample(0,0,fiber,fiberchan,false,false);
//	unsigned short chanid=chanSample.raw()&0xF800;
	preclen[linear]=0;
	CastorElectronicsId partialEid(fiberchan,fiber,spigot,dccnumber);
	// does this partial id exist?
	CastorElectronicsId fullEid;
	HcalGenericDetId genId;
	if (!emap.lookup(partialEid,fullEid,genId)) continue;

	// next, see if there is a digi with this id
	unsigned short* database=&(precdata[linear*CastorCORData::MAXIMUM_SAMPLES_PER_CHANNEL]);
	int mypresamples;
	int mysamples=findSamples(genId,inputs,database,mypresamples);
	if (mysamples>0) {
	  if (samples<0) samples=mysamples;
	  else if (samples!=mysamples) {
	    edm::LogError("CASTOR") << "Mismatch of samples in a single COR (unsupported) " << mysamples << " != " << samples;
	    continue;
	  }
	  if (presamples<0) {
	    presamples=mypresamples;
	    exampleEId=fullEid;
	  } else if (mypresamples!=presamples) {
	    edm::LogError("CASTOR") << "Mismatch of presamples in a single COR (unsupported) " << mypresamples << " != " << presamples;
	    continue;	    
	  }
	  preclen[linear]=(unsigned char)(samples);
	  npresent++;

	}	
      }
    /// pack into CastorCORData
    if (npresent>0) {
      spigots[spigot].pack(&(preclen[0]),&(precdata[0]),
			   &(triglen[0]),&(trigdata[0]),
			   true);
      constexpr int pipeline=0x22;
      constexpr int firmwareRev=0;
      int submodule=exampleEId.htrTopBottom()&0x1;
      submodule|=(exampleEId.htrSlot()&0x1F)<<1;
      submodule|=(exampleEId.readoutVMECrateId()&0x1f)<<6;
      spigots[spigot].packHeaderTrailer(nl1a,
					bcn,
					submodule,
					orbitn,
					pipeline,
					samples,
					presamples,
					firmwareRev);
      
    }
  }
  // calculate the total length, and resize the FEDRawData
  int theSize=0;
  for (int spigot=0; spigot<2; spigot++) {
    theSize+=spigots[spigot].getRawLength()*sizeof(unsigned short);
  }
  // the merger payload - not yet defined 
  CastorMergerData mergerdata;
  // would need to fill mergdata here
  theSize+=mergerdata.getRawLength()*sizeof(unsigned short);
  
  theSize+=sizeof(CastorCTDCHeader)+8; // 8 for trailer
  theSize+=(8-(theSize%8))%8; // even number of 64-bit words.
  output.resize(theSize);
  
  // construct the bare CTDC Header
  CastorCTDCHeader* dcc=(CastorCTDCHeader*)(output.data());
  dcc->clear();
  dcc->setHeader(fedid,bcn,nl1a,orbitn);

  // pack the HTR data into the FEDRawData block using CastorCTDCHeader
  for (int spigot=0; spigot<2; spigot++) {
    if (spigots[spigot].getRawLength()>0)
      dcc->copySpigotData(spigot,spigots[spigot],true,0);
  }
  if ( mergerdata.getRawLength()>0)
	dcc->copyMergerData(mergerdata,true);
  // trailer
  FEDTrailer fedTrailer(output.data()+(output.size()-8));
  fedTrailer.set(output.data()+(output.size()-8),
    output.size()/8,
    evf::compute_crc(output.data(),output.size()), 0, 0);

}
