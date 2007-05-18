#include "EventFilter/HcalRawToDigi/interface/HcalPacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalPacker::Collections::Collections() {
  hbhe=0;
  hoCont=0;
  hfCont=0;
  tpCont=0;
  zdcCont=0;
  calibCont=0;
}

template <class Coll, class DetIdClass> 
int process(const Coll* pt, const DetId& did, unsigned short* buffer, int& presamples) {
  if (pt==0) return 0;
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

static unsigned char processTrig(const HcalTrigPrimDigiCollection* pt, const HcalTrigTowerDetId& tid, unsigned short* buffer, int exppresamples, int expsize) {
  if (pt==0) return 0;
  int size=0;
  HcalTrigPrimDigiCollection::const_iterator i=pt->find(tid);
  if (i!=pt->end()) {
    int presamples=i->presamples();
    int samples=i->size();

    // rare care of no precision digis, but trig prim digis
    if (expsize<0) expsize=samples;
    if (exppresamples<0) exppresamples=presamples;
    // we must match samples and presamples with zero tps
    for (int j=0; j<expsize; j++)
      buffer[j]=0;
    
    int offset=exppresamples-presamples;

    for (int j=0; j<samples; j++) 
      if (j+offset>=0 && j+offset<expsize)
	buffer[j+offset]=(*i)[j].raw();
    size=expsize;
  }
  return size;
}

int HcalPacker::findSamples(const DetId& did, const Collections& inputs,
			    unsigned short* buffer, int &presamples) {
  if (did.det()!=DetId::Hcal) return 0;
  int size=0;
  HcalGenericDetId genId(did);
  
  switch (genId.genericSubdet()) {
  case(HcalGenericDetId::HcalGenBarrel):
  case(HcalGenericDetId::HcalGenEndcap):
    size=process<HBHEDigiCollection,HcalDetId>(inputs.hbhe,did,buffer,presamples);
    break;
  case(HcalGenericDetId::HcalGenOuter):
    size=process<HODigiCollection,HcalDetId>(inputs.hoCont,did,buffer,presamples);
    break;
  case(HcalGenericDetId::HcalGenForward):
    size=process<HFDigiCollection,HcalDetId>(inputs.hfCont,did,buffer,presamples);
    break;
  case(HcalGenericDetId::HcalGenZDC):
    size=process<ZDCDigiCollection,HcalZDCDetId>(inputs.zdcCont,did,buffer,presamples);
    break;
  case(HcalGenericDetId::HcalGenCalibration):
    size=process<HcalCalibDigiCollection,HcalCalibDetId>(inputs.calibCont,did,buffer,presamples);
    break;
  default: size=0;
  }
  return size;
}

void HcalPacker::pack(int fedid, int dccnumber,
		      int nl1a, int orbitn, int bcn,
		      const Collections& inputs, 
		      const HcalElectronicsMap& emap,
		      FEDRawData& output) {
  std::vector<unsigned short> precdata(HcalHTRData::CHANNELS_PER_SPIGOT*HcalHTRData::MAXIMUM_SAMPLES_PER_CHANNEL);
  std::vector<unsigned short> trigdata(HcalHTRData::CHANNELS_PER_SPIGOT*HcalHTRData::MAXIMUM_SAMPLES_PER_CHANNEL);
  std::vector<unsigned char> preclen(HcalHTRData::CHANNELS_PER_SPIGOT);
  std::vector<unsigned char> triglen(HcalHTRData::CHANNELS_PER_SPIGOT);
  const int HTRFormatVersion=1;

  HcalHTRData spigots[15];
  // loop over all valid channels in the given dcc, spigot by spigot.
  for (int spigot=0; spigot<15; spigot++) {
    spigots[spigot].allocate(HTRFormatVersion);
    HcalElectronicsId exampleEId;
    int npresent=0;
    int presamples=-1, samples=-1;
    for (int fiber=1; fiber<=8; fiber++) 
      for (int fiberchan=0; fiberchan<3; fiberchan++) {
	int linear=(fiber-1)*3+fiberchan;
	unsigned short chanid=(((fiber-1)&0x7)<<13)|((fiberchan&0x3)<<11);
	preclen[linear]=0;

	HcalElectronicsId partialEid(fiberchan,fiber,spigot,dccnumber);
	// does this partial id exist?
	HcalElectronicsId fullEid;
	DetId genId;
	if (!emap.lookup(partialEid,fullEid,genId)) continue;


	// next, see if there is a digi with this id
	unsigned short* database=&(precdata[linear*HcalHTRData::MAXIMUM_SAMPLES_PER_CHANNEL]);
	int mypresamples;
	int mysamples=findSamples(genId,inputs,database,mypresamples);

	if (mysamples>0) {
	  if (samples<0) samples=mysamples;
	  else if (samples!=mysamples) {
	    edm::LogError("HCAL") << "Mismatch of samples in a single HTR (unsupported) " << mysamples << " != " << samples;
	    continue;
	  }
	  if (presamples<0) {
	    presamples=mypresamples;
	    exampleEId=fullEid;
	  } else if (mypresamples!=presamples) {
	    edm::LogError("HCAL") << "Mismatch of presamples in a single HTR (unsupported) " << mypresamples << " != " << presamples;
	    continue;	    
	  }
	  for (int ii=0; ii<samples; ii++)
	    database[ii]=(database[ii]&0x7FF)|chanid;
	  preclen[linear]=(unsigned char)(samples);
	  npresent++;
	}	
      }
    for (int fiber=1; fiber<=8; fiber++) 
      for (int fiberchan=0; fiberchan<3; fiberchan++) {
	int linear=(fiber-1)*3+fiberchan;
	unsigned short chanid=(((fiber-1)&0x7)<<13)|((fiberchan&0x3)<<11);
	triglen[linear]=0;
	
	HcalElectronicsId partialEid(dccnumber,spigot,fiber,fiberchan);
	// does this partial id exist?
	HcalElectronicsId fullEid;
	DetId genId;
	if (!emap.lookup(partialEid,fullEid,genId)) continue;
          
	// finally, what about a trigger channel?
	HcalTrigTowerDetId tid=emap.lookupTrigger(fullEid);
	if (!tid.null()) {
	  unsigned short* trigbase=&(trigdata[linear*HcalHTRData::MAXIMUM_SAMPLES_PER_CHANNEL]);
	  triglen[linear]=processTrig(inputs.tpCont,tid,trigbase,samples,presamples);
	  if (samples<0 && triglen[linear]>0) samples=triglen[linear];
	  for (unsigned char q=0; q<triglen[linear]; q++)
	    trigbase[q]=(trigbase[q]&0x7FF)|(chanid&0x1F);
	}
      }
    /// pack into HcalHTRData
    if (npresent>0) {
      spigots[spigot].pack(&(preclen[0]),&(precdata[0]),
			   &(triglen[0]),&(trigdata[0]),
			   false);
      static const int pipeline=0x22;
      static const int firmwareRev=0;
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
  for (int spigot=0; spigot<15; spigot++) {
    theSize+=spigots[spigot].getRawLength()*sizeof(unsigned short);
  }
  theSize+=sizeof(HcalDCCHeader)+8; // 8 for trailer
  theSize+=(8-(theSize%8))%8; // even number of 64-bit words.
  output.resize(theSize);
  
  // construct the bare DCC Header
  HcalDCCHeader* dcc=(HcalDCCHeader*)(output.data());
  dcc->clear();
  dcc->setHeader(fedid,bcn,nl1a,orbitn);

  // pack the HTR data into the FEDRawData block using HcalDCCHeader
  for (int spigot=0; spigot<15; spigot++) {
    if (spigots[spigot].getRawLength()>0)
      dcc->copySpigotData(spigot,spigots[spigot],true,0);
  }
}
