// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
// $Id: MatacqDataFormatter.cc,v 1.1 2009/02/25 14:44:25 pgras Exp $

#include "EventFilter/EcalRawToDigi/src/MatacqDataFormatter.h"
#include "EventFilter/EcalRawToDigi/interface/MatacqRawEvent.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/EcalDigi/interface/EcalMatacqDigi.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

//#define MATACQ_DEBUG

void
MatacqDataFormatter::interpretRawData(const FEDRawData & data, EcalMatacqDigiCollection& matacqDigiCollection) {
#if MATACQ_DEBUG
  cout << "****************************************************************\n";
  cout << "********************** MATACQ decoder **************************\n";
  cout << "****************************************************************\n";
  cout << "FEDRawData: \n";
  char oldPad = cout.fill('0');
  for(int i=0; i < max(100, (int)data.size()); ++i){
    cout << hex << setw(2) << (int)(data.data()[i])
	 << ((i+1)%8?" ":"\n") ;
  }
  cout.fill(oldPad);
  cout << "======================================================================\n";
#endif //MATACQ_DEBUG defined
  interpretRawData(MatacqRawEvent(data.data(), data.size()),
		   matacqDigiCollection);  
}

void
MatacqDataFormatter::interpretRawData(const MatacqRawEvent& matacq, EcalMatacqDigiCollection& matacqDigiCollection) {
#if MATACQ_DEBUG
  printData(cout, matacq);
#endif //MATACQ_DEBUG defined

  const double ns = 1.e-9; //ns->s
  const double ps = 1.e-12;//ps->s
  double ts = ns/matacq.getFreqGHz();
  double tTrig = matacq.getTTrigPs()<.5*numeric_limits<int>::max()?
    ps*matacq.getTTrigPs():999.;
  int version = matacq.getMatacqDataFormatVersion();
  
  vector<int16_t> samples;
  //FIXME: the interpretRawData method should fill an EcalMatacqDigiCollection
  //instead of an EcalMatacqDigi because Matacq channels are several.
  //In the meamtime copy only the first channel appearing in data:
  const vector<MatacqRawEvent::ChannelData>& chData = matacq.getChannelData();
  for(unsigned iCh=0; iCh < chData.size(); ++iCh){
    //copy time samples into a vector:
    samples.resize(chData[iCh].nSamples);
    copy(chData[iCh].samples, chData[iCh].samples+chData[iCh].nSamples,
	 samples.begin());
    int chId = chData[iCh].chId;
    vector<int16_t> empty;
    EcalMatacqDigi matacqDigi(empty, chId, ts, version, tTrig);
#if (defined(ECAL_MATACQ_DIGI_VERS) && (ECAL_MATACQ_DIGI_VERS >= 2))
    matacqDigi.bxId(matacq.getBxId());
    matacqDigi.l1a(matacq.getEventId());
    matacqDigi.triggerType(matacq.getTriggerType());
    matacqDigi.orbitId(matacq.getOrbitId());
    matacqDigi.trigRec(matacq.getTrigRec());
    matacqDigi.postTrig(matacq.getPostTrig());
    matacqDigi.vernier(matacq.getVernier());
    matacqDigi.delayA(matacq.getDelayA());
    matacqDigi.emtcDelay(matacq.getEmtcDelay());
    matacqDigi.emtcPhase(matacq.getEmtcPhase());
    matacqDigi.attenuation_dB(matacq.getAttenuation_dB());
    matacqDigi.laserPower(matacq.getLaserPower());
    timeval t;
    matacq.getTimeStamp(t);
    matacqDigi.timeStamp(t);    
#endif //matacq digi version >=2
    matacqDigiCollection.push_back(matacqDigi);
    matacqDigiCollection.back().swap(samples); //swap is more efficient than a copy
  }
}

void MatacqDataFormatter::printData(ostream& out, const MatacqRawEvent& matacq) const{
  cout << "FED id: " << hex << "0x" << matacq.getFedId() << dec << "\n";
  cout << "Event id (lv1): " 
       << hex << "0x" << matacq.getEventId() << dec << "\n";
  cout << "FOV: " << hex << "0x" << matacq.getFov() << dec << "\n";
  cout << "BX id: " << hex << "0x" << matacq.getBxId() << dec << "\n";
  cout << "Trigger type: " 
       << hex << "0x" << matacq.getTriggerType() << dec << "\n";
  cout << "DCC Length: " << matacq.getDccLen() << "\n";
  cout << "Run number: " << matacq.getRunNum() << "\n";
  cout << "Field 'DCC errors': " 
       << hex << "0x" << matacq.getDccErrors() << dec << "\n";
  
  if(matacq.getStatus()){
    cout << "Error in matacq data. Errot code: "
	 << hex << "0x" << matacq.getStatus() << dec << "\n";
  }
  
  cout << "MATACQ data format version: " << matacq.getMatacqDataFormatVersion()
       << "\n";
  cout << "Sampling frequency: " << matacq.getFreqGHz() << " GHz\n";
  cout << "MATACQ channel count: " << matacq.getChannelCount() << "\n";
  time_t timeStamp = matacq.getTimeStamp();
  cout << "Data acquired on : " << ctime(&timeStamp);

  const vector<MatacqRawEvent::ChannelData>& channels = matacq.getChannelData();
  for(unsigned i=0; i< channels.size(); ++i){
    cout << "-------------------------------------- Channel "
	 << channels[i].chId
	 << ": " << setw(4) << channels[i].nSamples
	 << " samples --------------------------------------\n";
    
    for(int iSample = 0; iSample<channels[i].nSamples; ++iSample){
      MatacqRawEvent::int16le_t adc = (channels[i].samples)[iSample];
      cout << setw(4) << adc
	   << ((iSample%20==19)?"\n":" ");
    }
  }
  cout << "=================================================="
    "==================================================\n\n";   
}

