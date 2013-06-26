// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
// $Id: MatacqDataFormatter.cc,v 1.7 2008/01/22 18:59:16 muzaffar Exp $

#include "EventFilter/EcalTBRawToDigi/src/MatacqDataFormatter.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "EventFilter/EcalTBRawToDigi/src/MatacqRawEvent.h"
#include "DataFormats/EcalDigi/interface/EcalMatacqDigi.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <vector>



//#define MATACQ_DEBUG

void  MatacqTBDataFormatter::interpretRawData(const FEDRawData & data, EcalMatacqDigiCollection& matacqDigiCollection) {
#if MATACQ_DEBUG
  std::cout << "****************************************************************\n";
  std::cout << "********************** MATACQ decoder **************************\n";
  std::cout << "****************************************************************\n";
  std::cout << "FEDRawData: \n";
  char oldPad = std::cout.fill('0');
  for(int i=0; i < max(100, (int)data.size()); ++i){
    std::cout << std::hex << std::setw(2) << (int)(data.data()[i])
	 << ((i+1)%8?" ":"\n") ;
  }
  std::cout.fill(oldPad);
  std::cout << "======================================================================\n";
#endif //MATACQ_DEBUG defined
  
  MatacqTBRawEvent matacq(data.data(), data.size());

#if MATACQ_DEBUG
  printData(std::cout, matacq);
#endif //MATACQ_DEBUG defined

  const double ns = 1.e-9; //ns->s
  const double ps = 1.e-12;//ps->s
  double ts = ns/matacq.getFreqGHz();
  double tTrig = matacq.getTTrigPs()<.5*std::numeric_limits<int>::max()?
    ps*matacq.getTTrigPs():999.;
  int version = matacq.getMatacqDataFormatVersion();
  
  std::vector<int16_t> samples;
  //FIXME: the interpretRawData method should fill an EcalMatacqDigiCollection
  //instead of an EcalMatacqDigi because Matacq channels are several.
  //In the meamtime copy only the first channel appearing in data:
  const std::vector<MatacqTBRawEvent::ChannelData>& chData = matacq.getChannelData();
  for(unsigned iCh=0; iCh < chData.size(); ++iCh){
    //copy time samples into a vector:
    samples.resize(chData[iCh].nSamples);
    copy(chData[iCh].samples, chData[iCh].samples+chData[iCh].nSamples,
	 samples.begin());
    int chId = chData[iCh].chId;
    std::vector<int16_t> empty;
    EcalMatacqDigi matacqDigi(empty, chId, ts, version, tTrig);
    matacqDigiCollection.push_back(matacqDigi);
    matacqDigiCollection.back().swap(samples); //swap is more efficient than a copy
  }
}

void MatacqTBDataFormatter::printData(std::ostream& out, const MatacqTBRawEvent& matacq) const{
  std::cout << "FED id: " << std::hex << "0x" << matacq.getFedId() << std::dec << "\n";
  std::cout << "Event id (lv1): " 
       << std::hex << "0x" << matacq.getEventId() << std::dec << "\n";
  std::cout << "FOV: " << std::hex << "0x" << matacq.getFov() << std::dec << "\n";
  std::cout << "BX id: " << std::hex << "0x" << matacq.getBxId() << std::dec << "\n";
  std::cout << "Trigger type: " 
       << std::hex << "0x" << matacq.getTriggerType() << std::dec << "\n";
  std::cout << "DCC Length: " << matacq.getDccLen() << "\n";
  std::cout << "Run number: " << matacq.getRunNum() << "\n";
  std::cout << "Field 'DCC errors': " 
       << std::hex << "0x" << matacq.getDccErrors() << std::dec << "\n";
  
  if(matacq.getStatus()){
    std::cout << "Error in matacq data. Errot code: "
	 << std::hex << "0x" << matacq.getStatus() << std::dec << "\n";
  }
  
  std::cout << "MATACQ data format version: " << matacq.getMatacqDataFormatVersion()
       << "\n";
  std::cout << "Sampling frequency: " << matacq.getFreqGHz() << " GHz\n";
  std::cout << "MATACQ channel count: " << matacq.getChannelCount() << "\n";
  time_t timeStamp = matacq.getTimeStamp();
  std::cout << "Data acquired on : " << ctime(&timeStamp);

  const std::vector<MatacqTBRawEvent::ChannelData>& channels = matacq.getChannelData();
  for(unsigned i=0; i< channels.size(); ++i){
    std::cout << "-------------------------------------- Channel "
	 << channels[i].chId
	 << ": " << std::setw(4) << channels[i].nSamples
	 << " samples --------------------------------------\n";
    
    for(int iSample = 0; iSample<channels[i].nSamples; ++iSample){
      MatacqTBRawEvent::int16le_t adc = (channels[i].samples)[iSample];
      std::cout << std::setw(4) << adc
	   << ((iSample%20==19)?"\n":" ");
    }
  }
  std::cout << "=================================================="
    "==================================================\n\n";   
}
