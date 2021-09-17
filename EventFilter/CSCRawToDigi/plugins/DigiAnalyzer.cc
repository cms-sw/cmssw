/**
 * Demo analyzer for reading digis
 * author A.Tumanov 2/22/06
 * Updated 10.09.2013 but untested - Tim Cox
 *
 */

#include <iostream>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigiCollection.h"

#include "EventFilter/CSCRawToDigi/interface/CSCDDUHeader.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"

class DigiAnalyzer : public edm::EDAnalyzer {
public:
  explicit DigiAnalyzer(edm::ParameterSet const& conf);
  void analyze(edm::Event const& e, edm::EventSetup const& iSetup) override;

private:
  int eventNumber;

  edm::EDGetTokenT<CSCWireDigiCollection> wd_token;
  edm::EDGetTokenT<CSCStripDigiCollection> sd_token;
  edm::EDGetTokenT<CSCComparatorDigiCollection> cd_token;
  edm::EDGetTokenT<CSCALCTDigiCollection> al_token;
  edm::EDGetTokenT<CSCCLCTDigiCollection> cl_token;
  edm::EDGetTokenT<CSCRPCDigiCollection> rd_token;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> co_token;
  edm::EDGetTokenT<CSCDDUStatusDigiCollection> dd_token;
  edm::EDGetTokenT<CSCDCCFormatStatusDigiCollection> dc_token;
};

DigiAnalyzer::DigiAnalyzer(edm::ParameterSet const& conf) {
  eventNumber = 0;

  wd_token = consumes<CSCWireDigiCollection>(conf.getParameter<edm::InputTag>("wireDigiTag"));
  sd_token = consumes<CSCStripDigiCollection>(conf.getParameter<edm::InputTag>("stripDigiTag"));
  cd_token = consumes<CSCComparatorDigiCollection>(conf.getParameter<edm::InputTag>("compDigiTag"));
  al_token = consumes<CSCALCTDigiCollection>(conf.getParameter<edm::InputTag>("alctDigiTag"));
  cl_token = consumes<CSCCLCTDigiCollection>(conf.getParameter<edm::InputTag>("clctDigiTag"));
  rd_token = consumes<CSCRPCDigiCollection>(conf.getParameter<edm::InputTag>("rpcDigiTag"));
  co_token = consumes<CSCCorrelatedLCTDigiCollection>(conf.getParameter<edm::InputTag>("corrlctDigiTag"));
  dd_token = consumes<CSCDDUStatusDigiCollection>(conf.getParameter<edm::InputTag>("ddusDigiTag"));
  dc_token = consumes<CSCDCCFormatStatusDigiCollection>(conf.getParameter<edm::InputTag>("dccfDigiTag"));
}

void DigiAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCStripDigiCollection> strips;
  edm::Handle<CSCComparatorDigiCollection> comparators;
  edm::Handle<CSCALCTDigiCollection> alcts;
  edm::Handle<CSCCLCTDigiCollection> clcts;
  edm::Handle<CSCRPCDigiCollection> rpcs;
  edm::Handle<CSCCorrelatedLCTDigiCollection> correlatedlcts;
  edm::Handle<CSCDDUStatusDigiCollection> dduStatusDigi;
  edm::Handle<CSCDCCFormatStatusDigiCollection> formatStatusDigi;

  e.getByToken(dd_token, dduStatusDigi);
  e.getByToken(wd_token, wires);
  e.getByToken(sd_token, strips);
  e.getByToken(cd_token, comparators);
  e.getByToken(al_token, alcts);
  e.getByToken(cl_token, clcts);
  e.getByToken(rd_token, rpcs);
  e.getByToken(co_token, correlatedlcts);
  e.getByToken(dc_token, formatStatusDigi);

  for (CSCDCCFormatStatusDigiCollection::DigiRangeIterator j = formatStatusDigi->begin(); j != formatStatusDigi->end();
       j++) {
    std::vector<CSCDCCFormatStatusDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCDCCFormatStatusDigi>::const_iterator last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCDDUStatusDigiCollection::DigiRangeIterator j = dduStatusDigi->begin(); j != dduStatusDigi->end(); j++) {
    std::vector<CSCDDUStatusDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCDDUStatusDigi>::const_iterator last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      CSCDDUHeader header(*digiItr);
      std::cout << "L1 number = " << header.lvl1num() << std::endl;
      std::cout << "DDU number = " << header.source_id() << std::endl;
    }
  }

  for (CSCStripDigiCollection::DigiRangeIterator j = strips->begin(); j != strips->end(); j++) {
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    CSCDetId const cscDetId = (*j).first;
    std::cout << cscDetId << std::endl;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCWireDigiCollection::DigiRangeIterator j = wires->begin(); j != wires->end(); j++) {
    CSCDetId const cscDetId = (*j).first;
    std::cout << cscDetId << std::endl;
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCComparatorDigiCollection::DigiRangeIterator j = comparators->begin(); j != comparators->end(); j++) {
    std::vector<CSCComparatorDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCComparatorDigi>::const_iterator last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCALCTDigiCollection::DigiRangeIterator j = alcts->begin(); j != alcts->end(); j++) {
    std::vector<CSCALCTDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCALCTDigi>::const_iterator last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCCLCTDigiCollection::DigiRangeIterator j = clcts->begin(); j != clcts->end(); j++) {
    std::vector<CSCCLCTDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCCLCTDigi>::const_iterator last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCRPCDigiCollection::DigiRangeIterator j = rpcs->begin(); j != rpcs->end(); j++) {
    std::vector<CSCRPCDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCRPCDigi>::const_iterator last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator j = correlatedlcts->begin(); j != correlatedlcts->end(); j++) {
    std::vector<CSCCorrelatedLCTDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCCorrelatedLCTDigi>::const_iterator last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  eventNumber++;
  edm::LogInfo("DigiAnalyzer") << "end of event number " << eventNumber;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DigiAnalyzer);
