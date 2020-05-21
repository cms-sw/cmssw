/** 
 * Demo analyzer for reading digis
 * author A.Tumanov 2/22/06 
 * Code updated but not tested - needs updated config file - 10.09.2013 Tim Cox
 *   
 */

#include "EventFilter/CSCRawToDigi/interface/DigiAnalyzer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUHeader.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
    auto digiItr = (*j).second.first;
    auto last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCDDUStatusDigiCollection::DigiRangeIterator j = dduStatusDigi->begin(); j != dduStatusDigi->end(); j++) {
    auto digiItr = (*j).second.first;
    auto last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      CSCDDUHeader header(*digiItr);
      std::cout << "L1 number = " << header.lvl1num() << std::endl;
      std::cout << "DDU number = " << header.source_id() << std::endl;
    }
  }

  for (CSCStripDigiCollection::DigiRangeIterator j = strips->begin(); j != strips->end(); j++) {
    auto digiItr = (*j).second.first;
    CSCDetId const cscDetId = (*j).first;
    std::cout << cscDetId << std::endl;
    auto last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCWireDigiCollection::DigiRangeIterator j = wires->begin(); j != wires->end(); j++) {
    CSCDetId const cscDetId = (*j).first;
    std::cout << cscDetId << std::endl;
    auto digiItr = (*j).second.first;
    auto last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCComparatorDigiCollection::DigiRangeIterator j = comparators->begin(); j != comparators->end(); j++) {
    auto digiItr = (*j).second.first;
    auto last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCALCTDigiCollection::DigiRangeIterator j = alcts->begin(); j != alcts->end(); j++) {
    auto digiItr = (*j).second.first;
    auto last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCCLCTDigiCollection::DigiRangeIterator j = clcts->begin(); j != clcts->end(); j++) {
    auto digiItr = (*j).second.first;
    auto last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCRPCDigiCollection::DigiRangeIterator j = rpcs->begin(); j != rpcs->end(); j++) {
    auto digiItr = (*j).second.first;
    auto last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator j = correlatedlcts->begin(); j != correlatedlcts->end(); j++) {
    auto digiItr = (*j).second.first;
    auto last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  eventNumber++;
  edm::LogInfo("DigiAnalyzer") << "end of event number " << eventNumber;
}
