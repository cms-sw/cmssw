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

  wd_token = consumes<CSCWireDigiCollection>( conf.getParameter<edm::InputTag>("wireDigiTag") );
  sd_token = consumes<CSCStripDigiCollection>( conf.getParameter<edm::InputTag>("stripDigiTag") );
  cd_token = consumes<CSCComparatorDigiCollection>( conf.getParameter<edm::InputTag>("compDigiTag") );
  al_token = consumes<CSCALCTDigiCollection>( conf.getParameter<edm::InputTag>("alctDigiTag") );
  cl_token = consumes<CSCCLCTDigiCollection>( conf.getParameter<edm::InputTag>("clctDigiTag") );
  rd_token = consumes<CSCRPCDigiCollection>( conf.getParameter<edm::InputTag>("rpcDigiTag") );
  co_token = consumes<CSCCorrelatedLCTDigiCollection>( conf.getParameter<edm::InputTag>("corrlctDigiTag") );
  dd_token = consumes<CSCDDUStatusDigiCollection>( conf.getParameter<edm::InputTag>("ddusDigiTag") );
  dc_token = consumes<CSCDCCFormatStatusDigiCollection>( conf.getParameter<edm::InputTag>("dccfDigiTag") );

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

  e.getByToken( dd_token, dduStatusDigi );
  e.getByToken( wd_token, wires );
  e.getByToken( sd_token, strips );
  e.getByToken( cd_token, comparators );
  e.getByToken( al_token, alcts );
  e.getByToken( cl_token, clcts );
  e.getByToken( rd_token, rpcs );
  e.getByToken( co_token, correlatedlcts );
  e.getByToken( dc_token, formatStatusDigi );

  for (auto && j : *formatStatusDigi) {
    std::vector<CSCDCCFormatStatusDigi>::const_iterator digiItr = j.second.first;
    std::vector<CSCDCCFormatStatusDigi>::const_iterator last = j.second.second;
    for( ; digiItr != last; ++digiItr) {
	digiItr->print();
    }
  }


  for (auto && j : *dduStatusDigi) {
    std::vector<CSCDDUStatusDigi>::const_iterator digiItr = j.second.first;
    std::vector<CSCDDUStatusDigi>::const_iterator last = j.second.second;
    for( ; digiItr != last; ++digiItr) {
      CSCDDUHeader header(*digiItr);
      std::cout <<"L1 number = " << header.lvl1num() << std::endl; 
      std::cout <<"DDU number = " << header.source_id() << std::endl;
    }
  }

  for (auto && j : *strips) {
    std::vector<CSCStripDigi>::const_iterator digiItr = j.second.first;
    CSCDetId const cscDetId=j.first;
    std::cout<<cscDetId<<std::endl;
    std::vector<CSCStripDigi>::const_iterator last = j.second.second;
    for( ; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (auto && j : *wires) {
    CSCDetId const cscDetId=j.first;
    std::cout<<cscDetId<<std::endl;
    std::vector<CSCWireDigi>::const_iterator digiItr = j.second.first;
    std::vector<CSCWireDigi>::const_iterator last = j.second.second;
    for( ; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (auto && j : *comparators) {
 
    std::vector<CSCComparatorDigi>::const_iterator digiItr = j.second.first;
    std::vector<CSCComparatorDigi>::const_iterator last = j.second.second;
    for( ; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (auto && j : *alcts) {
 
    std::vector<CSCALCTDigi>::const_iterator digiItr = j.second.first;
    std::vector<CSCALCTDigi>::const_iterator last = j.second.second;
    for( ; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (auto && j : *clcts) {
 
    std::vector<CSCCLCTDigi>::const_iterator digiItr = j.second.first;
    std::vector<CSCCLCTDigi>::const_iterator last = j.second.second;
    for( ; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (auto && j : *rpcs) {
 
    std::vector<CSCRPCDigi>::const_iterator digiItr = j.second.first;
    std::vector<CSCRPCDigi>::const_iterator last = j.second.second;
    for( ; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  for (auto && j : *correlatedlcts) {
 
    std::vector<CSCCorrelatedLCTDigi>::const_iterator digiItr = j.second.first;
    std::vector<CSCCorrelatedLCTDigi>::const_iterator last = j.second.second;
    for( ; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  eventNumber++;
  edm::LogInfo ("DigiAnalyzer")  << "end of event number " << eventNumber;

}





