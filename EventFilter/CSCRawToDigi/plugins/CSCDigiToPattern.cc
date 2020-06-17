/** \file
 *
 *  \author A. Tumanov - Rice
 */

#include <iostream>

#include <FWCore/Framework/interface/ConsumesCollector.h>
#include <FWCore/Utilities/interface/InputTag.h>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

class CSCDigiToPattern : public edm::EDAnalyzer {
public:
  explicit CSCDigiToPattern(edm::ParameterSet const& conf);
  void analyze(edm::Event const& e, edm::EventSetup const& iSetup) override;

  //virtual void endJob();
private:
  // variables persistent across events should be declared here.
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> d_token;
  //
};

CSCDigiToPattern::CSCDigiToPattern(edm::ParameterSet const& conf) {
  d_token = consumes<CSCCorrelatedLCTDigiCollection>(conf.getParameter<edm::InputTag>("corrlctDigiTag"));
}

void CSCDigiToPattern::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  // These declarations create handles to the types of records that you want
  // to retrieve from event "e".
  //
  edm::Handle<CSCCorrelatedLCTDigiCollection> correlatedlcts;
  e.getByToken(d_token, correlatedlcts);

  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator j = correlatedlcts->begin(); j != correlatedlcts->end(); j++) {
    CSCDetId id = (*j).first;
    std::cout << id << std::endl;
    std::vector<CSCCorrelatedLCTDigi>::const_iterator digiIt = (*j).second.first;
    std::vector<CSCCorrelatedLCTDigi>::const_iterator last = (*j).second.second;
    for (; digiIt != last; ++digiIt) {
      uint16_t wire = digiIt->getKeyWG();       // 7 bits
      uint16_t pattern = digiIt->getPattern();  // 4 bits
      uint16_t quality = digiIt->getQuality();  // 4 bits
      uint16_t valid = digiIt->isValid();       // 1 bit
      uint16_t strip = digiIt->getStrip();      // 8 bits
      uint16_t bend = digiIt->getBend();        // 1 bit
      uint16_t syncErr = digiIt->getSyncErr();  // 1 bit
      uint16_t bx = digiIt->getBX();            // 1 bit
      uint16_t bx0 = digiIt->getBX0();          // 1 bit
      uint16_t cscId = digiIt->getCSCID();      // 4 bits
      //                                             __
      //                                             32 bits in total
      long unsigned int mpc = ((cscId & 0xF) << 28) | ((bx0 & 0x1) << 27) | ((bx & 0x1) << 26) |
                              ((syncErr & 0x1) << 25) | ((bend & 0x1) << 24) | ((strip & 0xFF) << 16) |
                              ((valid & 0x1) << 15) | ((quality & 0xF) << 11) | ((pattern & 0xF) << 7) | (wire & 0x7F);
      std::cout << "MPC" << digiIt->getTrknmb() << " " << std::hex << mpc << std::dec << std::endl;
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CSCDigiToPattern);
