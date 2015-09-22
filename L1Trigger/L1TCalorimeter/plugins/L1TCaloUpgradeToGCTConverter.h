#ifndef L1TCaloUpgradeToGCTConverter_h
#define L1TCaloUpgradeToGCTConverter_h

///
/// \class l1t::L1TCaloUpgradeToGCTConverter
///
/// Description: Emulator for the stage 1 jet algorithms.
///
///
/// \author: Ivan Amos Cali MIT
///


// system include files
#include <boost/shared_ptr.hpp>

// user include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/CaloSpare.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include <vector>



//
// class declaration
//

class L1TCaloUpgradeToGCTConverter : public edm::global::EDProducer<> {
  public:
  explicit L1TCaloUpgradeToGCTConverter(const edm::ParameterSet&);
    ~L1TCaloUpgradeToGCTConverter();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    virtual void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;    

    const edm::EDGetToken EGammaToken_;
    const edm::EDGetToken RlxTauToken_;
    const edm::EDGetToken IsoTauToken_;
    const edm::EDGetToken JetToken_;
    const edm::EDGetToken EtSumToken_;
    const edm::EDGetToken HfSumsToken_;
    const edm::EDGetToken HfCountsToken_;

    const int bxMin_;
    const int bxMax_;

  };

#endif
