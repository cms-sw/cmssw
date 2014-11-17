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
#include "FWCore/Framework/interface/EDProducer.h"
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

using namespace std;
using namespace edm;


//
// class declaration
//

  class L1TCaloUpgradeToGCTConverter : public EDProducer {
  public:
    explicit L1TCaloUpgradeToGCTConverter(const ParameterSet&);
    ~L1TCaloUpgradeToGCTConverter();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void produce(Event&, EventSetup const&) override;
    virtual void beginJob();
    virtual void endJob();
    virtual void beginRun(Run const&iR, EventSetup const&iE);
    virtual void endRun(Run const& iR, EventSetup const& iE);

    EDGetToken EGammaToken_;
    EDGetToken RlxTauToken_;
    EDGetToken IsoTauToken_;
    EDGetToken JetToken_;
    EDGetToken EtSumToken_;
    EDGetToken HfSumsToken_;
    EDGetToken HfCountsToken_;
  };

#endif
