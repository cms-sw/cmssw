#ifndef L1TPhysicalEtAdder_h
#define L1TPhysicalEtAdder_h

// Original Author:  Alex Barbieri
//
// This class adds physical values of eta, phi, and pt to the L1 Dataformats


// system include files
#include <memory>

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

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/CaloSpare.h"

//
// class declaration
//


  class L1TPhysicalEtAdder : public edm::EDProducer {
  public:
    explicit L1TPhysicalEtAdder(const edm::ParameterSet& ps);
    ~L1TPhysicalEtAdder();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
      virtual void beginJob() override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------

      edm::EDGetToken EGammaToken_;
      edm::EDGetToken RlxTauToken_;
      edm::EDGetToken IsoTauToken_;
      edm::EDGetToken JetToken_;
      edm::EDGetToken preGtJetToken_;
      edm::EDGetToken EtSumToken_;
      edm::EDGetToken HfSumsToken_;
      edm::EDGetToken HfCountsToken_;
  };


#endif
