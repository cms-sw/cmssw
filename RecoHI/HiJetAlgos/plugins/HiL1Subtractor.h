#ifndef RecoJets_JetProducers_plugins_HiL1Subtractor_h
#define RecoJets_JetProducers_plugins_HiL1Subtractor_h

// -*- C++ -*-
//
// Package:    HiL1Subtractor
// Class:      HiL1Subtractor
//
/**\class HiL1Subtractor HiL1Subtractor.cc RecoHI/HiJetAlgos/plugins/HiL1Subtractor.cc

 Description:

  Implementation:

*/
//
// Original Author:  "Matthew Nguyen"
//         Created:  Sun Nov 7 12:18:18 CDT 2010
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

//
// class declaration
//

class HiL1Subtractor : public edm::global::EDProducer<> {
protected:
  //
  // typedefs & structs
  //

public:
  explicit HiL1Subtractor(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  // input jet source
  edm::EDGetTokenT<edm::View<reco::GenJet> > genJetSrc_;
  edm::EDGetTokenT<edm::View<reco::CaloJet> > caloJetSrc_;
  edm::EDGetTokenT<edm::View<reco::PFJet> > pfJetSrc_;

protected:
  std::string jetType_;       // Type of jet
  std::string rhoTagString_;  // Algorithm for rho estimation

  edm::EDGetTokenT<std::vector<double> > rhoTag_;
};

#endif
