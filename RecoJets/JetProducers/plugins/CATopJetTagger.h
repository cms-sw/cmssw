#ifndef TopQuarkAnalysis_TopJetProducers_interface_CATopJetTagger_h
#define TopQuarkAnalysis_TopJetProducers_interface_CATopJetTagger_h

// -*- C++ -*-
//
// Package:    CATopJetTagger
// Class:      CATopJetTagger
// 
/**\class CATopJetTagger CATopJetTagger.cc TopQuarkAnalysis/TopJetProducers/src/CATopJetTagger.cc

 Description: This is a tagger to identify boosted top quark jets.

 Implementation: We input the jets from CATopJetProducer and make kinematic
                 cuts on the jet mass and the minimum invariant mass pairing
                 of the subjets. 
		 Described in "Top-tagging: A Method for Identifying Boosted Hadronic Tops"
		 David E. Kaplan, Keith Rehermann, Matthew D. Schwartz, Brock Tweedie
		 arXiv:0806.0848v1 [hep-ph] 

 Produces:       A list of pair<Jet,CATopJetTagInfo> that represents the tag decision.

*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Thu Jul  3 00:19:30 CDT 2008
//
//


// system include files
#include <memory>
#include <vector>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

#include <Math/VectorUtil.h>
#include <TH1.h>
#include <TH2.h>
#include <TTree.h>


//
// class decleration
//

class CATopJetTagger : public edm::global::EDProducer<> {
 public:
  explicit CATopJetTagger(const edm::ParameterSet&);
  ~CATopJetTagger();
  
  
 private:
  virtual void produce( edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
  // ----------member data ---------------------------
  
  const edm::InputTag   src_;
  
  const double      TopMass_;
  const double      WMass_;
  const bool        verbose_;

  const edm::EDGetTokenT<edm::View<reco::Jet> > input_jet_token_;

};



#endif
