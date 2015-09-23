// -*- C++ -*-
//
// Package:    PatShapeAna
// Class:      PatShapeAna
//
/**\class PatShapeAna PatShapeAna.h PhysicsTools/PatShapeAna/interface/PatShapeAna.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Christian AUTERMANN
//         Created:  Sat Mar 22 12:58:04 CET 2008
//
//

#ifndef PATHemisphereProducer_h
#define PATHemisphereProducer_h

// system include files
#include <memory>
#include <map>
#include <utility>//pair
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

#include "PhysicsTools/PatAlgos/interface/HemisphereAlgo.h"
//
// class decleration
//

class PATHemisphereProducer : public edm::global::EDProducer<> {
public:
  explicit PATHemisphereProducer(const edm::ParameterSet&);
  ~PATHemisphereProducer();
  
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
private:  
  // ----------member data ---------------------------
  /// Input: All PAT objects that are to cross-clean  or needed for that
  const edm::EDGetTokenT<reco::CandidateView> _patJetsToken;
  //       edm::EDGetTokenT<reco::CandidateView> _patMetsToken;
  const edm::EDGetTokenT<reco::CandidateView> _patMuonsToken;
  const edm::EDGetTokenT<reco::CandidateView> _patElectronsToken;
  const edm::EDGetTokenT<reco::CandidateView> _patPhotonsToken;
  const edm::EDGetTokenT<reco::CandidateView> _patTausToken;
  
  const float _minJetEt;
  const float _minMuonEt;
  const float _minElectronEt;
  const float _minTauEt;
  const float _minPhotonEt;
  
  const float _maxJetEta;
  const float _maxMuonEta;
  const float _maxElectronEta;
  const float _maxTauEta;
  const float _maxPhotonEta;
  
  const int _seedMethod;
  const int _combinationMethod;
  
  typedef std::vector<float> HemiAxis;
};

#endif


