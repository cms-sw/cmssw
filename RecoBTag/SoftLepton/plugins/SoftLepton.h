#ifndef RecoBTag_SoftLepton_SoftLepton_h
#define RecoBTag_SoftLepton_SoftLepton_h

// -*- C++ -*-
//
// Package:    SoftLepton
// Class:      SoftLepton
// 
/**\class SoftLepton SoftLepton.h RecoBTag/SoftLepton/plugin/SoftLepton.h

 Description: CMSSW EDProducer wrapper for sot lepton b tagging.

 Implementation:
     The actual tagging is performed by SoftLeptonAlgorithm.
*/
//
// Original Author:  fwyzard
//         Created:  Wed Oct 18 18:02:07 CEST 2006
// $Id: SoftLepton.h,v 1.1 2007/08/17 23:10:10 fwyzard Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SoftLeptonAlgorithm.h"

class edm::EventSetup;
class edm::Event;

class SoftLepton : public edm::EDProducer {
public:
  explicit SoftLepton(const edm::ParameterSet& iConfig);
  ~SoftLepton();

private:
  virtual void beginJob(const edm::EventSetup& iSetup);
  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void endJob(void);

  // configuration   
  const edm::InputTag m_jets;
  const edm::InputTag m_primaryVertex;
  const edm::InputTag m_leptons;

  // concrete algorithm
  SoftLeptonAlgorithm m_algo;

  // quality cuts
  double m_quality;

  // nominal beam spot position
  static const reco::Vertex s_nominalBeamSpot;
};

#endif // RecoBTag_SoftLepton_SoftLepton_h
