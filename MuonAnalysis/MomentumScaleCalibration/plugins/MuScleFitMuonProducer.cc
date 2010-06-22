// -*- C++ -*-
//
// Package:    MuScleFitMuonProducer
// Class:      MuScleFitMuonProducer
// 
/**\class MuScleFitMuonProducer MuScleFitMuonProducer.cc MuonAnalysis/MuScleFitMuonProducer/src/MuScleFitMuonProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco De Mattia,40 3-B32,+41227671551,
//         Created:  Tue Jun 22 13:50:22 CEST 2010
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"

class MuScleFitMuonProducer : public edm::EDProducer {
   public:
      explicit MuScleFitMuonProducer(const edm::ParameterSet&);
      ~MuScleFitMuonProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  edm::InputTag theMuonLabel_;
  smearFunctionType7 smearFunction;
};

MuScleFitMuonProducer::MuScleFitMuonProducer(const edm::ParameterSet& iConfig) :
  theMuonLabel_( iConfig.getParameter<edm::InputTag>( "MuonLabel" ) )
{
  produces<reco::MuonCollection>();
}


MuScleFitMuonProducer::~MuScleFitMuonProducer()
{
}

// ------------ method called to produce the data  ------------
void MuScleFitMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<reco::MuonCollection> allMuons;
  iEvent.getByLabel (theMuonLabel_, allMuons);

  std::auto_ptr<reco::MuonCollection> pOut(new reco::MuonCollection);

  // Apply any bias and/or smearing to the events
  for( std::vector<reco::Muon>::const_iterator muon = allMuons->begin(); muon != allMuons->end(); ++muon ) {

    double pt = muon->pt();

    // Bias
    double a_0 = 1.0019;
    double a_1 = -0.0004;
    // The miscalibration can be applied only for pt < 626 GeV/c
    if( pt < -a_0*a_0/(4*a_1) ) {
      pt = (-a_0 + sqrt(a_0*a_0 + 4*a_1*pt))/(2*a_1);
    }

    // Smearing
    // std::cout << "smearing muon" << std::endl;
    std::vector<double> par;
    double * y = 0;
    // double y[7];
    double eta = muon->eta();
    double phi = muon->phi();
    smearFunction.smear(pt, eta, phi, y, par);

    reco::Muon * newMuon = muon->clone();
    newMuon->setP4( reco::Particle::PolarLorentzVector( pt, eta, phi, muon->mass() ) );

    pOut->push_back(*newMuon);
  }

  // put into the Event
  // std::auto_ptr<reco::MuonCollection> pOut(new reco::MuonCollection(*allMuons));
  iEvent.put(pOut);
}

// ------------ method called once each job just before starting event loop  ------------
void 
MuScleFitMuonProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuScleFitMuonProducer::endJob()
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuScleFitMuonProducer);
