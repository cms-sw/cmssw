// Propagate L1Tracks to ECAL Entrance
// T. Ruggles
// 17 May 2017
//

//#include "FastSimulation/Particle/interface/RawParticle.h"
//#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"



#ifndef L1TkElectronTrackMatchAlgo_HH
#define L1TkElectronTrackMatchAlgo_HH

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
//#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"


namespace L1TkElectronTrackMatchAlgo {
   typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >    L1TkTrackCollectionType ;
  void doMatch(l1extra::L1EmParticleCollection::const_iterator egIter, const edm::Ptr< L1TkTrackType >& pTrk, double&  dph, double&  dr, double& deta);
  void doMatch(const GlobalPoint& epos, const edm::Ptr< L1TkTrackType >& pTrk, double& dph, double&  dr, double& deta);

  double deltaR(const GlobalPoint& epos, const edm::Ptr< L1TkTrackType >& pTrk);
  double deltaPhi(const GlobalPoint& epos, const edm::Ptr< L1TkTrackType >& pTrk);
  double deltaEta(const GlobalPoint& epos, const edm::Ptr< L1TkTrackType >& pTrk);
  GlobalPoint calorimeterPosition(double phi, double eta, double e);

}  
#endif
//
//
//
//
//
//
//
//
//
//         // Get the particle position upon entering ECal
//         RawParticle particle(genParticles[0].p4());
//         particle.setVertex(genParticles[0].vertex().x(), genParticles[0].vertex().y(), genParticles[0].vertex().z(), 0.);
//         //particle.setID(genParticles[0].pdgId());
//         // Skip setID requires some external libraries working well that
//         // define HepPDT::ParticleID
//         // in the end, setID sets the mass and charge of our particle.
//         // Try doing this by hand for the moment
//         particle.setMass(.511);
//         int pdgId = genParticles[0].pdgId();
//         if (pdgId > 0) {
//            particle.setCharge( -1.0 ); }
//         if (pdgId < 0) {
//            particle.setCharge( 1.0 ); }
//         BaseParticlePropagator prop(particle, 0., 0., 4.);
//         BaseParticlePropagator start(prop);
//         prop.propagateToEcalEntrance();
//         if(prop.getSuccess()!=0)
//         {
//            trueElectron = reco::Candidate::PolarLorentzVector(prop.E()*sin(prop.vertex().theta()), prop.vertex().eta(), prop.vertex().phi(), 0.);
//            if ( debug ) std::cout << "Propogated genParticle to ECal, position: " << prop.vertex() << " momentum = " << prop.momentum() << std::endl;
//            if ( debug ) std::cout << "                       starting position: " << start.vertex() << " momentum = " << start.momentum() << std::endl;
//            if ( debug ) std::cout << "                    genParticle position: " << genParticles[0].vertex() << " momentum = " << genParticles[0].p4() << std::endl;
//            if ( debug ) std::cout << "       old pt = " << genParticles[0].pt() << ", new pt = " << trueElectron.pt() << std::endl;
//         }
//         else
//         {
//            // something failed?
//            trueElectron = genParticles[0].polarP4();
//         }
