//
// $Id: PATConversionProducer.cc,v 1.1 2012/04/14 02:12:39 tjkim Exp $
//
#include "PhysicsTools/PatAlgos/plugins/PATConversionProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "PhysicsTools/PatUtils/interface/TrackerIsolationPt.h"
#include "PhysicsTools/PatUtils/interface/CaloIsolationEnergy.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/PatCandidates/interface/Conversion.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include <vector>
#include <memory>
#include "TMath.h"

using namespace pat;
using namespace std;


PATConversionProducer::PATConversionProducer(const edm::ParameterSet & iConfig){

  // general configurables
  electronSrc_      = iConfig.getParameter<edm::InputTag>( "electronSource" );

  // produces vector of muons
  produces<std::vector<Conversion> >();

}


PATConversionProducer::~PATConversionProducer() {
}


void PATConversionProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // Get the collection of electrons from the event
  edm::Handle<edm::View<reco::GsfElectron> > electrons;
  iEvent.getByLabel(electronSrc_, electrons);

  edm::Handle<reco::BeamSpot> bsHandle;
  iEvent.getByLabel("offlineBeamSpot", bsHandle);
  const reco::BeamSpot &beamspot = *bsHandle.product();

  // for conversion veto selection  
  edm::Handle<reco::ConversionCollection> hConversions;
  iEvent.getByLabel("allConversions", hConversions);

  std::vector<Conversion> * patConversions = new std::vector<Conversion>();

  for (reco::ConversionCollection::const_iterator conv = hConversions->begin(); conv!= hConversions->end(); ++conv) {

    reco::Vertex vtx = conv->conversionVertex();

    int index = 0; 
    for (edm::View<reco::GsfElectron>::const_iterator itElectron = electrons->begin(); itElectron != electrons->end(); ++itElectron) {

      //find matched conversions with electron and save those conversions with matched electron index
      if (ConversionTools::matchesConversion(*itElectron, *conv)) {

        double vtxProb = TMath::Prob( vtx.chi2(), vtx.ndof());
        math::XYZVector mom(conv->refittedPairMomentum());
        double dbsx = vtx.x() - beamspot.position().x();   
        double dbsy = vtx.y() - beamspot.position().y();
        double lxy = (mom.x()*dbsx + mom.y()*dbsy)/mom.rho();
        int nHitsMax = 0;

        for (std::vector<uint8_t>::const_iterator it = conv->nHitsBeforeVtx().begin(); it!=conv->nHitsBeforeVtx().end(); ++it) {
          if ((*it) > nHitsMax) nHitsMax = (*it);
        }

        pat::Conversion anConversion( index ); 
        anConversion.setVtxProb( vtxProb );
        anConversion.setLxy( lxy );
        anConversion.setNHitsMax(  nHitsMax );

        patConversions->push_back(anConversion);
        break;
      }
      index++;
    }
    
  }

  // add the electrons to the event output
  std::auto_ptr<std::vector<Conversion> > ptr(patConversions);
  iEvent.put(ptr);

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATConversionProducer);
