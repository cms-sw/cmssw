#include "PhysicsTools/PatAlgos/interface/PATPrimaryVertexSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

PATPrimaryVertexSelector::PATPrimaryVertexSelector (const edm::ParameterSet& cfg, edm::ConsumesCollector && iC) :
  multiplicityCut_(cfg.getParameter<unsigned int>("minMultiplicity")),
  ptSumCut_(cfg.getParameter<double>("minPtSum")),
  trackEtaCut_(cfg.getParameter<double>("maxTrackEta")),
  chi2Cut_(cfg.getParameter<double>("maxNormChi2")),
  dr2Cut_(cfg.getParameter<double>("maxDeltaR")),
  dzCut_(cfg.getParameter<double>("maxDeltaZ")) {
  // store squared cut (avoid using sqrt() for each vertex)
  dr2Cut_ *= dr2Cut_;
}

void
PATPrimaryVertexSelector::select (const edm::Handle<collection>& handle,
				  const edm::Event& event,
				  const edm::EventSetup& setup) {
  selected_.clear();
  // FIXME: the fixed reference point should be replaced with the measured beamspot
  const math::XYZPoint beamSpot(0.,0.,0.);
  unsigned int multiplicity;
  double ptSum;
  for ( collection::const_iterator iv=handle->begin(); iv!=handle->end(); ++iv ) {
    math::XYZVector displacement(iv->position()-beamSpot);
    if ( iv->normalizedChi2()<chi2Cut_ &&
	 fabs(displacement.z())<dzCut_ && displacement.perp2()<dr2Cut_ ) {
      getVertexVariables(*iv,multiplicity,ptSum);
      if ( multiplicity>=multiplicityCut_ && ptSum>ptSumCut_ ) selected_.push_back(&*iv);
    }
  }
  sort(selected_.begin(),selected_.end(),*this);
}

bool
PATPrimaryVertexSelector::operator() (const reco::Vertex* v1, const reco::Vertex* v2) const {
  unsigned int mult1;
  double ptSum1;
  getVertexVariables(*v1,mult1,ptSum1);
  unsigned int mult2;
  double ptSum2;
  getVertexVariables(*v2,mult2,ptSum2);
  return ptSum1>ptSum2;
}

void
PATPrimaryVertexSelector::getVertexVariables (const reco::Vertex& vertex,
					      unsigned int& multiplicity, double& ptSum) const {
  multiplicity = 0;
  ptSum = 0.;
  for(reco::Vertex::trackRef_iterator it=vertex.tracks_begin();
      it!=vertex.tracks_end(); ++it) {
    if(fabs((**it).eta())<trackEtaCut_) {
      ++multiplicity;
      ptSum += (**it).pt();
    }
  }
}
