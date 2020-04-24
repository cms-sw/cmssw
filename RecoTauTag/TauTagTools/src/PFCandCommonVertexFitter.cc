#include "RecoTauTag/TauTagTools/interface/PFCandCommonVertexFitter.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <sstream>
using namespace reco;
using namespace std;

void PFCandCommonVertexFitterBase::set(VertexCompositeCandidate & c) const {
  if(bField_ == nullptr)
    throw edm::Exception(edm::errors::InvalidReference)
      << "B-Field was not set up PFCandCommonVertexFitter.\n"
      << "the following method must be called before fitting a candidate:\n"
      << " PFCandCommonVertexFitter:.set( const MagneticField * )" << endl;
  std::vector<TransientTrack> tracks;
  std::vector<Candidate *> daughters;
  std::vector<RecoCandidate::TrackType> trackTypes;
  fill(tracks, daughters, trackTypes, c);
  assert(tracks.size() == daughters.size());
  TransientVertex vertex;
  if(fit(vertex, tracks)) {
    tracks = vertex.refittedTracks();    
    Candidate::Point vtx(vertex.position());
    c.setVertex(vtx);
    std::vector<TransientTrack>::const_iterator trackIt = tracks.begin(), tracksEnd = tracks.end();
    std::vector<Candidate *>::const_iterator daughterIt = daughters.begin();
    std::vector<RecoCandidate::TrackType>::const_iterator trackTypeIt = trackTypes.begin();
    Candidate::LorentzVector mp4(0, 0, 0, 0);
    for(; trackIt != tracksEnd; ++ trackIt, ++ daughterIt, ++trackTypeIt) {
      const Track & track = trackIt->track();
      Candidate & daughter = * * daughterIt;
      double px = track.px(), py = track.py(), pz = track.pz(), p = track.p();
      double energy;
      daughter.setVertex( vtx );
      if(*trackTypeIt == RecoCandidate::recoTrackType) {
	double mass = daughter.mass();
	energy = sqrt(p*p + mass*mass);
      } else {
	energy = daughter.energy();
	double scale = energy / p;
	px *= scale; py *= scale; pz *= scale; 
      }
      Candidate::LorentzVector dp4(px, py, pz, energy);
      daughter.setP4(dp4);
      mp4 += dp4;
    }
    c.setP4(mp4);
    Vertex v = vertex;
    c.setChi2AndNdof(chi2_ = v.chi2(), ndof_ = v.ndof());
    v.fill(cov_);
    c.setCovariance(cov_);
  } else {
    c.setChi2AndNdof(chi2_ = -1, ndof_ = 0);
    c.setCovariance(cov_ = CovarianceMatrix(ROOT::Math::SMatrixIdentity())); 
  }
}

void PFCandCommonVertexFitterBase::fill(std::vector<TransientTrack> & tracks, 
				      std::vector<Candidate *> & daughters,
				      std::vector<RecoCandidate::TrackType> & trackTypes,
				      Candidate & c) const {
  size_t nDau = c.numberOfDaughters();
  for(unsigned int j = 0; j < nDau ; ++j) {
    Candidate * d = c.daughter(j);
    if(d == nullptr) {
      ostringstream message;
      message << "Can't access in write mode candidate daughters. "
	      << "pdgId = " << c.pdgId() << ".\n";
      const Candidate * d1 = c.daughter(j);
      if(d1 == nullptr)
	message << "Null daughter also found in read-only mode\n";
      else
	message << "Daughter found in read-only mode with id: " << d1->pdgId() << "\n";
      throw edm::Exception(edm::errors::InvalidReference) << message.str();
    }
    if(d->numberOfDaughters() > 0)
      fill(tracks, daughters, trackTypes, * d);
    else {
       const Track * trk = nullptr;
       RecoCandidate::TrackType type = RecoCandidate::recoTrackType;
       if (d->hasMasterClone())
       {
          //get the PFCandidate
          const PFCandidate* myPFCand = dynamic_cast<const PFCandidate*>(d->masterClone().get());
          trk = myPFCand->trackRef().get();
       }
      if(trk != nullptr) {
	tracks.push_back(TransientTrack(* trk, bField_));
	daughters.push_back(d);
	trackTypes.push_back(type);
      } else {
	cerr << ">>> warning: candidate of type " << d->pdgId() 
	     << " has no track reference." << endl;
      }
    }
  }
}
