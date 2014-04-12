#include "PhysicsTools/RecoUtils/interface/CandCommonVertexFitter.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <sstream>
using namespace reco;
using namespace std;

void CandCommonVertexFitterBase::set(VertexCompositeCandidate & c) const {
  if(bField_ == 0)
    throw edm::Exception(edm::errors::InvalidReference)
      << "B-Field was not set up CandCommonVertexFitter.\n"
      << "the following method must be called before fitting a candidate:\n"
      << " CandCommonVertexFitter:.set( const MagneticField * )" << endl;
  vector<TransientTrack> tracks;
  vector<Candidate *> daughters;
  vector<RecoCandidate::TrackType> trackTypes;
  fill(tracks, daughters, trackTypes, c);
  assert(tracks.size() == daughters.size());
  TransientVertex vertex;
  if(fit(vertex, tracks)) {
    tracks = vertex.refittedTracks();    
    Candidate::Point vtx(vertex.position());
    c.setVertex(vtx);
    vector<TransientTrack>::const_iterator trackIt = tracks.begin(), tracksEnd = tracks.end();
    vector<Candidate *>::const_iterator daughterIt = daughters.begin();
    vector<RecoCandidate::TrackType>::const_iterator trackTypeIt = trackTypes.begin();
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

void CandCommonVertexFitterBase::fill(vector<TransientTrack> & tracks, 
				      vector<Candidate *> & daughters,
				      vector<RecoCandidate::TrackType> & trackTypes,
				      Candidate & c) const {
  size_t nDau = c.numberOfDaughters();
  for(unsigned int j = 0; j < nDau ; ++j) {
    Candidate * d = c.daughter(j);
    if(d == 0) {
      ostringstream message;
      message << "Can't access in write mode candidate daughters. "
	      << "pdgId = " << c.pdgId() << ".\n";
      const Candidate * d1 = c.daughter(j);
      if(d1 == 0)
	message << "Null daughter also found in read-only mode\n";
      else
	message << "Daughter found in read-only mode with id: " << d1->pdgId() << "\n";
      throw edm::Exception(edm::errors::InvalidReference) << message.str();
    }
    if(d->numberOfDaughters() > 0)
      fill(tracks, daughters, trackTypes, * d);
    else {
      const Track * trk = d->get<const Track *>();
      RecoCandidate::TrackType type = d->get<RecoCandidate::TrackType>();
      if(trk != 0) {
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
