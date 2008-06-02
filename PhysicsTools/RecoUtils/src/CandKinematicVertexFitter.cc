#include "PhysicsTools/RecoUtils/interface/CandKinematicVertexFitter.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <sstream>
#include <iostream>
using namespace reco;
using namespace std;

bool CandKinematicVertexFitter::fit(const vector<RefCountedKinematicParticle> & particles) const {
  try {
    tree_ = fitter_.fit(particles);
  } catch (std::exception & err) {
    std::cerr << ">>> exception thrown by KinematicParticleVertexFitter:\n"
	      << err.what() << "\n"
	      << ">>> candidate not fitted to common vertex" << std::endl;
    return false;
  }
  return true;
}

void CandKinematicVertexFitter::set(VertexCompositeCandidate & c) const {
  if(bField_ == 0)
    throw edm::Exception(edm::errors::InvalidReference)
      << "B-Field was not set up CandKinematicVertexFitter.\n"
      << "the following method must be called before fitting a candidate:\n"
      << " CandKinematicVertexFitter:.set( const MagneticField * )" << endl;
  vector<RefCountedKinematicParticle> particles;
  vector<Candidate *> daughters;
  vector<RecoCandidate::TrackType> trackTypes;
  fill(particles, daughters, trackTypes, c);
  assert(particles.size() == daughters.size());
  if(fit(particles)) {
    tree_->movePointerToTheTop();
    RefCountedKinematicVertex vertex = tree_->currentDecayVertex();
    if(vertex->vertexIsValid()) {
      Candidate::Point vtx(vertex->position());
      c.setVertex(vtx);
      vector<RefCountedKinematicParticle> treeParticles = tree_->daughterParticles();
      vector<RefCountedKinematicParticle>::const_iterator particleIt = treeParticles.begin();
      vector<Candidate *>::const_iterator daughterIt = daughters.begin(), daughtersEnd = daughters.end();
      vector<RecoCandidate::TrackType>::const_iterator trackTypeIt = trackTypes.begin();
      Candidate::LorentzVector mp4(0, 0, 0, 0);
      for(; daughterIt != daughtersEnd; ++ particleIt, ++ daughterIt, ++trackTypeIt) {
	GlobalVector p3 = (*particleIt)->currentState().globalMomentum();
	double px = p3.x(), py = p3.y(), pz = p3.z(), p = p3.mag();
	double energy;
	Candidate & daughter = * * daughterIt;
	if(!daughter.longLived()) daughter.setVertex(vtx);
	double scale;
	switch(*trackTypeIt) {
	case RecoCandidate::gsfTrackType :
	  energy = daughter.energy();
	  scale = energy / p;
	  px *= scale; py *= scale; pz *= scale; 
	default:
	  double mass = daughter.mass();
	  energy = sqrt(p*p + mass*mass);
	};
	Candidate::LorentzVector dp4(px, py, pz, energy);
	daughter.setP4(dp4);
	mp4 += dp4;
      }
      c.setP4(mp4);
      c.setChi2AndNdof(chi2_ = vertex->chiSquared(), ndof_ = vertex->degreesOfFreedom());
      GlobalError err = vertex->error();
      cov_(0,0) = err.cxx();
      cov_(0,1) = err.cyx();
      cov_(0,2) = err.czx();
      cov_(1,2) = err.czy();
      cov_(1,1) = err.cyy();
      cov_(2,2) = err.czz();
      c.setCovariance(cov_);
    }
  } else {
    c.setChi2AndNdof(chi2_ = -1, ndof_ = 0);
    c.setCovariance(cov_ = CovarianceMatrix(ROOT::Math::SMatrixIdentity())); 
  }
}

void CandKinematicVertexFitter::fill(vector<RefCountedKinematicParticle> & particles, 
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
    if(d->numberOfDaughters() > 0) {
      VertexCompositeCandidate * vtxDau = dynamic_cast<VertexCompositeCandidate*>(d);
      if(vtxDau!=0 && vtxDau->longLived()) {
	fitters_->push_back(CandKinematicVertexFitter(*this));
	CandKinematicVertexFitter & fitter = fitters_->back();
	fitter.set(*vtxDau);
	RefCountedKinematicParticle current = fitter.currentParticle();
	particles.push_back(current);
	daughters.push_back(d);
	trackTypes.push_back(RecoCandidate::noTrackType);
      } else
	fill(particles, daughters, trackTypes, *d);
    }
    else {
      const Track * trk = d->get<const Track *>();
      RecoCandidate::TrackType type = d->get<RecoCandidate::TrackType>();
      if(trk != 0) {
	TransientTrack trTrk(*trk, bField_);
	float chi2 = 0, ndof = 0;
	ParticleMass mass = d->mass();
	float sigma = mass *1.e-6;
	particles.push_back(factory_.particle(trTrk, mass, chi2, ndof, sigma));
	daughters.push_back(d);
	trackTypes.push_back(type);
      } else {
	cerr << ">>> warning: candidate of type " << d->pdgId() 
	     << " has no track reference." << endl;
      }
    }
  }
}
