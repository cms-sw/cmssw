#include "PhysicsTools/RecoUtils/interface/CandKinematicVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <sstream>
#include <iostream>
using namespace reco;
using namespace std;

// perform the kinematic fit
bool CandKinematicVertexFitter::fit(const vector<RefCountedKinematicParticle> &particles) const {
  try {
    tree_ = fitter_.fit(particles);
  } catch (std::exception &err) {
    std::cerr << ">>> exception thrown by KinematicParticleVertexFitter:\n"
              << err.what() << "\n"
              << ">>> candidate not fitted to common vertex" << std::endl;
    return false;
  }
  //check tree_ is valid here!
  if (tree_->isValid())
    return true;
  else
    return false;
}

// main method called by CandProducer sets the VertexCompositeCandidate
void CandKinematicVertexFitter::set(VertexCompositeCandidate &c) const {
  if (bField_ == nullptr)
    throw edm::Exception(edm::errors::InvalidReference)
        << "B-Field was not set up CandKinematicVertexFitter.\n"
        << "the following method must be called before fitting a candidate:\n"
        << " CandKinematicVertexFitter:.set( const MagneticField * )" << endl;
  vector<RefCountedKinematicParticle> particles;
  vector<Candidate *> daughters;
  vector<RecoCandidate::TrackType> trackTypes;
  // fill particles with KinematicParticles and daughters with Candidates of the daughters of c
  fill(particles, daughters, trackTypes, c);
  assert(particles.size() == daughters.size());

  // attempt to fit the KinematicParticles, particles
  if (fit(particles)) {
    // after the fit, tree_ contains the KinematicTree from the fit
    tree_->movePointerToTheTop();
    // set the kinematic properties of the daughters from the fit
    RefCountedKinematicVertex vertex = tree_->currentDecayVertex();
    if (vertex->vertexIsValid()) {
      Candidate::Point vtx(vertex->position());
      c.setVertex(vtx);
      vector<RefCountedKinematicParticle> treeParticles = tree_->daughterParticles();
      vector<RefCountedKinematicParticle>::const_iterator particleIt = treeParticles.begin();
      vector<Candidate *>::const_iterator daughterIt = daughters.begin(), daughtersEnd = daughters.end();
      vector<RecoCandidate::TrackType>::const_iterator trackTypeIt = trackTypes.begin();
      Candidate::LorentzVector mp4(0, 0, 0, 0);
      for (; daughterIt != daughtersEnd; ++particleIt, ++daughterIt, ++trackTypeIt) {
        Candidate &daughter = **daughterIt;
        GlobalVector p3 = (*particleIt)->currentState().globalMomentum();
        double px = p3.x(), py = p3.y(), pz = p3.z(), p = p3.mag();
        double energy;

        if (!daughter.longLived())
          daughter.setVertex(vtx);
        double scale;
        switch (*trackTypeIt) {
          case RecoCandidate::gsfTrackType:
            //gsf used for electron tracks
            energy = daughter.energy();
            scale = energy / p;
            px *= scale;
            py *= scale;
            pz *= scale;
            [[fallthrough]];
          default:
            double mass = (*particleIt)->currentState().mass();
            energy = sqrt(p * p + mass * mass);
        };
        Candidate::LorentzVector dp4(px, py, pz, energy);
        daughter.setP4(dp4);
        mp4 += dp4;
      }
      c.setP4(mp4);
      c.setChi2AndNdof(chi2_ = vertex->chiSquared(), ndof_ = vertex->degreesOfFreedom());
      GlobalError err = vertex->error();
      cov_(0, 0) = err.cxx();
      cov_(0, 1) = err.cyx();
      cov_(0, 2) = err.czx();
      cov_(1, 2) = err.czy();
      cov_(1, 1) = err.cyy();
      cov_(2, 2) = err.czz();
      c.setCovariance(cov_);
    }
  } else {
    c.setChi2AndNdof(chi2_ = -1, ndof_ = 0);
    c.setCovariance(cov_ = CovarianceMatrix(ROOT::Math::SMatrixIdentity()));
  }
}

// methond to fill the properties of a CompositeCandidate's daughters
void CandKinematicVertexFitter::fill(vector<RefCountedKinematicParticle> &particles,
                                     vector<Candidate *> &daughters,
                                     vector<RecoCandidate::TrackType> &trackTypes,
                                     Candidate &c) const {
  size_t nDau = c.numberOfDaughters();
  // loop through CompositeCandidate daughters
  for (unsigned int j = 0; j < nDau; ++j) {
    Candidate *d = c.daughter(j);
    if (d == nullptr) {
      ostringstream message;
      message << "Can't access in write mode candidate daughters. "
              << "pdgId = " << c.pdgId() << ".\n";
      const Candidate *d1 = c.daughter(j);
      if (d1 == nullptr)
        message << "Null daughter also found in read-only mode\n";
      else
        message << "Daughter found in read-only mode with id: " << d1->pdgId() << "\n";
      throw edm::Exception(edm::errors::InvalidReference) << message.str();
    }
    //check for a daughter which itself is a composite
    if (d->numberOfDaughters() > 0) {
      //try to cast to VertexCompositeCandiate
      VertexCompositeCandidate *vtxDau = dynamic_cast<VertexCompositeCandidate *>(d);
      if (vtxDau != nullptr && vtxDau->vertexChi2() > 0) {
        // if VertexCompositeCandidate refit vtxDau via the set method
        (*this).set(*vtxDau);
        // if mass constraint is desired, do it here
        if (vtxDau->massConstraint()) {
          KinematicParticleFitter csFitter;
          //get particle mass from pdg table via pdgid number
          const ParticleData *data = pdt_->particle(vtxDau->pdgId());
          ParticleMass mass = data->mass();
          float mass_sigma = mass * 0.000001;  //needs a sigma for the fit
          // create a KinematicConstraint and refit the tree with it
          //KinematicConstraint * mass_c = new MassKinematicConstraint(mass,mass_sigma);
          MassKinematicConstraint mkc(mass, mass_sigma);
          KinematicConstraint *mass_c(&mkc);
          tree_ = csFitter.fit(mass_c, tree_);
          //CHECK THIS! the following works, but might not be safe
          //tree_ = csFitter.fit(&(MassKinematicConstraint(mass,mass_sigma)),tree_);
        }
        // add the kinematic particle from the fit to particles
        RefCountedKinematicParticle current = (*this).currentParticle();
        particles.push_back(current);
        daughters.push_back(d);
        trackTypes.push_back(RecoCandidate::noTrackType);
      } else {
        fill(particles, daughters, trackTypes, *d);
      }
    } else {
      //get track, make KinematicParticle and add to particles so it can be fit
      TrackRef trk = d->get<TrackRef>();
      RecoCandidate::TrackType type = d->get<RecoCandidate::TrackType>();
      if (!trk.isNull()) {
        TransientTrack trTrk(trk, bField_);
        float chi2 = 0, ndof = 0;
        ParticleMass mass = d->mass();
        float sigma = mass * 1.e-6;
        particles.push_back(factory_.particle(trTrk, mass, chi2, ndof, sigma));
        daughters.push_back(d);
        trackTypes.push_back(type);
      } else {
        cerr << ">>> warning: candidate of type " << d->pdgId() << " has no track reference." << endl;
      }
    }
  }
}
