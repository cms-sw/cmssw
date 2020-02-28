#include "GeneratorInterface/GenFilters/plugins/PythiaFilterIsolatedTrack.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <iostream>
#include <list>
#include <vector>
#include <cmath>

std::pair<double, double> PythiaFilterIsolatedTrack::GetEtaPhiAtEcal(
    double etaIP, double phiIP, double pT, int charge, double vtxZ) {
  double deltaPhi;
  double etaEC = 100;
  double phiEC = 100;
  double Rcurv = pT * 33.3 * 100 / 38;  //r(m)=pT(GeV)*33.3/B(kG)
  double theta = 2 * atan(exp(-etaIP));
  double zNew;
  if (theta > CLHEP::halfpi)
    theta = CLHEP::pi - theta;
  if (fabs(etaIP) < 1.479) {
    deltaPhi = -charge * asin(0.5 * ecRad_ / Rcurv);
    double alpha1 = 2 * asin(0.5 * ecRad_ / Rcurv);
    double z = ecRad_ / tan(theta);
    if (etaIP > 0)
      zNew = z * (Rcurv * alpha1) / ecRad_ + vtxZ;  //new z-coordinate of track
    else
      zNew = -z * (Rcurv * alpha1) / ecRad_ + vtxZ;  //new z-coordinate of track
    double zAbs = fabs(zNew);
    if (zAbs < ecDist_) {
      etaEC = -log(tan(0.5 * atan(ecRad_ / zAbs)));
      deltaPhi = -charge * asin(0.5 * ecRad_ / Rcurv);
    }
    if (zAbs > ecDist_) {
      zAbs = (fabs(etaIP) / etaIP) * ecDist_;
      double Zflight = fabs(zAbs - vtxZ);
      double alpha = (Zflight * ecRad_) / (z * Rcurv);
      double Rec = 2 * Rcurv * sin(alpha / 2);
      deltaPhi = -charge * alpha / 2;
      etaEC = -log(tan(0.5 * atan(Rec / ecDist_)));
    }
  } else {
    zNew = (fabs(etaIP) / etaIP) * ecDist_;
    double Zflight = fabs(zNew - vtxZ);
    double Rvirt = fabs(Zflight * tan(theta));
    double Rec = 2 * Rcurv * sin(Rvirt / (2 * Rcurv));
    deltaPhi = -(charge) * (Rvirt / (2 * Rcurv));
    etaEC = -log(tan(0.5 * atan(Rec / ecDist_)));
  }

  if (zNew < 0)
    etaEC = -etaEC;
  phiEC = phiIP + deltaPhi;

  if (phiEC < -CLHEP::pi)
    phiEC = 2 * CLHEP::pi + phiEC;
  if (phiEC > CLHEP::pi)
    phiEC = -2 * CLHEP::pi + phiEC;

  std::pair<double, double> retVal(etaEC, phiEC);
  return retVal;
}

double PythiaFilterIsolatedTrack::getDistInCM(double eta1, double phi1, double eta2, double phi2) {
  double dR, Rec;
  if (fabs(eta1) < 1.479)
    Rec = ecRad_;
  else
    Rec = ecDist_;
  double ce1 = cosh(eta1);
  double ce2 = cosh(eta2);
  double te1 = tanh(eta1);
  double te2 = tanh(eta2);

  double z = cos(phi1 - phi2) / ce1 / ce2 + te1 * te2;
  if (z != 0)
    dR = fabs(Rec * ce1 * sqrt(1. / z / z - 1.));
  else
    dR = 999999.;
  return dR;
}

PythiaFilterIsolatedTrack::PythiaFilterIsolatedTrack(const edm::ParameterSet &iConfig,
                                                     const PythiaFilterIsoTracks::Counters *counters)
    : token_(consumes<edm::HepMCProduct>(
          iConfig.getUntrackedParameter("moduleLabel", edm::InputTag("generator", "unsmeared")))),
      maxSeedEta_(iConfig.getUntrackedParameter<double>("maxSeedEta", 2.3)),
      minSeedEta_(iConfig.getUntrackedParameter<double>("minSeedEta", 0.0)),
      minSeedMom_(iConfig.getUntrackedParameter<double>("minSeedMom", 20.)),
      minIsolTrackMom_(iConfig.getUntrackedParameter<double>("minIsolTrackMom", 2.0)),
      isolCone_(iConfig.getUntrackedParameter<double>("isolCone", 40.0)),
      onlyHadrons_(iConfig.getUntrackedParameter<bool>("onlyHadrons", true)),
      nAll_(0),
      nGood_(0),
      ecDist_(317.0),
      ecRad_(129.0) {
  edm::LogVerbatim("PythiaFilter") << "PythiaFilterIsolatedTrack: Eta " << minSeedEta_ << ":" << maxSeedEta_ << " p > "
                                   << minSeedMom_ << " Isolation Cone " << isolCone_ << " with p > " << minIsolTrackMom_
                                   << " OnlyHadron " << onlyHadrons_;
}

PythiaFilterIsolatedTrack::~PythiaFilterIsolatedTrack() {}

// ------------ method called to produce the data  ------------
bool PythiaFilterIsolatedTrack::filter(edm::Event &iEvent, edm::EventSetup const &iSetup) {
  ++nAll_;
  edm::ESHandle<ParticleDataTable> pdt;
  iSetup.getData(pdt);

  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent *myGenEvent = evt->GetEvent();

  // all of the stable, charged particles with momentum>minIsolTrackMom_ and |eta|<maxSeedEta_+0.5
  std::vector<const HepMC::GenParticle *> chargedParticles;

  // all of the stable, charged particles with momentum>minSeedMom_ and minSeedEta_<=|eta|<maxSeedEta_
  std::vector<const HepMC::GenParticle *> seeds;

  // fill the vector of charged particles and seeds in the event
  for (HepMC::GenEvent::particle_const_iterator iter = myGenEvent->particles_begin();
       iter != myGenEvent->particles_end();
       ++iter) {
    const HepMC::GenParticle *p = *iter;
    if (!(pdt->particle(p->pdg_id())))
      continue;
    int charge3 = pdt->particle(p->pdg_id())->ID().threeCharge();
    int status = p->status();
    double momentum = p->momentum().rho();
    double abseta = fabs(p->momentum().eta());

    // only consider stable, charged particles
    if (abs(charge3) == 3 && status == 1 && momentum > minIsolTrackMom_ && abseta < maxSeedEta_ + 0.5) {
      chargedParticles.push_back(p);
      if (momentum > minSeedMom_ && abseta < maxSeedEta_ && abseta >= minSeedEta_) {
        seeds.push_back(p);
      }
    }
  }

  // loop over all the seeds and see if any of them are isolated
  unsigned int ntrk(0);
  for (std::vector<const HepMC::GenParticle *>::const_iterator it1 = seeds.begin(); it1 != seeds.end(); ++it1) {
    const HepMC::GenParticle *p1 = *it1;
    if (!(pdt->particle(p1->pdg_id())))
      continue;
    if (p1->pdg_id() < -100 || p1->pdg_id() > 100 || (!onlyHadrons_)) {  // Select hadrons only
      std::pair<double, double> EtaPhi1 = GetEtaPhiAtEcal(p1->momentum().eta(),
                                                          p1->momentum().phi(),
                                                          p1->momentum().perp(),
                                                          (pdt->particle(p1->pdg_id()))->ID().threeCharge() / 3,
                                                          0.0);

      // loop over all of the other charged particles in the event, and see if any are close by
      bool failsIso = false;
      for (std::vector<const HepMC::GenParticle *>::const_iterator it2 = chargedParticles.begin();
           it2 != chargedParticles.end();
           ++it2) {
        const HepMC::GenParticle *p2 = *it2;

        // don't consider the seed particle among the other charge particles
        if (p1 != p2) {
          std::pair<double, double> EtaPhi2 = GetEtaPhiAtEcal(p2->momentum().eta(),
                                                              p2->momentum().phi(),
                                                              p2->momentum().perp(),
                                                              (pdt->particle(p2->pdg_id()))->ID().threeCharge() / 3,
                                                              0.0);

          // find out how far apart the particles are
          // if the seed fails the isolation requirement, try a different seed
          // occasionally allow a seed to pass to isolation requirement
          if (getDistInCM(EtaPhi1.first, EtaPhi1.second, EtaPhi2.first, EtaPhi2.second) < isolCone_) {
            failsIso = true;
            break;
          }
        }
      }

      if (!failsIso)
        ++ntrk;
    }
  }  //loop over seeds
  if (ntrk > 0) {
    ++nGood_;
    return true;
  } else {
    return false;
  }
}

void PythiaFilterIsolatedTrack::endStream() {
  globalCache()->nAll_ += nAll_;
  globalCache()->nGood_ += nGood_;
}

void PythiaFilterIsolatedTrack::globalEndJob(const PythiaFilterIsoTracks::Counters *count) {
  edm::LogVerbatim("PythiaFilter") << "PythiaFilterIsolatedTrack::Accepts " << count->nGood_ << " events out of "
                                   << count->nAll_ << std::endl;
}
