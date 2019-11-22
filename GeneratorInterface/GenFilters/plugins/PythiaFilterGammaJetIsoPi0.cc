#include "GeneratorInterface/GenFilters/plugins/PythiaFilterGammaJetIsoPi0.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <iostream>
#include <list>
#include <vector>
#include <cmath>

using namespace edm;
using namespace std;

namespace {

  double deltaR2(double eta0, double phi0, double eta, double phi) {
    double dphi = phi - phi0;
    if (dphi > M_PI)
      dphi -= 2 * M_PI;
    else if (dphi <= -M_PI)
      dphi += 2 * M_PI;
    return dphi * dphi + (eta - eta0) * (eta - eta0);
  }

  double deltaPhi(double phi0, double phi) {
    double dphi = phi - phi0;
    if (dphi > M_PI)
      dphi -= 2 * M_PI;
    else if (dphi <= -M_PI)
      dphi += 2 * M_PI;
    return dphi;
  }

  class ParticlePtGreater {
  public:
    int operator()(const HepMC::GenParticle* p1, const HepMC::GenParticle* p2) const {
      return p1->momentum().perp() > p2->momentum().perp();
    }
  };
}  // namespace

PythiaFilterGammaJetIsoPi0::PythiaFilterGammaJetIsoPi0(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      etaMin(iConfig.getUntrackedParameter<double>("MinPi0Eta", 1.65)),
      PtMin(iConfig.getUntrackedParameter<double>("MinPi0Pt", 50.)),
      etaMax(iConfig.getUntrackedParameter<double>("MaxPi0Eta", 2.5)),
      PtMax(iConfig.getUntrackedParameter<double>("MaxPi0Pt", 100.)),
      isocone(iConfig.getUntrackedParameter<double>("IsoCone", 0.3)),
      isodr(iConfig.getUntrackedParameter<double>("IsoDR", 0.5)),
      ebEtaMax(1.479) {
  //
  deltaEB = 0.01745 / 2 * 5;     // delta_eta, delta_phi
  deltaEE = 2.93 / 317 / 2 * 5;  // delta_x/z, delta_y/z
  theNumberOfTestedEvt = 0;
  theNumberOfSelected = 0;

  cout << " Cut Definition: " << endl;
  cout << " MinPi0Pt = " << PtMin << endl;
  cout << " MaxPi0Pt = " << PtMax << endl;
  cout << " MinPi0Eta = " << etaMin << endl;
  cout << " MaxPi0Eta = " << etaMax << endl;
  //
  cout << " Pi0 Isolation Cone = " << isocone << endl;
  cout << " Max dR between pi0 and Photon  = " << isodr << endl;
  cout << " 5x5 crystal cone  around pi0 axis in ECAL Barrel = " << deltaEB << endl;
  cout << " 5x5 crystal cone  around pi0 axis in ECAL Endcap = " << deltaEE << endl;

  theNumberOfTestedEvt = 0;
  theNumberOfSelected = 0;
}

PythiaFilterGammaJetIsoPi0::~PythiaFilterGammaJetIsoPi0() {}

// ------------ method called to produce the data  ------------
bool PythiaFilterGammaJetIsoPi0::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  theNumberOfTestedEvt++;
  if (theNumberOfTestedEvt % 1000 == 0)
    cout << "Number of tested events = " << theNumberOfTestedEvt << endl;

  bool accepted = false;
  Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  list<const HepMC::GenParticle*> pi0_seeds;

  int pi0_id = -1;
  int particle_id = 1;
  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    double Particle_pt = (*p)->momentum().perp();
    double Particle_eta = (*p)->momentum().eta();

    if ((*p)->pdg_id() == 111) {
      if (Particle_pt > PtMin && Particle_pt < PtMax && Particle_eta < etaMax && Particle_eta > etaMin) {
        pi0_id = particle_id;
        pi0_seeds.push_back(*p);
        break;
      }
    }
    particle_id++;
  }

  HepMC::GenEvent::particle_const_iterator ppp = myGenEvent->particles_begin();
  for (int i = 0; i < 7; ++i)
    ppp++;
  HepMC::GenParticle* particle8 = (*ppp);

  int photon_id = 7;
  if (particle8->pdg_id() == 22) {
    photon_id = 8;
  }

  if (pi0_seeds.size() == 1) {
    double eta_photon = myGenEvent->barcode_to_particle(photon_id)->momentum().eta();
    double phi_photon = myGenEvent->barcode_to_particle(photon_id)->momentum().eta();
    double eta_pi0 = myGenEvent->barcode_to_particle(pi0_id)->momentum().eta();
    double phi_pi0 = myGenEvent->barcode_to_particle(pi0_id)->momentum().phi();

    double dr_pi0_photon = sqrt(deltaR2(eta_photon, phi_photon, eta_pi0, phi_pi0));
    // check if pi0 comes from the jet and is far from the photon
    // ----------------------------------------------------------
    if (dr_pi0_photon > isodr) {
      bool inEB(false);
      double tgx(0);
      double tgy(0);
      if (std::abs(eta_pi0) < ebEtaMax)
        inEB = true;
      else {
        tgx = myGenEvent->barcode_to_particle(pi0_id)->momentum().px() /
              myGenEvent->barcode_to_particle(pi0_id)->momentum().pz();
        tgy = myGenEvent->barcode_to_particle(pi0_id)->momentum().py() /
              myGenEvent->barcode_to_particle(pi0_id)->momentum().pz();
      }

      double etPi0 = 0;
      double etPi0Charged = 0;
      double etCone = 0;
      double etConeCharged = 0;
      double ptMaxHadron = 0;

      for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
           ++p) {
        if ((*p)->status() != 1)
          continue;  // use only stable particles
        int pid = (*p)->pdg_id();
        int apid = std::abs(pid);
        if (apid > 11 && apid < 21)
          continue;  //get rid of muons and neutrinos
        double eta = (*p)->momentum().eta();
        double phi = (*p)->momentum().phi();
        if (sqrt(deltaR2(eta_pi0, phi_pi0, eta, phi)) > isocone)
          continue;
        double pt = (*p)->momentum().perp();
        etCone += pt;  // add the pt if particle inside the cone

        ESHandle<ParticleDataTable> pdt;
        iSetup.getData(pdt);

        int charge3 = ((pdt->particle((*p)->pdg_id()))->ID().threeCharge());

        if (charge3)
          etConeCharged += pt;

        //select particles matching a crystal array centered on pi0 direction
        if (inEB) {
          if (std::abs(eta - eta_pi0) > deltaEB || std::abs(deltaPhi(phi, phi_pi0)) > deltaEB)
            continue;
        } else if (fabs((*p)->momentum().px() / (*p)->momentum().pz() - tgx) > deltaEE ||
                   fabs((*p)->momentum().py() / (*p)->momentum().pz() - tgy) > deltaEE)
          continue;

        etPi0 += pt;

        if (charge3)
          etPi0Charged += pt;

        if (apid > 100 && apid != 310 && pt > ptMaxHadron)
          ptMaxHadron = pt;  // 310 -> K0s

      }  // for ( HepMC::GenEvent::particle_const_iterator

      //isolation cuts

      double iso_cut1 = 5 + etPi0 / 20 - etPi0 * etPi0 / 1e4;
      double iso_cut2 = 3 + etPi0 / 20 - etPi0 * etPi0 * etPi0 / 1e6;
      double iso_cut3 = 4.5 + etPi0 / 40;
      if (etPi0 > 165.) {
        iso_cut1 = 5. + 165. / 20. - 165. * 165. / 1e4;
        iso_cut2 = 3. + 165. / 20. - 165. * 165. * 165. / 1e6;
        iso_cut3 = 4.5 + 165. / 40.;
      }
      double iso_cut4 = 0.02;  // Fraction of charged energy in the cone 2%

      double iso_val1 = etCone - etPi0;
      double iso_val2 = iso_val1 - (etConeCharged - etPi0Charged);
      double iso_val3 = etConeCharged / etPi0;

      if (iso_val1 < iso_cut1) {
        if (iso_val2 < iso_cut2) {
          if (ptMaxHadron < iso_cut3) {
            if (iso_val3 < iso_cut4) {
              accepted = true;
            }
          }  // if(ptMaxHadron < iso_cut3)
        }    // if( iso_val2 < iso_cut2)
      }      // if( iso_val1 < iso_cut1)
    }        // if(dr_pi0_jet < 0.1 && deta_pi0_photon > 1)
  }          //if(pi0_seeds.size() == 1) {

  if (accepted) {
    theNumberOfSelected++;
    cout << "========>  Event: " << iEvent.id() << " of Proccess ID " << myGenEvent->signal_process_id()
         << " preselected " << endl;
    cout << " Number of preselected events: " << theNumberOfSelected << endl;
    return true;
  } else
    return false;
}
