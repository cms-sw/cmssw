/** \class PythiaFilterGammaJet
 *
 *  PythiaFilterGammaJet filter implements generator-level preselections
 *  for photon+jet like events to be used in jet energy calibration.
 *  Ported from fortran code written by V.Konoplianikov.
 *
 * \author A.Ulyanov, ITEP
 *
 ************************************************************/

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"

#include <cmath>
#include <cstdlib>
#include <list>
#include <string>

class PythiaFilterGammaJetWithOutBg : public edm::global::EDFilter<> {
public:
  explicit PythiaFilterGammaJetWithOutBg(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const edm::ESGetToken<ParticleDataTable, PDTRecord> particleDataTableToken_;

  const double etaMax;
  const double ptSeed;
  const double ptMin;
  const double ptMax;
  const double dphiMin;
  const double detaMax;
  const double etaPhotonCut2;

  const double cone;
  const double ebEtaMax;
  const double deltaEB;
  const double deltaEE;
};

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

PythiaFilterGammaJetWithOutBg::PythiaFilterGammaJetWithOutBg(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      particleDataTableToken_(esConsumes<ParticleDataTable, PDTRecord>()),
      etaMax(iConfig.getUntrackedParameter<double>("MaxPhotonEta", 2.8)),
      ptSeed(iConfig.getUntrackedParameter<double>("PhotonSeedPt", 5.)),
      ptMin(iConfig.getUntrackedParameter<double>("MinPhotonPt")),
      ptMax(iConfig.getUntrackedParameter<double>("MaxPhotonPt")),
      dphiMin(iConfig.getUntrackedParameter<double>("MinDeltaPhi", -1) / 180 * M_PI),
      detaMax(iConfig.getUntrackedParameter<double>("MaxDeltaEta", 10.)),
      etaPhotonCut2(iConfig.getUntrackedParameter<double>("MinPhotonEtaForwardJet", 1.3)),
      cone(0.5),
      ebEtaMax(1.479),
      deltaEB(0.01745 / 2 * 5),       // delta_eta, delta_phi
      deltaEE(2.93 / 317 / 2 * 5) {}  // delta_x/z, delta_y/z

bool PythiaFilterGammaJetWithOutBg::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  std::list<const HepMC::GenParticle*> seeds;
  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if ((*p)->pdg_id() == 22 && (*p)->status() == 1 && (*p)->momentum().perp() > ptSeed &&
        std::abs((*p)->momentum().eta()) < etaMax)
      seeds.push_back(*p);
  }

  seeds.sort(ParticlePtGreater());

  for (std::list<const HepMC::GenParticle*>::const_iterator is = seeds.begin(); is != seeds.end(); is++) {
    double etaPhoton = (*is)->momentum().eta();
    double phiPhoton = (*is)->momentum().phi();

    HepMC::GenEvent::particle_const_iterator ppp = myGenEvent->particles_begin();
    for (int i = 0; i < 6; ++i)
      ppp++;
    HepMC::GenParticle* particle7 = (*ppp);
    ppp++;
    HepMC::GenParticle* particle8 = (*ppp);

    double dphi7 = std::abs(deltaPhi(phiPhoton, particle7->momentum().phi()));
    double dphi8 = std::abs(deltaPhi(phiPhoton, particle8->momentum().phi()));

    double etaJet = dphi7 > dphi8 ? particle7->momentum().eta() : particle8->momentum().eta();

    double eta1 = etaJet - detaMax;
    double eta2 = etaJet + detaMax;
    if (eta1 > etaPhotonCut2)
      eta1 = etaPhotonCut2;
    if (eta2 < -etaPhotonCut2)
      eta2 = -etaPhotonCut2;

    if (etaPhoton < eta1 || etaPhoton > eta2) {
      continue;
    }
    bool inEB(false);
    double tgx(0);
    double tgy(0);
    if (std::abs(etaPhoton) < ebEtaMax)
      inEB = true;
    else {
      tgx = (*is)->momentum().px() / (*is)->momentum().pz();
      tgy = (*is)->momentum().py() / (*is)->momentum().pz();
    }

    double etPhoton = 0;
    double ptMaxHadron = 0;

    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
         ++p) {
      if ((*p)->status() != 1)
        continue;
      int apid = std::abs((*p)->pdg_id());
      if (apid > 11 && apid < 21)
        continue;  //get rid of muons and neutrinos
      double eta = (*p)->momentum().eta();
      double phi = (*p)->momentum().phi();
      if (deltaR2(etaPhoton, phiPhoton, eta, phi) > cone * cone)
        continue;
      double pt = (*p)->momentum().perp();

      //select particles matching a crystal array centered on photon
      if (inEB) {
        if (std::abs(eta - etaPhoton) > deltaEB || std::abs(deltaPhi(phi, phiPhoton)) > deltaEB)
          continue;
      } else if (std::abs((*p)->momentum().px() / (*p)->momentum().pz() - tgx) > deltaEE ||
                 std::abs((*p)->momentum().py() / (*p)->momentum().pz() - tgy) > deltaEE)
        continue;

      etPhoton += pt;
      if (apid > 100 && apid != 310 && pt > ptMaxHadron)
        ptMaxHadron = pt;
    }

    if (etPhoton < ptMin || etPhoton > ptMax) {
      continue;
    }

    accepted = true;
    break;

  }  //loop over seeds
  return accepted;
}

DEFINE_FWK_MODULE(PythiaFilterGammaJetWithOutBg);
