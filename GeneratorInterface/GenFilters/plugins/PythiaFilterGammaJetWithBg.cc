#include "GeneratorInterface/GenFilters/plugins/PythiaFilterGammaJetWithBg.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <iostream>
#include <list>
#include <vector>
#include <cmath>

//using namespace edm;
//using namespace std;

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

PythiaFilterGammaJetWithBg::PythiaFilterGammaJetWithBg(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      etaMax(iConfig.getUntrackedParameter<double>("MaxPhotonEta", 2.8)),
      ptSeed(iConfig.getUntrackedParameter<double>("PhotonSeedPt", 5.)),
      ptMin(iConfig.getUntrackedParameter<double>("MinPhotonPt")),
      ptMax(iConfig.getUntrackedParameter<double>("MaxPhotonPt")),
      dphiMin(iConfig.getUntrackedParameter<double>("MinDeltaPhi", -1) / 180 * M_PI),
      detaMax(iConfig.getUntrackedParameter<double>("MaxDeltaEta", 10.)),
      etaPhotonCut2(iConfig.getUntrackedParameter<double>("MinPhotonEtaForwardJet", 1.3)),
      cone(0.5),
      ebEtaMax(1.479),
      maxnumberofeventsinrun(iConfig.getUntrackedParameter<int>("MaxEvents", 10)) {
  deltaEB = 0.01745 / 2 * 5;     // delta_eta, delta_phi
  deltaEE = 2.93 / 317 / 2 * 5;  // delta_x/z, delta_y/z
  theNumberOfSelected = 0;
}

PythiaFilterGammaJetWithBg::~PythiaFilterGammaJetWithBg() {}

// ------------ method called to produce the data  ------------
bool PythiaFilterGammaJetWithBg::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // <<<<<<< PythiaFilterGammaJetWithBg.cc
  //  if(theNumberOfSelected>=maxnumberofeventsinrun)   {
  //    throw cms::Exception("endJob")<<"we have reached the maximum number of events ";
  //  }
  // =======
  // >>>>>>> 1.4

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

  //  std::cout<<" Number of seeds "<<seeds.size()<<" ProccessID "<<myGenEvent->signal_process_id()<<std::endl;
  //  std::cout<<" ParticleId 7= "<<myGenEvent->particle(7)->pdg_id()
  //  <<" pT "<<myGenEvent->particle(7)->momentum().perp()
  //  <<" Eta "<<myGenEvent->particle(7)->momentum().eta()
  //  <<" Phi "<<myGenEvent->particle(7)->momentum().phi()<<std::endl;
  //  std::cout<<" ParticleId 8= "<<myGenEvent->particle(8)->pdg_id()<<" pT "<<myGenEvent->particle(8)->momentum().perp()<<" Eta "<<myGenEvent->particle(8)->momentum().eta()<<" Phi "<<myGenEvent->particle(8)->momentum().phi()<<std::endl;

  for (std::list<const HepMC::GenParticle*>::const_iterator is = seeds.begin(); is != seeds.end(); is++) {
    double etaPhoton = (*is)->momentum().eta();
    double phiPhoton = (*is)->momentum().phi();

    /*
    double dphi7=std::abs(deltaPhi(phiPhoton, 
			       myGenEvent->particle(7)->momentum().phi()));
    double dphi=std::abs(deltaPhi(phiPhoton, 
			      myGenEvent->particle(8)->momentum().phi()));
    */

    //***
    HepMC::GenEvent::particle_const_iterator ppp = myGenEvent->particles_begin();
    for (int i = 0; i < 6; ++i)
      ppp++;
    HepMC::GenParticle* particle7 = (*ppp);
    ppp++;
    HepMC::GenParticle* particle8 = (*ppp);

    double dphi7 = std::abs(deltaPhi(phiPhoton, particle7->momentum().phi()));
    double dphi = std::abs(deltaPhi(phiPhoton, particle8->momentum().phi()));
    //***

    int jetline = 8;
    if (dphi7 > dphi) {
      dphi = dphi7;
      jetline = 7;
    }

    //    std::cout<<" Dphi "<<dphi<<" "<<dphiMin<<std::endl;
    //    if(dphi<dphiMin) {
    //      std::cout<<"Reject dphi"<<std::endl;
    //      continue;
    //    }

    //double etaJet= myGenEvent->particle(jetline)->momentum().eta();
    //***
    double etaJet = 0.0;
    if (jetline == 8)
      etaJet = particle8->momentum().eta();
    else
      etaJet = particle7->momentum().eta();
    //***

    double eta1 = etaJet - detaMax;
    double eta2 = etaJet + detaMax;
    if (eta1 > etaPhotonCut2)
      eta1 = etaPhotonCut2;
    if (eta2 < -etaPhotonCut2)
      eta2 = -etaPhotonCut2;
    //    std::cout<<" Etaphoton "<<etaPhoton<<" "<<eta1<<" "<<eta2<<std::endl;
    if (etaPhoton < eta1 || etaPhoton > eta2) {
      //       std::cout<<"Reject eta"<<std::endl;
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
    double etPhotonCharged = 0;
    double etCone = 0;
    double etConeCharged = 0;
    double ptMaxHadron = 0;

    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
         ++p) {
      if ((*p)->status() != 1)
        continue;
      int pid = (*p)->pdg_id();
      int apid = std::abs(pid);
      if (apid > 11 && apid < 21)
        continue;  //get rid of muons and neutrinos
      double eta = (*p)->momentum().eta();
      double phi = (*p)->momentum().phi();
      if (deltaR2(etaPhoton, phiPhoton, eta, phi) > cone * cone)
        continue;
      double pt = (*p)->momentum().perp();

      //***
      edm::ESHandle<ParticleDataTable> pdt;
      iSetup.getData(pdt);

      // double charge=(*p)->particledata().charge();
      // int charge3=(*p)->particleID().threeCharge();

      int charge3 = ((pdt->particle((*p)->pdg_id()))->ID().threeCharge());
      //***

      etCone += pt;
      if (charge3 && pt < 2)
        etConeCharged += pt;

      //select particles matching a crystal array centered on photon
      if (inEB) {
        if (std::abs(eta - etaPhoton) > deltaEB || std::abs(deltaPhi(phi, phiPhoton)) > deltaEB)
          continue;
      } else if (std::abs((*p)->momentum().px() / (*p)->momentum().pz() - tgx) > deltaEE ||
                 std::abs((*p)->momentum().py() / (*p)->momentum().pz() - tgy) > deltaEE)
        continue;

      etPhoton += pt;
      if (charge3 && pt < 2)
        etPhotonCharged += pt;
      if (apid > 100 && apid != 310 && pt > ptMaxHadron)
        ptMaxHadron = pt;
    }
    //    std::cout<<" etPhoton "<<etPhoton<<" "<<ptMin<<" "<<ptMax<<std::endl;

    if (etPhoton < ptMin || etPhoton > ptMax) {
      //     std::cout<<" Reject etPhoton "<<std::endl;
      continue;
    }
    //isolation cuts

    //    double isocut1 = 5+etPhoton/20-etPhoton*etPhoton/1e4;
    double isocut2 = 3 + etPhoton / 20 - etPhoton * etPhoton * etPhoton / 1e6;
    double isocut3 = 4.5 + etPhoton / 40;
    if (etPhoton > 165.) {
      //     isocut1 = 5.+165./20.-165.*165./1e4;
      isocut2 = 3. + 165. / 20. - 165. * 165. * 165. / 1e6;
      isocut3 = 4.5 + 165. / 40.;
    }

    //    std::cout<<" etCone "<<etCone<<" "<<etPhoton<<" "<<etCone-etPhoton<<" "<<isocut1<<std::endl;
    //    std::cout<<"Second cut on iso "<<etCone-etPhoton-(etConeCharged-etPhotonCharged)<<" cut value "<<isocut2<<" etPhoton "<<etPhoton<<std::endl;
    //    std::cout<<" PtHadron "<<ptMaxHadron<<" "<<4.5+etPhoton/40<<std::endl;

    if (etCone - etPhoton > 5 + etPhoton / 20 - etPhoton * etPhoton / 1e4)
      continue;
    //    std::cout<<" Accept 1"<<std::endl;
    if (etCone - etPhoton - (etConeCharged - etPhotonCharged) > isocut2)
      continue;
    //     std::cout<<" Accept 2"<<std::endl;
    if (ptMaxHadron > isocut3)
      continue;

    //    std::cout<<"Accept event "<<std::endl;
    accepted = true;
    break;

  }  //loop over seeds

  if (accepted) {
    theNumberOfSelected++;
    std::cout << " Event preselected " << theNumberOfSelected << " Proccess ID " << myGenEvent->signal_process_id()
              << std::endl;
    return true;
  } else
    return false;
}
