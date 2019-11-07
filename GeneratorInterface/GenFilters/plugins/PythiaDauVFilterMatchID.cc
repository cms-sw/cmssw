#include "GeneratorInterface/GenFilters/plugins/PythiaDauVFilterMatchID.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>
#include <vector>

using namespace edm;
using namespace std;
using namespace Pythia8;

PythiaDauVFilterMatchID::PythiaDauVFilterMatchID(const edm::ParameterSet& iConfig)
    : fVerbose(iConfig.getUntrackedParameter("verbose", 0)),
      token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      particleID(iConfig.getUntrackedParameter("ParticleID", 0)),
      motherID(iConfig.getUntrackedParameter("MotherID", 0)),
      chargeconju(iConfig.getUntrackedParameter("ChargeConjugation", true)),
      ndaughters(iConfig.getUntrackedParameter("NumberDaughters", 0)),
      maxptcut(iConfig.getUntrackedParameter("MaxPt", 14000.)) {
  //now do what ever initialization is needed
  vector<int> defdauID;
  defdauID.push_back(0);
  dauIDs = iConfig.getUntrackedParameter<vector<int> >("DaughterIDs", defdauID);
  vector<double> defminptcut;
  defminptcut.push_back(0.);
  minptcut = iConfig.getUntrackedParameter<vector<double> >("MinPt", defminptcut);
  vector<double> defminetacut;
  defminetacut.push_back(-10.);
  minetacut = iConfig.getUntrackedParameter<vector<double> >("MinEta", defminetacut);
  vector<double> defmaxetacut;
  defmaxetacut.push_back(10.);
  maxetacut = iConfig.getUntrackedParameter<vector<double> >("MaxEta", defmaxetacut);

  edm::LogInfo("PythiaDauVFilterMatchID")
      << "----------------------------------------------------------------------" << endl;
  edm::LogInfo("PythiaDauVFilterMatchID") << "--- PythiaDauVFilterMatchID" << endl;
  for (unsigned int i = 0; i < dauIDs.size(); ++i) {
    edm::LogInfo("PythiaDauVFilterMatchID")
        << "ID: " << dauIDs[i] << " pT > " << minptcut[i] << " " << minetacut[i] << " eta < " << maxetacut[i] << endl;
  }
  edm::LogInfo("PythiaDauVFilterMatchID") << "maxptcut   = " << maxptcut << endl;
  edm::LogInfo("PythiaDauVFilterMatchID") << "particleID = " << particleID << endl;
  edm::LogInfo("PythiaDauVFilterMatchID") << "motherID   = " << motherID << endl;

  // create pythia8 instance to access particle data
  edm::LogInfo("PythiaDauVFilterMatchID") << "Creating pythia8 instance for particle properties" << endl;
  if (!fLookupGen.get())
    fLookupGen.reset(new Pythia());
}

PythiaDauVFilterMatchID::~PythiaDauVFilterMatchID() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool PythiaDauVFilterMatchID::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  bool accepted = false;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  int OK(1);
  vector<int> vparticles;

  HepMC::GenEvent* myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));

  if (fVerbose > 5) {
    edm::LogInfo("PythiaDauVFilterMatchID") << "looking for " << particleID << endl;
  }

  for (HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p) {
    if ((*p)->pdg_id() != particleID)
      continue;

    // -- Check for mother of this particle
    if (0 != motherID) {
      OK = 0;
      for (HepMC::GenVertex::particles_in_const_iterator des = (*p)->production_vertex()->particles_in_const_begin();
           des != (*p)->production_vertex()->particles_in_const_end();
           ++des) {
        if (fVerbose > 10) {
          edm::LogInfo("PythiaDauVFilterMatchID")
              << "mother: " << (*des)->pdg_id() << " pT: " << (*des)->momentum().perp()
              << " eta: " << (*des)->momentum().eta() << endl;
        }
        if (abs(motherID) == abs((*des)->pdg_id())) {
          OK = 1;
          break;
        }
      }
    }
    if (0 == OK)
      continue;

    //generate targets
    std::vector<decayTarget> targets;
    for (unsigned int i = 0; i < dauIDs.size(); i++) {
      decayTarget target;
      target.pdgID = dauIDs[i];
      target.minPt = minptcut[i];
      target.maxPt = maxptcut;
      target.minEta = minetacut[i];
      target.maxEta = maxetacut[i];
      targets.push_back(target);
    }
    if (fVerbose > 10) {
      edm::LogInfo("PythiaDauVFilterMatchID") << "created target: ";
      for (unsigned int i = 0; i < targets.size(); i++) {
        edm::LogInfo("PythiaDauVFilterMatchID") << targets[i].pdgID << " ";
      }
      edm::LogInfo("PythiaDauVFilterMatchID") << endl;
    }

    // -- check for daugthers
    int ndau = 0;
    if (fVerbose > 5) {
      edm::LogInfo("PythiaDauVFilterMatchID") << "found ID: " << (*p)->pdg_id() << " pT: " << (*p)->momentum().perp()
                                              << " eta: " << (*p)->momentum().eta() << endl;
    }
    vparticles.push_back((*p)->pdg_id());
    if ((*p)->end_vertex()) {
      for (HepMC::GenVertex::particle_iterator des = (*p)->end_vertex()->particles_begin(HepMC::children);
           des != (*p)->end_vertex()->particles_end(HepMC::children);
           ++des) {
        if (TMath::Abs((*des)->pdg_id()) == 22) {
          continue;
        }
        ++ndau;
        if (fVerbose > 5) {
          edm::LogInfo("PythiaDauVFilterMatchID") << "ID: " << (*des)->pdg_id() << " pT: " << (*des)->momentum().perp()
                                                  << " eta: " << (*des)->momentum().eta() << endl;
        }
        {  // protect matchedIdx
          int matchedIdx = -1;
          for (unsigned int i = 0; i < targets.size(); i++) {
            if ((*des)->pdg_id() != targets[i].pdgID) {
              continue;
            }
            if (fVerbose > 5) {
              edm::LogInfo("PythiaDauVFilterMatchID") << "i = " << i << " pT = " << (*des)->momentum().perp()
                                                      << " eta = " << (*des)->momentum().eta() << endl;
            }

            if ((*des)->momentum().perp() > targets[i].minPt && (*des)->momentum().perp() < targets[i].maxPt &&
                (*des)->momentum().eta() > targets[i].minEta && (*des)->momentum().eta() < targets[i].maxEta) {
              vparticles.push_back((*des)->pdg_id());
              if (fVerbose > 2) {
                edm::LogInfo("PythiaDauVFilterMatchID")
                    << "  accepted this particle " << (*des)->pdg_id() << " pT = " << (*des)->momentum().perp()
                    << " eta = " << (*des)->momentum().eta() << endl;
              }
              matchedIdx = i;
              break;
            }
          }
          if (matchedIdx > -1) {
            targets.erase(targets.begin() + matchedIdx);
          }
          if (fVerbose > 10) {
            edm::LogInfo("PythiaDauVFilterMatchID") << "remaining targets: ";
            for (unsigned int i = 0; i < targets.size(); i++) {
              edm::LogInfo("PythiaDauVFilterMatchID") << targets[i].pdgID << " ";
            }
            edm::LogInfo("PythiaDauVFilterMatchID") << endl;
          }
        }
      }
    }

    if (ndau == ndaughters && targets.empty()) {
      accepted = true;
      if (fVerbose > 0) {
        edm::LogInfo("PythiaDauVFilterMatchID") << "  accepted this decay: ";
        for (unsigned int iv = 0; iv < vparticles.size(); ++iv)
          edm::LogInfo("PythiaDauVFilterMatchID") << vparticles[iv] << " ";
        edm::LogInfo("PythiaDauVFilterMatchID") << " from mother = " << motherID << endl;
      }
      break;
    }
  }

  int anti_particleID = -particleID;
  if (!(fLookupGen->particleData.isParticle(anti_particleID))) {
    anti_particleID = 0;
    if (fVerbose > 5)
      edm::LogInfo("PythiaDauVFilterMatchID")
          << "Particle " << particleID << " is its own anti-particle, skipping further testing " << endl;
  }
  if (!accepted && chargeconju && anti_particleID) {
    OK = 1;

    for (HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p) {
      if ((*p)->pdg_id() != anti_particleID)
        continue;

      // -- Check for mother of this particle
      if (0 != motherID) {
        OK = 0;
        for (HepMC::GenVertex::particles_in_const_iterator des = (*p)->production_vertex()->particles_in_const_begin();
             des != (*p)->production_vertex()->particles_in_const_end();
             ++des) {
          if (fVerbose > 10) {
            edm::LogInfo("PythiaDauVFilterMatchID")
                << "mother: " << (*des)->pdg_id() << " pT: " << (*des)->momentum().perp()
                << " eta: " << (*des)->momentum().eta() << endl;
          }
          if (abs(motherID) == abs((*des)->pdg_id())) {
            OK = 1;
            break;
          }
        }
      }
      if (0 == OK)
        continue;

      //generate anti targets
      std::vector<decayTarget> targets;
      for (unsigned int i = 0; i < dauIDs.size(); i++) {
        decayTarget target;
        int IDanti = -dauIDs[i];
        if (!(fLookupGen->particleData.isParticle(IDanti)))
          IDanti = dauIDs[i];
        target.pdgID = IDanti;
        target.minPt = minptcut[i];
        target.maxPt = maxptcut;
        target.minEta = minetacut[i];
        target.maxEta = maxetacut[i];
        targets.push_back(target);
      }
      if (fVerbose > 10) {
        edm::LogInfo("PythiaDauVFilterMatchID") << "created anti target: ";
        for (unsigned int i = 0; i < targets.size(); i++) {
          edm::LogInfo("PythiaDauVFilterMatchID") << targets[i].pdgID << " ";
        }
        edm::LogInfo("PythiaDauVFilterMatchID") << endl;
      }

      if (fVerbose > 5) {
        edm::LogInfo("PythiaDauVFilterMatchID") << "found ID: " << (*p)->pdg_id() << " pT: " << (*p)->momentum().perp()
                                                << " eta: " << (*p)->momentum().eta() << endl;
      }
      vparticles.push_back((*p)->pdg_id());
      int ndau = 0;
      if ((*p)->end_vertex()) {
        for (HepMC::GenVertex::particle_iterator des = (*p)->end_vertex()->particles_begin(HepMC::children);
             des != (*p)->end_vertex()->particles_end(HepMC::children);
             ++des) {
          if (TMath::Abs((*des)->pdg_id()) == 22) {
            continue;
          }
          ++ndau;
          if (fVerbose > 5) {
            edm::LogInfo("PythiaDauVFilterMatchID")
                << "ID: " << (*des)->pdg_id() << " pT: " << (*des)->momentum().perp()
                << " eta: " << (*des)->momentum().eta() << endl;
          }
          {  // protect matchedIdx
            int matchedIdx = -1;
            for (unsigned int i = 0; i < targets.size(); i++) {
              if ((*des)->pdg_id() != targets[i].pdgID) {
                continue;
              }
              if (fVerbose > 5) {
                edm::LogInfo("PythiaDauVFilterMatchID") << "i = " << i << " pT = " << (*des)->momentum().perp()
                                                        << " eta = " << (*des)->momentum().eta() << endl;
              }

              if ((*des)->momentum().perp() > targets[i].minPt && (*des)->momentum().perp() < targets[i].maxPt &&
                  (*des)->momentum().eta() > targets[i].minEta && (*des)->momentum().eta() < targets[i].maxEta) {
                vparticles.push_back((*des)->pdg_id());
                if (fVerbose > 2) {
                  edm::LogInfo("PythiaDauVFilterMatchID")
                      << "  accepted this particle " << (*des)->pdg_id() << " pT = " << (*des)->momentum().perp()
                      << " eta = " << (*des)->momentum().eta() << endl;
                }
                matchedIdx = i;
                break;
              }
            }
            if (matchedIdx > -1) {
              targets.erase(targets.begin() + matchedIdx);
            }
            if (fVerbose > 10) {
              edm::LogInfo("PythiaDauVFilterMatchID") << "remaining targets: ";
              for (unsigned int i = 0; i < targets.size(); i++) {
                edm::LogInfo("PythiaDauVFilterMatchID") << targets[i].pdgID << " ";
              }
              edm::LogInfo("PythiaDauVFilterMatchID") << endl;
            }
          }
        }
      }
      if (ndau == ndaughters && targets.empty()) {
        accepted = true;
        if (fVerbose > 0) {
          edm::LogInfo("PythiaDauVFilterMatchID") << "  accepted this decay: ";
          for (unsigned int iv = 0; iv < vparticles.size(); ++iv)
            edm::LogInfo("PythiaDauVFilterMatchID") << vparticles[iv] << " ";
          edm::LogInfo("PythiaDauVFilterMatchID") << " from mother = " << motherID << endl;
        }
        break;
      }
    }
  }

  delete myGenEvent;
  return accepted;
}
