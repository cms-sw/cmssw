#include "PythiaAllDauVFilter.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>
#include <memory>
#include <vector>

using namespace edm;
using namespace std;
using namespace Pythia8;

PythiaAllDauVFilter::PythiaAllDauVFilter(const edm::ParameterSet& iConfig)
    : fVerbose(iConfig.getUntrackedParameter("verbose", 0)),
      token_(consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter<edm::InputTag>("moduleLabel"))),
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

  // create pythia8 instance to access particle data
  edm::LogInfo("PythiaAllDauVFilter") << "Creating pythia8 instance for particle properties" << endl;
  if (!fLookupGen.get())
    fLookupGen = std::make_unique<Pythia>();

  if (chargeconju) {
    antiParticleID = -particleID;
    if (!(fLookupGen->particleData.isParticle(antiParticleID)))
      antiParticleID = particleID;

    int antiId;
    for (size_t i = 0; i < dauIDs.size(); i++) {
      antiId = -dauIDs[i];
      if (!(fLookupGen->particleData.isParticle(antiId)))
        antiId = dauIDs[i];

      antiDauIDs.push_back(antiId);
    }
  }

  edm::LogInfo("PythiaAllDauVFilter") << "----------------------------------------------------------------------"
                                      << endl;
  edm::LogInfo("PythiaAllDauVFilter") << "--- PythiaAllDauVFilter" << endl;
  for (unsigned int i = 0; i < dauIDs.size(); ++i) {
    edm::LogInfo("PythiaAllDauVFilter") << "ID: " << dauIDs[i] << " pT > " << minptcut[i] << " " << minetacut[i]
                                        << " eta < " << maxetacut[i] << endl;
  }
  if (chargeconju)
    for (unsigned int i = 0; i < antiDauIDs.size(); ++i) {
      edm::LogInfo("PythiaAllDauVFilter") << "ID: " << antiDauIDs[i] << " pT > " << minptcut[i] << " " << minetacut[i]
                                          << " eta < " << maxetacut[i] << endl;
    }
  edm::LogInfo("PythiaAllDauVFilter") << "maxptcut   = " << maxptcut << endl;
  edm::LogInfo("PythiaAllDauVFilter") << "particleID = " << particleID << endl;
  if (chargeconju)
    edm::LogInfo("PythiaAllDauVFilter") << "antiParticleID = " << antiParticleID << endl;

  edm::LogInfo("PythiaAllDauVFilter") << "motherID   = " << motherID << endl;
}

PythiaAllDauVFilter::~PythiaAllDauVFilter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool PythiaAllDauVFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  bool accepted = false;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  int OK(1);
  vector<int> vparticles;
  vector<bool> foundDaughter(dauIDs.size(), false);
  auto dauCollection = &dauIDs;

  HepMC::GenEvent* myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));

  if (fVerbose > 5) {
    edm::LogInfo("PythiaAllDauVFilter") << "looking for " << particleID << endl;
  }

  for (HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p) {
    if ((*p)->pdg_id() == particleID) {
      dauCollection = &(dauIDs);
    } else if (chargeconju and ((*p)->pdg_id() == antiParticleID)) {
      dauCollection = &(antiDauIDs);
    } else {
      continue;
    }

    // -- Check for mother of this particle
    if (0 != motherID) {
      OK = 0;
      for (HepMC::GenVertex::particles_in_const_iterator des = (*p)->production_vertex()->particles_in_const_begin();
           des != (*p)->production_vertex()->particles_in_const_end();
           ++des) {
        if (fVerbose > 10) {
          edm::LogInfo("PythiaAllDauVFilter") << "mother: " << (*des)->pdg_id() << " pT: " << (*des)->momentum().perp()
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

    // -- check for daugthers
    int ndau = 0;
    for (unsigned int i = 0; i < foundDaughter.size(); ++i) {
      foundDaughter[i] = false;
    }
    if (fVerbose > 5) {
      edm::LogInfo("PythiaAllDauVFilter") << "found ID: " << (*p)->pdg_id() << " pT: " << (*p)->momentum().perp()
                                          << " eta: " << (*p)->momentum().eta() << endl;
    }
    if ((*p)->end_vertex()) {
      for (HepMC::GenVertex::particle_iterator des = (*p)->end_vertex()->particles_begin(HepMC::children);
           des != (*p)->end_vertex()->particles_end(HepMC::children);
           ++des) {
        ++ndau;
        if (fVerbose > 5) {
          edm::LogInfo("PythiaAllDauVFilter")
              << "\t daughter : ID: " << (*des)->pdg_id() << " pT: " << (*des)->momentum().perp()
              << " eta: " << (*des)->momentum().eta() << endl;
        }
        for (unsigned int i = 0; i < dauCollection->size(); ++i) {
          if ((*des)->pdg_id() != dauCollection->at(i))
            continue;

          // possible to have more than one daughter of same pdgID and same/different kinematic constraints
          if (foundDaughter[i])
            continue;

          if (fVerbose > 5) {
            edm::LogInfo("PythiaAllDauVFilter")
                << "\t\t checking cuts of , daughter i = " << i << " pT = " << (*des)->momentum().perp()
                << " eta = " << (*des)->momentum().eta() << endl;
          }
          if ((*des)->momentum().perp() > minptcut[i] && (*des)->momentum().perp() < maxptcut &&
              (*des)->momentum().eta() > minetacut[i] && (*des)->momentum().eta() < maxetacut[i]) {
            foundDaughter[i] = true;
            vparticles.push_back((*des)->pdg_id());
            if (fVerbose > 2) {
              edm::LogInfo("PythiaAllDauVFilter")
                  << "\t  accepted this particle " << (*des)->pdg_id() << " pT = " << (*des)->momentum().perp()
                  << " eta = " << (*des)->momentum().eta() << endl;
            }
            break;
          }
        }
      }
    }

    // -- ( number of daughtrs == daughters passing cut ) and ( all daughters specified are found)
    if (ndau == ndaughters) {
      accepted = true;
      for (unsigned int i = 0; i < foundDaughter.size(); ++i) {
        if (!foundDaughter[i]) {
          accepted = false;
        }
      }
      if (accepted and (fVerbose > 0)) {
        edm::LogInfo("PythiaAllDauVFilter") << "  accepted this decay from " << (*p)->pdg_id();
        for (unsigned int iv = 0; iv < vparticles.size(); ++iv)
          edm::LogInfo("PythiaAllDauVFilter") << vparticles[iv] << " ";
        edm::LogInfo("PythiaAllDauVFilter") << " from mother = " << motherID << endl;
      }
    }

    if (accepted)
      break;
  }

  delete myGenEvent;
  return accepted;
}

//define this as a plug-in
DEFINE_FWK_MODULE(PythiaAllDauVFilter);
