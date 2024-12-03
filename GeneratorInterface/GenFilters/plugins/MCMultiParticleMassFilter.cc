#include "GeneratorInterface/GenFilters/plugins/MCFilterZboostHelper.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <vector>
#include <iostream>
#include <unordered_map>
#include <tuple>

class MCMultiParticleMassFilter : public edm::global::EDFilter<> {
public:
  explicit MCMultiParticleMassFilter(const edm::ParameterSet&);
  ~MCMultiParticleMassFilter() override;

private:
  bool filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override;
  bool recurseLoop(std::vector<HepMC::GenParticle*>& particlesThatPassCuts, std::vector<int> indices, int depth) const;

  /* Member data */
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const std::vector<int> particleID;
  std::vector<double> ptMin;
  std::vector<double> etaMax;
  std::vector<int> status;

  //Maps each particle ID provided to its required pt, max eta, and status
  std::unordered_map<int, std::tuple<double, double, int>> cutPerParticle;

  const double minTotalMass;
  const double maxTotalMass;

  double minTotalMassSq;
  double maxTotalMassSq;
  int nParticles;

  int particleSumTo;
  int particleProdTo;
};

using namespace edm;
using namespace std;

MCMultiParticleMassFilter::MCMultiParticleMassFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          iConfig.getUntrackedParameter<edm::InputTag>("src", edm::InputTag("generator", "unsmeared")))),
      particleID(iConfig.getParameter<std::vector<int>>("ParticleID")),
      ptMin(iConfig.getParameter<std::vector<double>>("PtMin")),
      etaMax(iConfig.getParameter<std::vector<double>>("EtaMax")),
      status(iConfig.getParameter<std::vector<int>>("Status")),
      minTotalMass(iConfig.getParameter<double>("minTotalMass")),
      maxTotalMass(iConfig.getParameter<double>("maxTotalMass")) {
  nParticles = particleID.size();
  minTotalMassSq = minTotalMass * minTotalMass;
  maxTotalMassSq = maxTotalMass * maxTotalMass;

  //These two values dictate what particles it accepts as combinations
  particleSumTo = 0;
  particleProdTo = 1;
  for (const int i : particleID) {
    particleSumTo += i;
    particleProdTo *= i;
  }

  // if any of the vectors are of size 1, take that to mean it is a new default
  double defaultPtMin = 1.8;
  if ((int)ptMin.size() == 1) {
    defaultPtMin = ptMin[0];
  }

  if ((int)ptMin.size() < nParticles) {
    edm::LogWarning("MCMultiParticleMassFilter") << "Warning: Given pT value size"
                                                    "< size of the number of particle IDs."
                                                    " Filling remaining values with "
                                                 << defaultPtMin << endl;
    while ((int)ptMin.size() < nParticles) {
      ptMin.push_back(defaultPtMin);
    }
  } else if ((int)ptMin.size() > nParticles) {
    edm::LogWarning("MCMultiParticleMassFilter") << "Warning: Given pT value size"
                                                    "> size of the number of particle IDs."
                                                    " Ignoring extraneous values "
                                                 << endl;
    ptMin.resize(nParticles);
  }

  double defaultEtaMax = 2.7;
  if ((int)etaMax.size() == 1) {
    defaultEtaMax = etaMax[0];
  }
  if ((int)etaMax.size() < nParticles) {
    edm::LogWarning("MCMultiParticleMassFilter") << "Warning: Given eta value size"
                                                    "< size of the number of particle IDs."
                                                    " Filling remaining values with "
                                                 << defaultEtaMax << endl;
    while ((int)etaMax.size() < nParticles) {
      etaMax.push_back(defaultEtaMax);
    }
  } else if ((int)etaMax.size() > nParticles) {
    edm::LogWarning("MCMultiParticleMassFilter") << "Warning: Given eta value size"
                                                    "> size of the number of particle IDs."
                                                    " Ignoring extraneous values "
                                                 << endl;
    etaMax.resize(nParticles);
  }

  int defaultStatus = 1;
  if ((int)status.size() == 1) {
    defaultStatus = status[0];
  }
  if ((int)status.size() < nParticles) {
    edm::LogWarning("MCMultiParticleMassFilter") << "Warning: Given status value size"
                                                    "< size of the number of particle IDs."
                                                    " Filling remaining values with "
                                                 << defaultStatus << endl;
    while ((int)status.size() < nParticles) {
      status.push_back(defaultStatus);
    }
  } else if ((int)status.size() > nParticles) {
    edm::LogWarning("MCMultiParticleMassFilter") << "Warning: Given status value size"
                                                    "> size of the number of particle IDs."
                                                    " Ignoring extraneous values "
                                                 << endl;
    status.resize(nParticles);
  }

  for (int i = 0; i < nParticles; i++) {
    std::tuple<double, double, int> cutForParticle = std::make_tuple(ptMin[i], etaMax[i], status[i]);
    cutPerParticle[particleID[i]] = cutForParticle;
  }  //assign the set of cuts you decided upon matched to each ID value in order
}

MCMultiParticleMassFilter::~MCMultiParticleMassFilter() {}

bool MCMultiParticleMassFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  std::vector<HepMC::GenParticle*> particlesThatPassCuts = std::vector<HepMC::GenParticle*>();
  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    for (const int i : particleID) {
      if (i == (*p)->pdg_id()) {
        //if the particle ID is one of the ones you specified, check for cuts per ID
        const auto cuts = cutPerParticle.at(i);
        if (((*p)->status() == get<2>(cuts)) && ((*p)->momentum().perp() >= get<0>(cuts)) &&
            (std::fabs((*p)->momentum().eta()) <= get<1>(cuts))) {
          particlesThatPassCuts.push_back(*p);
          break;
        }
      }
    }
  }
  int nIterables = particlesThatPassCuts.size();
  if (nIterables < nParticles) {
    return false;
  } else {
    int i = 0;
    //only iterate while there are enough particles that pass cuts
    while ((nIterables - i) >= nParticles) {
      vector<int> indices;
      //start recursing from index 0, 1, 2, ...
      indices.push_back(i);
      bool success = recurseLoop(particlesThatPassCuts, indices, 1);
      if (success) {
        return true;
      }
      i++;
    }
  }
  return false;
}

bool MCMultiParticleMassFilter::recurseLoop(std::vector<HepMC::GenParticle*>& particlesThatPassCuts,
                                            std::vector<int> indices,
                                            int depth) const {
  int lastIndex = indices.back();
  int nIterables = (int)(particlesThatPassCuts.size());
  if (lastIndex >= nIterables) {
    return false;
  } else if (depth == nParticles) {
    //you have the right number of particles!
    int tempSum = 0;
    int tempProd = 1;

    double px, py, pz, e;
    px = py = pz = e = 0;
    for (const int i : indices) {
      int id = particlesThatPassCuts[i]->pdg_id();
      tempSum += id;
      tempProd *= id;
      HepMC::FourVector tempVec = particlesThatPassCuts[i]->momentum();
      px += tempVec.px();
      py += tempVec.py();
      pz += tempVec.pz();
      e += tempVec.e();
    }
    if ((tempSum != particleSumTo) || (tempProd != particleProdTo)) {
      return false;  //check if the ids are what you want
    }
    double invMassSq = e * e - px * px - py * py - pz * pz;
    //taking the root is computationally expensive, so use the squared term
    if ((invMassSq >= minTotalMassSq) && (invMassSq <= maxTotalMassSq)) {
      return true;
    }
    return false;
  } else {
    vector<bool> recursionResult;
    //propagate recursion across all combinations of remaining indices
    for (int i = 1; i < nIterables - lastIndex; i++) {
      vector<int> newIndices = indices;
      newIndices.push_back(lastIndex + i);
      //always up the depth by 1 to make sure there is no infinite recursion
      recursionResult.push_back(recurseLoop(particlesThatPassCuts, newIndices, depth + 1));
    }
    //search the results to look for one successful combination
    for (bool r : recursionResult) {
      if (r) {
        return true;
      }
    }
    return false;
  }
}
DEFINE_FWK_MODULE(MCMultiParticleMassFilter);

