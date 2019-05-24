// -*- C++ -*-
//
// Package:    BsToPhiPhiTo4KFilter
// Class:      BsToPhiPhiTo4KFilter
// 
/**\class BsToPhiPhiTo4KFilter L1Trigger/L1TTrackMatch/test/BsToPhiPhiTo4KFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

// Gen-level stuff
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "TH1F.h"

using namespace l1t;

class BsToPhiPhiTo4KFilter: public edm::EDFilter {
public:

  explicit BsToPhiPhiTo4KFilter(const edm::ParameterSet&);
  ~BsToPhiPhiTo4KFilter() {}
  
private:
  virtual void beginJob() override;
  virtual bool filter(edm::Event&, edm::EventSetup const&) override;
  virtual void endJob() override;

  bool matchGenInfo(const reco::Candidate* a, const reco::Candidate* b);
  bool passedGenFilter(const reco::GenParticleCollection& genParticles, bool verbose=false);

  const edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken_;
};

BsToPhiPhiTo4KFilter::BsToPhiPhiTo4KFilter(const edm::ParameterSet& iConfig):
  genParticleToken_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("GenParticleInputTag")))
{
}
void BsToPhiPhiTo4KFilter::beginJob() {
}
bool BsToPhiPhiTo4KFilter::filter(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  using namespace edm;

  std::cout << "Entering BsToPhiPhiTo4KFilter" << std::endl;
  // Apply Gen Filter
  edm::Handle<reco::GenParticleCollection> genParticleHandle;
  bool res = iEvent.getByToken(genParticleToken_, genParticleHandle);
  if (res && genParticleHandle.isValid()) {
    reco::GenParticleCollection genParticles = (*genParticleHandle.product());
    if (passedGenFilter(genParticles, false)) return true;
  }
  else {
    std::cerr << "filter: GenParticleCollection for InputTag genParticles not found!" << std::endl; 
  }
  return false;
}
bool BsToPhiPhiTo4KFilter::passedGenFilter(const reco::GenParticleCollection& genParticles, bool verbose) {
  auto gbeg = genParticles.begin(); 
  if (verbose) {
    std::cout << std::setiosflags(std::ios::fixed);
    std::cout << std::setprecision(2);
    std::cout << "indx    status    pdgId  charge     eta      phi      pt     energy             mID                             dID"
	      << std::endl;
  }
  int nphi = 0;  
  size_t i = 0;
  for (auto it = genParticles.begin(); it != genParticles.end(); ++it) {
    if (std::abs(it->pdgId()) != 333) continue; // Phi

    // First mother
    int idm = -1;
    const reco::Candidate* m = it->mother();
    if (m != nullptr) {
      for (auto mit = genParticles.begin(); mit != genParticles.end(); ++mit) {
        const reco::Candidate* ap = &(*mit);
        if (matchGenInfo(m, ap)) {
	  idm = std::distance(gbeg, mit);
  	  break;
        }
      }
    }
    int motherIndex = idm;
    if (motherIndex < 0 || std::abs(m->pdgId()) != 531) continue; // Bs

    std::vector<int> daughterIndices;
    std::ostringstream dID;
    for (size_t j = 0; j < it->numberOfDaughters(); ++j) {
      const reco::Candidate* d = it->daughter(j);
      if (std::abs(d->pdgId()) == 321 && d->pt() > 2.0 && std::fabs(d->eta()) < 2.5) {
	//if (!(std::abs(d->pdgId()) == 321 && d->pt() > 2.0)) continue;
	for (auto dit = genParticles.begin(); dit != genParticles.end(); ++dit) {
	  const reco::Candidate* ap = &(*dit);  
	  if (matchGenInfo(d, ap)) {
	    int idd = std::distance(gbeg, dit);
	    daughterIndices.push_back(idd);
	    dID << " " << idd;
	    break;
	  }
	}
      }
    }
    if (daughterIndices.size() < 2) continue;
    ++nphi;
    if (verbose) {
      std::string ds = dID.str();
      if (!ds.length()) ds = " -";
      std::cout << std::setw(4)  << i++
		<< std::setw(8)  << it->status()
		<< std::setw(10) << it->pdgId()
		<< std::setw(8)  << it->charge()
		<< std::setw(10) << it->eta()
		<< std::setw(9)  << it->phi()
		<< std::setw(9)  << it->pt()
		<< std::setw(9)  << it->energy()
		<< std::setw(16) << idm
		<< ds
		<< std::endl;
    }
  }
  std::cout << std::resetiosflags(std::ios::fixed);
  return (nphi > 1);
}
bool BsToPhiPhiTo4KFilter::matchGenInfo(const reco::Candidate* a, const reco::Candidate* b) {
  if ( a->pdgId()  == b->pdgId()  &&
       a->status() == b->status() &&        
       a->pt()     == b->pt()     &&        
       a->eta()    == b->eta()    &&       
       a->phi()    == b->phi() ) return true;
  return false;
}
void BsToPhiPhiTo4KFilter::endJob() {
}
// define this as a plug-in
DEFINE_FWK_MODULE(BsToPhiPhiTo4KFilter);
