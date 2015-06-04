#ifndef HeavyFlavorAnalysis_Onia2MuMu_Onia2MuMuPAT_h
#define HeavyFlavorAnalysis_Onia2MuMu_Onia2MuMuPAT_h


// system include files
#include <memory>

// FW include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/PtComparator.h"

// DataFormat includes
#include <DataFormats/PatCandidates/interface/CompositeCandidate.h>
#include <DataFormats/PatCandidates/interface/Muon.h>

#include <CommonTools/UtilAlgos/interface/StringCutObjectSelector.h>
#include "RecoVertex/VertexTools/interface/InvariantMassFromVertex.h"

template<typename T>
struct GreaterByVProb {
  typedef T first_argument_type;
  typedef T second_argument_type;
  bool operator()( const T & t1, const T & t2 ) const {
    return t1.userFloat("vProb") > t2.userFloat("vProb");
  }
};


//
// class decleration
//

class Onia2MuMuPAT : public edm::EDProducer {
 public:
  explicit Onia2MuMuPAT(const edm::ParameterSet&);
  ~Onia2MuMuPAT();

 private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  bool isAbHadron(int pdgID);
  bool isAMixedbHadron(int pdgID, int momPdgID);
  std::pair<int, float> findJpsiMCInfo(reco::GenParticleRef genJpsi);

  // ----------member data ---------------------------
 private:
  edm::InputTag muons_;
  edm::InputTag thebeamspot_;
  edm::InputTag thePVs_;
  StringCutObjectSelector<pat::Muon> higherPuritySelection_;
  StringCutObjectSelector<pat::Muon> lowerPuritySelection_; 
  StringCutObjectSelector<reco::Candidate, true> dimuonSelection_;
  bool addCommonVertex_, addMuonlessPrimaryVertex_;
  bool resolveAmbiguity_;
  bool addMCTruth_;
  GreaterByPt<pat::CompositeCandidate> pTComparator_;
  GreaterByVProb<pat::CompositeCandidate> vPComparator_;

  InvariantMassFromVertex massCalculator;

};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//

#endif
