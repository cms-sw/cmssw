/*
Package:    GeneralInterface/GenFilters/ModelpMSSMFilter
Class:      ModelpMSSMFilter

class ModelpMSSMFilter ModelpMSSMFilter.cc GeneratorInterface/GenFilters/plugins/ModelpMSSMFilter.cc

Description: EDFilter which checks the event passes a baseline selection for the run-II pMSSM effort.

Implementation: 

The following input parameters are used 
  gpssrc = cms.InputTag("X") : gen particle collection label as input
  jetsrc  = cms.InputTag("X") : genjet collection collection label as input
  jetPtCut = cms.double(#) : GenJet pT cut for HT
  jetEtaCut = cms.double(#) : GenJet eta cut for HT
  genHTcut = cms.double(#) : GenHT cut
  muPtCut = cms.double(#) : muon pT cut
  muEtaCut = cms.double(#) : muon eta cut
  elPtCut = cms.double(#) : electron pT cut
  elEtaCut = cms.double(#) : electron eta cut
  gammaPtCut = cms.double(#) : photon pT cut
  gammaEtaCut = cms.double(#) : photon eta cut
  loosemuPtCut = cms.double(#) : loose muon pT cut
  looseelPtCut = cms.double(#) : loose electron pT cut
  loosegammaPtCut = cms.double(#) : loose photon pT cut
  veryloosegammaPtCut = cms.double(#) : even looser photon pT cut
Original Author:  Malte Mrowietz
         Created:  Jun 2019
*/

//System include files
#include <cmath>
#include <memory>
#include <vector>
//User include files
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

//Class declaration
class ModelpMSSMFilter : public edm::global::EDFilter<> {
public:
  explicit ModelpMSSMFilter(const edm::ParameterSet&);
  ~ModelpMSSMFilter() override;

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  //Member data
  edm::EDGetTokenT<reco::GenParticleCollection> token_;
  edm::EDGetTokenT<reco::GenJetCollection> token2_;
  double muPtCut_, muEtaCut_, tauPtCut_, tauEtaCut_, elPtCut_, elEtaCut_, gammaPtCut_, gammaEtaCut_, loosemuPtCut_,
      looseelPtCut_, loosegammaPtCut_, veryloosegammaPtCut_, jetPtCut_, jetEtaCut_, genHTcut_;
};

//Constructor
ModelpMSSMFilter::ModelpMSSMFilter(const edm::ParameterSet& params)
    : token_(consumes<reco::GenParticleCollection>(params.getParameter<edm::InputTag>("gpssrc"))),
      token2_(consumes<reco::GenJetCollection>(params.getParameter<edm::InputTag>("jetsrc"))),
      muPtCut_(params.getParameter<double>("muPtCut")),
      muEtaCut_(params.getParameter<double>("muEtaCut")),
      tauPtCut_(params.getParameter<double>("tauPtCut")),
      tauEtaCut_(params.getParameter<double>("tauEtaCut")),
      elPtCut_(params.getParameter<double>("elPtCut")),
      elEtaCut_(params.getParameter<double>("elEtaCut")),
      gammaPtCut_(params.getParameter<double>("gammaPtCut")),
      gammaEtaCut_(params.getParameter<double>("gammaEtaCut")),
      loosemuPtCut_(params.getParameter<double>("loosemuPtCut")),
      looseelPtCut_(params.getParameter<double>("looseelPtCut")),
      loosegammaPtCut_(params.getParameter<double>("loosegammaPtCut")),
      veryloosegammaPtCut_(params.getParameter<double>("veryloosegammaPtCut")),
      jetPtCut_(params.getParameter<double>("jetPtCut")),
      jetEtaCut_(params.getParameter<double>("jetEtaCut")),
      genHTcut_(params.getParameter<double>("genHTcut")) {}

//Destructor
ModelpMSSMFilter::~ModelpMSSMFilter() {}

bool ModelpMSSMFilter::filter(edm::StreamID, edm::Event& evt, const edm::EventSetup& params) const {
  using namespace std;
  using namespace edm;
  using namespace reco;
  edm::Handle<reco::GenParticleCollection> gps;
  evt.getByToken(token_, gps);
  edm::Handle<reco::GenJetCollection> generatedJets;
  evt.getByToken(token2_, generatedJets);
  int looseel = 0;
  int loosemu = 0;
  int loosegamma = 0;
  int veryloosegamma = 0;
  float decaylength;
  for (std::vector<reco::GenParticle>::const_iterator it = gps->begin(); it != gps->end(); ++it) {
    const reco::GenParticle& gp = *it;
    if (gp.isLastCopy()) {
      if (fabs(gp.pdgId()) == 15) {
        if (gp.pt() > tauPtCut_ && fabs(gp.eta()) < tauEtaCut_) {
          return true;
        }
      }
      if (fabs(gp.pdgId()) == 13) {
        if (gp.pt() > muPtCut_ && fabs(gp.eta()) < muEtaCut_) {
          return true;
        }
        if (gp.pt() > loosemuPtCut_ && fabs(gp.eta()) < muEtaCut_) {
          loosemu += 1;
        }
      }
      if (fabs(gp.pdgId()) == 11) {
        if (gp.pt() > elPtCut_ && fabs(gp.eta()) < elEtaCut_) {
          return true;
        }
        if (gp.pt() > looseelPtCut_ && fabs(gp.eta()) < elEtaCut_) {
          looseel += 1;
        }
      }
      if (fabs(gp.pdgId()) == 22) {
        if (gp.pt() > gammaPtCut_ && fabs(gp.eta()) < gammaEtaCut_) {
          return true;
        }
        if (gp.pt() > loosegammaPtCut_ && fabs(gp.eta()) < gammaEtaCut_) {
          loosegamma += 1;
        } else {
          if (gp.pt() > veryloosegammaPtCut_ && fabs(gp.eta()) < gammaEtaCut_) {
            veryloosegamma += 1;
          }
        }
      }
      if (fabs(gp.pdgId()) == 1000024) {
        if (gp.numberOfDaughters() > 0) {
          decaylength = sqrt(pow(gp.vx() - gp.daughter(0)->vx(), 2) + pow(gp.vy() - gp.daughter(0)->vy(), 2));
          if (decaylength > 300) {
            return true;
          }
        } else {
          return true;
        }
      }
    }
  }
  if (looseel + loosemu + loosegamma > 1) {
    return true;
  }
  if (loosegamma > 0 && veryloosegamma > 0) {
    return true;
  }
  double genHT = 0.0;
  for (std::vector<reco::GenJet>::const_iterator it = generatedJets->begin(); it != generatedJets->end(); ++it) {
    const reco::GenJet& gjet = *it;
    //Add GenJet pt to genHT if GenJet complies with given HT definition
    if (gjet.pt() > jetPtCut_ && fabs(gjet.eta()) < jetEtaCut_) {
      genHT += gjet.pt();
    }
  }
  return (genHT > genHTcut_);
}

// Define module as a plug-in
DEFINE_FWK_MODULE(ModelpMSSMFilter);
