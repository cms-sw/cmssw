#ifndef HiCaloJetAnalyzer_caloJetAnalyzer_
#define HiCaloJetAnalyzer_caloJetAnalyzer_

// system include files
#include <memory>
#include <string>
#include <iostream>

// ROOT headers
#include "TTree.h"

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "fastjet/contrib/Njettiness.hh"
//

/**\class HiCaloJetAnalyzer

   \author Jussi Viinikainen
   \date   September 2023
*/

class HiCaloJetAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HiCaloJetAnalyzer(const edm::ParameterSet&);

  ~HiCaloJetAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
  void endRun(const edm::Run& run, const edm::EventSetup& es) override;

  void beginJob() override;

private:

  edm::InputTag jetTagLabel_;
  edm::EDGetTokenT<reco::CaloJetCollection> jetTag_;
  edm::EDGetTokenT<edm::View<pat::PackedCandidate>> pfCandidateLabel_;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticleSrc_;
  edm::EDGetTokenT<edm::View<reco::GenJet>> genjetTag_;
  edm::EDGetTokenT<edm::HepMCProduct> eventInfoTag_;
  edm::EDGetTokenT<GenEventInfoProduct> eventGenInfoTag_;

  std::string jetName_;  //used as prefix for jet structures

  bool isMC_;
  bool useHepMC_;
  bool fillGenJets_;
  bool useQuality_;
  std::string trackQuality_;

  double genPtMin_;

  bool doHiJetID_;
  bool doCaloEnergyFractions_;

  double r2Param;
  double hardPtMin_;
  double jetPtMin_;
  double jetAbsEtaMax_;

  TTree* caloJetTree_;
  edm::Service<TFileService> fs1;

  static const int MAXJETS = 1000;
  static const int MAXTRACKS = 5000;

  struct JRA {
    int nref = 0;
    int run = 0;
    int evt = 0;
    int lumi = 0;

    // Basic jet kinematic variables
    float rawpt[MAXJETS] = {0};
    float jtpt[MAXJETS] = {0};
    float jteta[MAXJETS] = {0};
    float jtphi[MAXJETS] = {0};

    // Advanced jet variables
    float jty[MAXJETS] = {0};
    float jtpu[MAXJETS] = {0};
    float jtm[MAXJETS] = {0};
    float jtarea[MAXJETS] = {0};

    // Arrays for HiJetID
    float trackMax[MAXJETS] = {0};
    float trackSum[MAXJETS] = {0};
    int trackN[MAXJETS] = {0};

    float chargedMax[MAXJETS] = {0};
    float chargedSum[MAXJETS] = {0};
    int chargedN[MAXJETS] = {0};

    float photonMax[MAXJETS] = {0};
    float photonSum[MAXJETS] = {0};
    int photonN[MAXJETS] = {0};

    float trackHardSum[MAXJETS] = {0};
    float chargedHardSum[MAXJETS] = {0};
    float photonHardSum[MAXJETS] = {0};

    int trackHardN[MAXJETS] = {0};
    int chargedHardN[MAXJETS] = {0};
    int photonHardN[MAXJETS] = {0};

    float neutralMax[MAXJETS] = {0};
    float neutralSum[MAXJETS] = {0};
    int neutralN[MAXJETS] = {0};

    float eMax[MAXJETS] = {0};
    float eSum[MAXJETS] = {0};
    int eN[MAXJETS] = {0};

    float muMax[MAXJETS] = {0};
    float muSum[MAXJETS] = {0};
    int muN[MAXJETS] = {0};

    float genChargedSum[MAXJETS] = {0};
    float genHardSum[MAXJETS] = {0};
    float signalChargedSum[MAXJETS] = {0};
    float signalHardSum[MAXJETS] = {0};

    // Arrays for calorimeter energy fractions
    float emEnergyFraction[MAXJETS] = {0};
    float hadronicEnergyFraction[MAXJETS] = {0};

    // Variables for generator level jets
    float pthat = 0;
    int beamId1 = 0;
    int beamId2 = 0;
    int ngen = 0;
    float genpt[MAXJETS] = {0};
    float geneta[MAXJETS] = {0};
    float genphi[MAXJETS] = {0};
    float genm[MAXJETS] = {0};
    float geny[MAXJETS] = {0};

  };

  JRA jets_;
};

#endif
