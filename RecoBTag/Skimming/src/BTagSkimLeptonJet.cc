/** \class BTaSkimLeptonJet
 *
 *
 *
 * \author Francisco Yumiceva, FERMILAB
 *
 */

#include <cmath>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

//#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
//#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "RecoBTag/Skimming/interface/BTagSkimLeptonJet.h"

#include "TVector3.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

using namespace edm;
using namespace std;
using namespace reco;

BTagSkimLeptonJet::BTagSkimLeptonJet(const edm::ParameterSet& iConfig) : nEvents_(0), nAccepted_(0) {
  CaloJetInput_ = iConfig.getParameter<InputTag>("CaloJet");
  MinCaloJetPt_ = iConfig.getParameter<double>("MinimumCaloJetPt");
  MaxCaloJetEta_ = iConfig.getParameter<double>("MaximumCaloJetEta");

  LeptonType_ = iConfig.getParameter<std::string>("LeptonType");
  if (LeptonType_ != "electron" && LeptonType_ != "muon")
    edm::LogError("BTagSkimLeptonJet") << "unknown lepton type !!";
  LeptonInput_ = iConfig.getParameter<InputTag>("Lepton");
  MinLeptonPt_ = iConfig.getParameter<double>("MinimumLeptonPt");
  MaxLeptonEta_ = iConfig.getParameter<double>("MaximumLeptonEta");
  //MinNLepton_ = iConfig.getParameter<int>( "MinimumNLepton" );

  MaxDeltaR_ = iConfig.getParameter<double>("MaximumDeltaR");

  MinPtRel_ = iConfig.getParameter<double>("MinimumPtRel");

  MinNLeptonJet_ = iConfig.getParameter<int>("MinimumNLeptonJet");
  if (MinNLeptonJet_ < 1)
    edm::LogError("BTagSkimLeptonJet") << "MinimunNCaloJet < 1 !!";
}

/*------------------------------------------------------------------------*/

BTagSkimLeptonJet::~BTagSkimLeptonJet() {}

/*------------------------------------------------------------------------*/

bool BTagSkimLeptonJet::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  nEvents_++;

  Handle<CaloJetCollection> CaloJetsHandle;
  //try {
  iEvent.getByLabel(CaloJetInput_, CaloJetsHandle);
  //}
  //catch ( cms::Exception& ex ) {
  //edm::LogError( "BTagSkimLeptonJet" )
  //  << "Unable to get CaloJet collection "
  //  << CaloJetInput_.label();
  //return false;
  //}
  if (CaloJetsHandle->empty())
    return false;
  CaloJetCollection TheCaloJets = *CaloJetsHandle;
  std::stable_sort(TheCaloJets.begin(), TheCaloJets.end(), PtSorter());

  //int nLepton = 0;

  Handle<MuonCollection> MuonHandle;
  MuonCollection TheMuons;
  if (LeptonType_ == "muon") {
    //try{
    iEvent.getByLabel(LeptonInput_, MuonHandle);
    //  }
    //	  catch ( cms::Exception& ex ) {
    //edm::LogError( "BTagSkimLeptonJet" )
    //	  << "Unable to get muon collection "
    //	  << LeptonInput_.label();
    //return false;
    //}
    TheMuons = *MuonHandle;
    std::stable_sort(TheMuons.begin(), TheMuons.end(), PtSorter());
  }

  Handle<GsfElectronCollection> ElectronHandle;
  GsfElectronCollection TheElectrons;
  if (LeptonType_ == "electron") {
    //try{
    iEvent.getByLabel(LeptonInput_, ElectronHandle);
    //  }
    // catch ( cms::Exception& ex ) {
    // edm::LogError( "BTagSkimLeptonJet" )
    //		  << "Unable to get electron collection "
    //		  << LeptonInput_.label();
    // return false;
    // }
    TheElectrons = *ElectronHandle;
    std::stable_sort(TheElectrons.begin(), TheElectrons.end(), PtSorter());
  }

  // Jet cuts
  int nJet = 0;
  for (const auto& TheCaloJet : TheCaloJets) {
    if ((fabs(TheCaloJet.eta()) < MaxCaloJetEta_) && (TheCaloJet.pt() > MinCaloJetPt_)) {
      if (LeptonType_ == "muon") {
        for (const auto& TheMuon : TheMuons) {
          // select good muon
          if ((TheMuon.pt() > MinLeptonPt_) && (fabs(TheMuon.eta()) < MaxLeptonEta_)) {
            double deltar = ROOT::Math::VectorUtil::DeltaR(TheCaloJet.p4().Vect(), TheMuon.momentum());

            TVector3 jetvec(TheCaloJet.p4().Vect().X(), TheCaloJet.p4().Vect().Y(), TheCaloJet.p4().Vect().Z());
            TVector3 muvec(TheMuon.momentum().X(), TheMuon.momentum().Y(), TheMuon.momentum().Z());
            jetvec += muvec;
            double ptrel = muvec.Perp(jetvec);

            if ((deltar < MaxDeltaR_) && (ptrel > MinPtRel_))
              nJet++;
            if (nJet >= MinNLeptonJet_)
              break;
          }
        }
      }

      if (LeptonType_ == "electron") {
        for (const auto& TheElectron : TheElectrons) {
          if ((TheElectron.pt() > MinLeptonPt_) && (fabs(TheElectron.eta()) < MaxLeptonEta_)) {
            double deltar = ROOT::Math::VectorUtil::DeltaR(TheCaloJet.p4().Vect(), TheElectron.momentum());

            TVector3 jetvec(TheCaloJet.p4().Vect().X(), TheCaloJet.p4().Vect().Y(), TheCaloJet.p4().Vect().Z());
            TVector3 evec(TheElectron.momentum().X(), TheElectron.momentum().Y(), TheElectron.momentum().Z());
            jetvec += evec;
            double ptrel = evec.Perp(jetvec);

            if ((deltar < MaxDeltaR_) && (ptrel > MinPtRel_))
              nJet++;
            if (nJet >= MinNLeptonJet_)
              break;
          }
        }
      }

    }  // close jet selection
  }

  if (nJet < MinNLeptonJet_)
    return false;

  nAccepted_++;

  return true;
}

/*------------------------------------------------------------------------*/

void BTagSkimLeptonJet::endJob() {
  edm::LogVerbatim("BTagSkimLeptonJet")
      << "=============================================================================\n"
      << " Events read: " << nEvents_ << "\n Events accepted by (" << LeptonType_
      << ") BTagSkimLeptonJet: " << nAccepted_ << "\n Efficiency: " << (double)(nAccepted_) / (double)(nEvents_)
      << "\n===========================================================================" << endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BTagSkimLeptonJet);
