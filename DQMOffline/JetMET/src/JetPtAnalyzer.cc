/*
 *  See header file for a description of this class.
 *
 *  \author J.Weng ETH Zuerich
 */

#include "DQMOffline/JetMET/interface/JetPtAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace edm;

// ***********************************************************
JetPtAnalyzer::JetPtAnalyzer(const edm::ParameterSet& pSet) {

  parameters   = pSet;

  // Note: one could also add cut on energies from short and long fibers
  // or fraction of energy from the hottest HPD readout 

  //-------------------
}

// ***********************************************************
JetPtAnalyzer::~JetPtAnalyzer() { }


// ***********************************************************
void JetPtAnalyzer::beginJob(DQMStore * dbe) {

  jetname = "jetPtAnalyzer";

  LogTrace(jetname)<<"[JetPtAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("JetMET/Jet/"+_source);

  jetME = dbe->book1D("jetReco", "jetReco", 3, 1, 4);
  jetME->setBinLabel(1,"CaloJets",1);

  //initialize JetID
  jetID = new reco::helper::JetIDHelper(parameters.getParameter<ParameterSet>("JetIDParams"));

  // monitoring of eta parameter
  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");

  // monitoring of phi paramater
  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");

  // monitoring of the transverse momentum
  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");

  mEta                     = dbe->book2D("EtaVsPt", "EtaVsPt",ptBin, ptMin, ptMax, etaBin, etaMin, etaMax);
  mPhi                     = dbe->book2D("PhiVsPt", "PhiVsPt",ptBin, ptMin, ptMax, phiBin, phiMin, phiMax);
  mConstituents            = dbe->book2D("ConstituentsVsPt", "# of ConstituentsVsPt",ptBin, ptMin, ptMax, 100, 0, 100);
  //  mNJets                   = dbe->book2D("NJetsVsPt", "number of jetsVsPt",ptBin, ptMin, ptMax, 100, 0, 100);

  // CaloJet specific
  mHFrac                  = dbe->book2D("HFracVsPt", "HFracVsPt",ptBin, ptMin, ptMax, 120, -0.1, 1.1);
  mEFrac                  = dbe->book2D("EFracVsPt", "EFracVsPt", ptBin, ptMin, ptMax,120, -0.1, 1.1);
  mMaxEInEmTowers         = dbe->book2D("MaxEInEmTowersVsPt", "MaxEInEmTowersVsPt", ptBin, ptMin, ptMax,100, 0, 100);
  mMaxEInHadTowers        = dbe->book2D("MaxEInHadTowersVsPt", "MaxEInHadTowersVsPt",ptBin, ptMin, ptMax, 100, 0, 100);
  mHadEnergyInHO          = dbe->book2D("HadEnergyInHOVsPt", "HadEnergyInHOVsPt",ptBin, ptMin, ptMax, 100, 0, 10);
  mHadEnergyInHB          = dbe->book2D("HadEnergyInHBVsPt", "HadEnergyInHBVsPt",ptBin, ptMin, ptMax, 100, 0, 50);
  mHadEnergyInHF          = dbe->book2D("HadEnergyInHFVsPt", "HadEnergyInHFVsPt",ptBin, ptMin, ptMax ,100, 0, 50);
  mHadEnergyInHE          = dbe->book2D("HadEnergyInHEVsPt", "HadEnergyInHEVsPt",ptBin, ptMin, ptMax, 100, 0, 100);
  mEmEnergyInEB           = dbe->book2D("EmEnergyInEBVsPt", "EmEnergyInEBVsPt",ptBin, ptMin, ptMax, 100, 0, 50);
  mEmEnergyInEE           = dbe->book2D("EmEnergyInEEVsPt", "EmEnergyInEEVsPt",ptBin, ptMin, ptMax, 100, 0, 50);
  mEmEnergyInHF           = dbe->book2D("EmEnergyInHFVsPt", "EmEnergyInHFVsPt",ptBin, ptMin, ptMax, 120, -20, 100); 
  //jetID variables
  mN90Hits                    = dbe->book2D("N90HitsVsPt", "N90HitsVsPt",ptBin, ptMin, ptMax, 50, 0, 50);
  mresEMF                    = dbe->book2D("resEMFVsPt", "resEMFVsPt",ptBin, ptMin, ptMax,50, 0., 1.);
  mfHPD                   = dbe->book2D("fHPDVsPt", "fHPDVsPt",ptBin, ptMin, ptMax,50, 0., 1.);
  mfRBX                    = dbe->book2D("fRBXVsPt", "fRBXVsPt",ptBin, ptMin, ptMax,50, 0., 1.);


}
void JetPtAnalyzer::endJob() {

  delete jetID;

}


// ***********************************************************
void JetPtAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			  const reco::CaloJetCollection& caloJets) {
  //  int numofjets=0;
  LogTrace(jetname)<<"[JetPtAnalyzer] Analyze Calo Jet";

  for (reco::CaloJetCollection::const_iterator jet = caloJets.begin(); jet!=caloJets.end(); ++jet){
  //-----------------------------
  jetME->Fill(1);

  jetID->calculate(iEvent, *jet);
  
  if (mEta)  mEta->Fill (jet->pt(),jet->eta());
  if (mPhi)  mPhi->Fill (jet->pt(),jet->phi());

  //  if (mNJets)    mNJets->Fill (jet->pt(),numofjets);
  if (mConstituents) mConstituents->Fill (jet->pt(),jet->nConstituents());
  if (mHFrac)        mHFrac->Fill (jet->pt(),jet->energyFractionHadronic());
  if (mEFrac)        mEFrac->Fill (jet->pt(),jet->emEnergyFraction());

 
  if (mMaxEInEmTowers)  mMaxEInEmTowers->Fill (jet->pt(),jet->maxEInEmTowers());
  if (mMaxEInHadTowers) mMaxEInHadTowers->Fill (jet->pt(),jet->maxEInHadTowers());

  if (mHadEnergyInHO)   mHadEnergyInHO->Fill (jet->pt(),jet->hadEnergyInHO());
  if (mHadEnergyInHB)   mHadEnergyInHB->Fill (jet->pt(),jet->hadEnergyInHB());
  if (mHadEnergyInHF)   mHadEnergyInHF->Fill (jet->pt(),jet->hadEnergyInHF());
  if (mHadEnergyInHE)   mHadEnergyInHE->Fill (jet->pt(),jet->hadEnergyInHE());
  if (mEmEnergyInEB)    mEmEnergyInEB->Fill (jet->pt(),jet->emEnergyInEB());
  if (mEmEnergyInEE)    mEmEnergyInEE->Fill (jet->pt(),jet->emEnergyInEE());
  if (mEmEnergyInHF)    mEmEnergyInHF->Fill (jet->pt(),jet->emEnergyInHF());

  if (mN90Hits)         mN90Hits->Fill (jet->pt(),jetID->n90Hits());  
  if (mresEMF)          mresEMF->Fill(jet->pt(),jetID->restrictedEMF());
  if (mfHPD)            mfHPD->Fill(jet->pt(),jetID->fHPD());
  if (mfRBX)            mfRBX->Fill(jet->pt(),jetID->fRBX());

  }
}
