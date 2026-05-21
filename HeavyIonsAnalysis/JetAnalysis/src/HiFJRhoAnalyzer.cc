// -*- C++ -*-
//
// Package:    HiJetBackground/HiFJRhoAnalyzer
// Class:      HiFJRhoAnalyzer
// 
/**\class HiFJRhoAnalyzer HiFJRhoAnalyzer.cc HiJetBackground/HiFJRhoAnalyzer/plugins/HiFJRhoAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marta Verweij
//         Created:  Thu, 16 Jul 2015 10:57:12 GMT
//
//

#include <memory>
#include <string>

#include "TTree.h"

#include "HeavyIonsAnalysis/JetAnalysis/interface/HiFJRhoAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/Handle.h"

#include "CommonTools/Utils/interface/PtComparator.h"

using namespace edm;

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
HiFJRhoAnalyzer::HiFJRhoAnalyzer(const edm::ParameterSet& iConfig) 
{
  etaToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "etaMap" ));
  rhoToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "rho" ));
  rhomToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "rhom" ));
  rhoCorrToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "rhoCorr" ));
  rhomCorrToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "rhomCorr" ));
  rhoCorr1BinToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "rhoCorr1Bin" ));
  rhomCorr1BinToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "rhomCorr1Bin" ));
  //rhoGridToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "rhoGrid" ));
  //meanRhoGridToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "meanRhoGrid" ));
  //etaMinRhoGridToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "etaMinRhoGrid" ));
  //etaMaxRhoGridToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "etaMaxRhoGrid" ));
  ptJetsToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "ptJets" ));
  areaJetsToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "areaJets" ));
  etaJetsToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "etaJets" ));
  useModulatedRho_ = iConfig.getParameter<bool>("useModulatedRho");
  if (useModulatedRho_) {
    rhoFlowFitParamsToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "rhoFlowFitParams" ));
    nTowToken_ = consumes<std::vector<int>>(iConfig.getParameter<edm::InputTag>( "nTow" ));
    towExcludePtToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "towExcludePt" ));
    towExcludePhiToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "towExcludePhi" ));
    towExcludeEtaToken_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>( "towExcludeEta" ));
  }
}


HiFJRhoAnalyzer::~HiFJRhoAnalyzer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to analyze the data  ------------
void HiFJRhoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //clear vectors
  rhoObj_.etaMin.clear();
  rhoObj_.etaMax.clear();
  rhoObj_.rho.clear();
  rhoObj_.rhom.clear();
  rhoObj_.rhoCorr.clear();
  rhoObj_.rhomCorr.clear();
  rhoObj_.rhoCorr1Bin.clear();
  rhoObj_.rhomCorr1Bin.clear();
  
  rhoObj_.rhoGrid.clear();
  rhoObj_.meanRhoGrid.clear();
  rhoObj_.etaMinRhoGrid.clear();
  rhoObj_.etaMaxRhoGrid.clear();
  
  rhoObj_.ptJets.clear();
  rhoObj_.areaJets.clear();
  rhoObj_.etaJets.clear();

  rhoObj_.rhoFlowFitParams.clear();
  rhoObj_.nTow.clear();
  rhoObj_.towExcludePt.clear();
  rhoObj_.towExcludePhi.clear();
  rhoObj_.towExcludeEta.clear();
  
  // Get the vector of background densities
  edm::Handle<std::vector<double>> etaRanges;
  edm::Handle<std::vector<double>> rho;
  edm::Handle<std::vector<double>> rhom;
  edm::Handle<std::vector<double>> rhoCorr;
  edm::Handle<std::vector<double>> rhomCorr;
  edm::Handle<std::vector<double>> rhoCorr1Bin;
  edm::Handle<std::vector<double>> rhomCorr1Bin;
  
  // edm::Handle<std::vector<double>> rhoGrid;
  // edm::Handle<std::vector<double>> meanRhoGrid;
  // edm::Handle<std::vector<double>> etaMinRhoGrid;
  // edm::Handle<std::vector<double>> etaMaxRhoGrid;
  
  edm::Handle<std::vector<double>> ptJets;
  edm::Handle<std::vector<double>> areaJets;
  edm::Handle<std::vector<double>> etaJets;

  edm::Handle<std::vector<double>> rhoFlowFitParams;
  edm::Handle<std::vector<int>> nTow;
  edm::Handle<std::vector<double>> towExcludePt;
  edm::Handle<std::vector<double>> towExcludePhi;
  edm::Handle<std::vector<double>> towExcludeEta;
  
  iEvent.getByToken(etaToken_, etaRanges);
  iEvent.getByToken(rhoToken_, rho);
  iEvent.getByToken(rhomToken_, rhom);
  iEvent.getByToken(rhoCorrToken_, rhoCorr);
  iEvent.getByToken(rhomCorrToken_, rhomCorr);
  iEvent.getByToken(rhoCorr1BinToken_, rhoCorr1Bin);
  iEvent.getByToken(rhomCorr1BinToken_, rhomCorr1Bin);
  // iEvent.getByToken(rhoGridToken_, rhoGrid);
  // iEvent.getByToken(meanRhoGridToken_, meanRhoGrid);
  // iEvent.getByToken(etaMinRhoGridToken_, etaMinRhoGrid);
  // iEvent.getByToken(etaMaxRhoGridToken_, etaMaxRhoGrid);
  
  iEvent.getByToken(ptJetsToken_, ptJets);
  iEvent.getByToken(areaJetsToken_, areaJets);
  iEvent.getByToken(etaJetsToken_, etaJets);

  if (useModulatedRho_) {
    iEvent.getByToken(rhoFlowFitParamsToken_, rhoFlowFitParams);
    iEvent.getByToken(nTowToken_, nTow);
    iEvent.getByToken(towExcludePtToken_, towExcludePt);
    iEvent.getByToken(towExcludePhiToken_, towExcludePhi);
    iEvent.getByToken(towExcludeEtaToken_, towExcludeEta);
  }
  
  int neta = (int)etaRanges->size();
  for(int ieta = 0; ieta<(neta-1); ieta++) {
    rhoObj_.etaMin.push_back(etaRanges->at(ieta));
    rhoObj_.etaMax.push_back(etaRanges->at(ieta+1));
    rhoObj_.rho.push_back(rho->at(ieta));
    rhoObj_.rhom.push_back(rhom->at(ieta));
    rhoObj_.rhoCorr.push_back(rhoCorr->at(ieta));
    rhoObj_.rhomCorr.push_back(rhomCorr->at(ieta));
    rhoObj_.rhoCorr1Bin.push_back(rhoCorr1Bin->at(ieta));
    rhoObj_.rhomCorr1Bin.push_back(rhomCorr1Bin->at(ieta));
  }

  int njets = (int)ptJets->size();
  for(int ijet = 0; ijet<njets; ijet++) {
    rhoObj_.ptJets.push_back(ptJets->at(ijet));
    rhoObj_.areaJets.push_back(areaJets->at(ijet));
    rhoObj_.etaJets.push_back(etaJets->at(ijet));
  }

  if (useModulatedRho_) {
    rhoObj_.rhoFlowFitParams = *rhoFlowFitParams;
    rhoObj_.nTow = *nTow;
    rhoObj_.towExcludePt = *towExcludePt;
    rhoObj_.towExcludePhi = *towExcludePhi;
    rhoObj_.towExcludeEta = *towExcludeEta;
  }
  
  // int netaGrid = (int)rhoGrid->size();
  // for(int igrid = 0; igrid<netaGrid; igrid++) {
  //   rhoObj_.rhoGrid.push_back(rhoGrid->at(igrid));
  //   rhoObj_.meanRhoGrid.push_back(meanRhoGrid->at(igrid));
  //   rhoObj_.etaMinRhoGrid.push_back(etaMinRhoGrid->at(igrid));
  //   rhoObj_.etaMaxRhoGrid.push_back(etaMaxRhoGrid->at(igrid));
  // }
  
  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void HiFJRhoAnalyzer::beginJob() {

  tree_ = fs_->make<TTree>("t", "HiFJRho Jet background analysis tree");
  
  tree_->Branch("etaMin",&(rhoObj_.etaMin));
  tree_->Branch("etaMax",&(rhoObj_.etaMax));
  tree_->Branch("rho",&(rhoObj_.rho));
  tree_->Branch("rhom",&(rhoObj_.rhom));
  tree_->Branch("rhoCorr",&(rhoObj_.rhoCorr));
  tree_->Branch("rhomCorr",&(rhoObj_.rhomCorr));
  tree_->Branch("rhoCorr1Bin",&(rhoObj_.rhoCorr1Bin));
  tree_->Branch("rhomCorr1Bin",&(rhoObj_.rhomCorr1Bin));
  tree_->Branch("rhoGrid",&(rhoObj_.rhoGrid));
  tree_->Branch("meanRhoGrid",&(rhoObj_.meanRhoGrid));
  tree_->Branch("etaMinRhoGrid",&(rhoObj_.etaMinRhoGrid));
  tree_->Branch("etaMaxRhoGrid",&(rhoObj_.etaMaxRhoGrid));
  tree_->Branch("ptJets",&(rhoObj_.ptJets));
  tree_->Branch("etaJets",&(rhoObj_.etaJets));
  tree_->Branch("areaJets",&(rhoObj_.areaJets));
  if (useModulatedRho_) {
    tree_->Branch("rhoFlowFitParams",&(rhoObj_.rhoFlowFitParams));
    tree_->Branch("nTow",&(rhoObj_.nTow));
    tree_->Branch("towExcludePt",&(rhoObj_.towExcludePt));
    tree_->Branch("towExcludePhi",&(rhoObj_.towExcludePhi));
    tree_->Branch("towExcludeEta",&(rhoObj_.towExcludeEta));
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void HiFJRhoAnalyzer::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HiFJRhoAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  // edm::ParameterSetDescription desc;
  // desc.setUnknown();
  // descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiFJRhoAnalyzer);
