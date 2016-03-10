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

#include "TLorentzVector.h"
#include "TMath.h"
#include "TString.h"
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
  using namespace edm;

  //clear vectors
  rhoObj_.etaMin.clear();
  rhoObj_.etaMax.clear();
  rhoObj_.rho.clear();
  rhoObj_.rhom.clear();
  
  // Get the vector of background densities
  edm::Handle<std::vector<double>> etaRanges;
  edm::Handle<std::vector<double>> rho;
  edm::Handle<std::vector<double>> rhom;
  iEvent.getByToken(etaToken_, etaRanges);
  iEvent.getByToken(rhoToken_, rho);
  iEvent.getByToken(rhomToken_, rhom);

  int neta = (int)etaRanges->size();
  for(int ieta = 0; ieta<(neta-1); ieta++) {
    rhoObj_.etaMin.push_back(etaRanges->at(ieta));
    rhoObj_.etaMax.push_back(etaRanges->at(ieta+1));
    rhoObj_.rho.push_back(rho->at(ieta));
    rhoObj_.rhom.push_back(rhom->at(ieta));
  }
  
  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void 
HiFJRhoAnalyzer::beginJob()
{

  TString jetTagTitle = "HiFJRho Jet background analysis tree";
  tree_ = fs_->make<TTree>("t",jetTagTitle.Data());
  
  tree_->Branch("etaMin",&(rhoObj_.etaMin));
  tree_->Branch("etaMax",&(rhoObj_.etaMax));
  tree_->Branch("rho",&(rhoObj_.rho));
  tree_->Branch("rhom",&(rhoObj_.rhom));
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HiFJRhoAnalyzer::endJob() {
}

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
