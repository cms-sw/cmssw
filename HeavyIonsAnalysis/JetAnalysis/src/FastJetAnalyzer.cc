// -*- C++ -*-
//
// Package:    FastJetAnalyzer
// Class:      FastJetAnalyzer
//
/**\class FastJetAnalyzer FastJetAnalyzer.cc CmsHi/FastJetAnalyzer/src/FastJetAnalyzer.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz,32 4-A08,+41227673039,
//         Created:  Wed Feb 13 14:57:10 CET 2013
// $Id: FastJetAnalyzer.cc,v 1.2 2013/02/13 18:53:32 yilmaz Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TTree.h"

//
// class declaration
//

using namespace std;


struct MyBkg{
  int n;
  float rho[50];
  float sigma[50];
};


class FastJetAnalyzer : public edm::EDAnalyzer {
public:
  explicit FastJetAnalyzer(const edm::ParameterSet&);
  ~FastJetAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  // ----------member data ---------------------------


  vector<string> labels_;
  vector<MyBkg> bkgs_;
  vector<TTree*> trees_;

  edm::Service<TFileService> fs;

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
FastJetAnalyzer::FastJetAnalyzer(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed

  labels_ = iConfig.getParameter<vector<string> >("algos");
  bkgs_.reserve(labels_.size());

}


FastJetAnalyzer::~FastJetAnalyzer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
FastJetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  for(unsigned int ialgo = 0; ialgo < labels_.size(); ++ialgo){

    edm::Handle<vector<double> > rhos;
    edm::Handle<vector<double> > sigmas;
    iEvent.getByLabel(edm::InputTag(labels_[ialgo].data(),"rhos"),rhos);
    iEvent.getByLabel(edm::InputTag(labels_[ialgo].data(),"sigmas"),sigmas);

    bkgs_[ialgo].n = rhos->size();
    for(unsigned int i = 0; i < rhos->size(); ++i){
      bkgs_[ialgo].rho[i] = (*rhos)[i];
      bkgs_[ialgo].sigma[i] = (*sigmas)[i];
    }
  }


  for(unsigned int ialgo = 0; ialgo < labels_.size(); ++ialgo){
    trees_[ialgo]->Fill();
  }



}


// ------------ method called once each job just before starting event loop  ------------
void
FastJetAnalyzer::beginJob()
{

  for(unsigned int ialgo = 0; ialgo < labels_.size(); ++ialgo){
    MyBkg b;
    bkgs_.push_back(b);
    trees_.push_back(fs->make<TTree>(Form("%s",labels_[ialgo].data()),""));
    trees_[ialgo]->Branch("n",&bkgs_[ialgo].n,"n/I");
    trees_[ialgo]->Branch("rho",bkgs_[ialgo].rho,"rho[n]/F");
    trees_[ialgo]->Branch("sigma",bkgs_[ialgo].sigma,"sigma[n]/F");
  }

}

// ------------ method called once each job just after ending the event loop  ------------
void
FastJetAnalyzer::endJob()
{
}

// ------------ method called when starting to processes a run  ------------
void
FastJetAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void
FastJetAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void
FastJetAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
FastJetAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
FastJetAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(FastJetAnalyzer);
