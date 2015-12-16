// -*- C++ -*-
//
// Package:    HiForestInfo
// Class:      HiForestInfo
//
/**\class HiForestInfo HiForestInfo.cc CmsHi/HiForestInfo/src/HiForestInfo.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Fri May 25 06:50:40 EDT 2012
// $Id: HiForestInfo.cc,v 1.1 2012/05/25 11:14:32 yilmaz Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TH1.h"
#include "TTree.h"

//
// class declaration
//

class HiForestInfo : public edm::EDAnalyzer {
public:
  explicit HiForestInfo(const edm::ParameterSet&);
  ~HiForestInfo();

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

  edm::Service<TFileService> fs;

  std::vector<std::string> inputLines_;

  TTree* HiForestVersionTree;
  std::string HiForestVersion_;
  std::string GlobalTagLabel_;
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
HiForestInfo::HiForestInfo(const edm::ParameterSet& iConfig)
{
  inputLines_ = iConfig.getParameter<std::vector<std::string> >("inputLines");
  HiForestVersion_ = iConfig.getParameter<std::string>("HiForestVersion");
  GlobalTagLabel_ = iConfig.getParameter<std::string>("GlobalTagLabel");
}


HiForestInfo::~HiForestInfo()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
HiForestInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

}


// ------------ method called once each job just before starting event loop  ------------
void
HiForestInfo::beginJob()
{
  HiForestVersionTree = fs->make<TTree>("HiForestInfo","HiForestInfo");
  std::vector<char *>inputLines_c;
  inputLines_c.resize(inputLines_.size());
  for(unsigned i = 0; i < inputLines_.size(); ++i){
    char * cstr = new char [inputLines_[i].length()+1];
    std::strcpy (cstr, inputLines_[i].c_str());
    inputLines_c[i] = cstr;
    HiForestVersionTree->Branch(Form("InputLines_%i",i),inputLines_c[i],"InputLines/C");
  }

  char *HiForestVersion_c = new char[HiForestVersion_.length()+1];
  std::strcpy(HiForestVersion_c, HiForestVersion_.c_str());
  HiForestVersionTree->Branch("HiForestVersion",HiForestVersion_c,"HiForestVersion/C");

  char *GlobalTagLabel_c = new char[GlobalTagLabel_.length()+1];
  std::strcpy(GlobalTagLabel_c, GlobalTagLabel_.c_str());
  HiForestVersionTree->Branch("GlobalTag",GlobalTagLabel_c,"GlobalTag/C");

  HiForestVersionTree->Fill();
}

// ------------ method called once each job just after ending the event loop  ------------
void
HiForestInfo::endJob()
{
}

// ------------ method called when starting to processes a run  ------------
void
HiForestInfo::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void
HiForestInfo::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void
HiForestInfo::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
HiForestInfo::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HiForestInfo::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiForestInfo);
