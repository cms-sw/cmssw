// -*- C++ -*-
//
// Package:    CondTools/BeamSpot
// Class:      BeamSpotRcdReader
// 
/**\class BeamSpotRcdReader BeamSpotRcdReader.cc CondTools/BeamSpot/plugins/BeamSpotRcdReader.cc

 Description: simple emd::one::EDAnalyzer to retrieve and ntuplize BeamSpot data from the conditions database

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Tue, 18 Oct 2016 11:00:44 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

// For ROOT
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TTree.h>

#include <sstream>
#include <fstream>

//
// class declaration
//

class BeamSpotRcdReader : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit BeamSpotRcdReader(const edm::ParameterSet&);
      ~BeamSpotRcdReader() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      void beginJob() override;
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob() override;

      struct theBSfromDB
      {
	int   run;
	int   ls;
	float BSx0_;
	float BSy0_;
	float BSz0_;
	float Beamsigmaz_;
	float Beamdxdz_;   
	float BeamWidthX_;
	float BeamWidthY_;
	void init();
      } theBSfromDB_;

      edm::Service<TFileService> tFileService; 
      TTree * bstree_;

      // ----------member data ---------------------------
      edm::ESWatcher<BeamSpotObjectsRcd> watcher_;
      std::unique_ptr<std::ofstream> output_;
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
BeamSpotRcdReader::BeamSpotRcdReader(const edm::ParameterSet& iConfig) :
  bstree_(nullptr)
{
  //now do what ever initialization is needed
  usesResource("TFileService");
  std::string fileName(iConfig.getUntrackedParameter<std::string>("rawFileName"));
  if (!fileName.empty()) {
    output_.reset(new std::ofstream(fileName.c_str()));
    if (!output_->good()) {
      edm::LogError("IOproblem") << "Could not open output file " << fileName << ".";
      output_.reset();
    }
  }
}


BeamSpotRcdReader::~BeamSpotRcdReader()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

void 
BeamSpotRcdReader::theBSfromDB::init()
{

  float dummy_float = 9999.0;
  int   dummy_int   = 9999;

  run         = dummy_int;	  
  ls          = dummy_int;	  
  BSx0_       = dummy_float;	  
  BSy0_       = dummy_float;	  
  BSz0_       = dummy_float;	  
  Beamsigmaz_ = dummy_float;
  Beamdxdz_   = dummy_float;
  BeamWidthX_ = dummy_float;
  BeamWidthY_ = dummy_float;

}


// ------------ method called for each event  ------------
void
BeamSpotRcdReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   std::ostringstream output;   

   // initialize the ntuple
   theBSfromDB_.init();

   if (watcher_.check(iSetup)) { // check for new IOV for this run / LS
     
     output << " for runs: " << iEvent.id().run() << " - " << iEvent.id().luminosityBlock() << std::endl;
     
     // Get BeamSpot from EventSetup:
     edm::ESHandle< BeamSpotObjects > beamhandle;
     iSetup.get<BeamSpotObjectsRcd>().get(beamhandle);
     const BeamSpotObjects *mybeamspot = beamhandle.product();
      
     theBSfromDB_.run         = iEvent.id().run();
     theBSfromDB_.ls          = iEvent.id().luminosityBlock();
     theBSfromDB_.BSx0_       = mybeamspot->GetX();
     theBSfromDB_.BSy0_       = mybeamspot->GetY();
     theBSfromDB_.BSz0_       = mybeamspot->GetZ();
     theBSfromDB_.Beamsigmaz_ = mybeamspot->GetSigmaZ(); 
     theBSfromDB_.Beamdxdz_   = mybeamspot->Getdxdz(); 
     theBSfromDB_.BeamWidthX_ = mybeamspot->GetBeamWidthX(); 
     theBSfromDB_.BeamWidthY_ = mybeamspot->GetBeamWidthY();

     bstree_->Fill();

     output <<  *mybeamspot << std::endl;

   }

   // Final output - either message logger or output file:
   if (output_.get()) *output_ << output.str();
   else edm::LogInfo("") << output.str();
}


// ------------ method called once each job just before starting event loop  ------------
void 
BeamSpotRcdReader::beginJob()
{
  bstree_        = tFileService->make<TTree>("BSNtuple","BeamSpot analyzer ntuple");
  
  //Tree Branches
  bstree_->Branch("run",&theBSfromDB_.run,"run/I");
  bstree_->Branch("ls",&theBSfromDB_.ls,"ls/I");
  bstree_->Branch("BSx0",&theBSfromDB_.BSx0_,"BSx0/F");
  bstree_->Branch("BSy0",&theBSfromDB_.BSy0_,"BSy0/F");	   
  bstree_->Branch("BSz0",&theBSfromDB_.BSz0_,"BSz0/F");	   
  bstree_->Branch("Beamsigmaz",&theBSfromDB_.Beamsigmaz_ ,"Beamsigmaz/F");	   
  bstree_->Branch("Beamdxdz",&theBSfromDB_.Beamdxdz_,"Beamdxdz/F");	   
  bstree_->Branch("BeamWidthX",&theBSfromDB_.BeamWidthX_,"BeamWidthX/F");	   
  bstree_->Branch("BeamWidthY",&theBSfromDB_.BeamWidthY_,"BeamWidthY/F");	   

}

// ------------ method called once each job just after ending the event loop  ------------
void 
BeamSpotRcdReader::endJob() 
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
BeamSpotRcdReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotRcdReader);
