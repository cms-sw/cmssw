///////////////////////////////////////////////////////////////////////////////
// File: ReconstructerFP420.cc
// Date: 11.2007
// Description: ReconstructerFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include <memory>
#include <string>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoRomanPot/RecoFP420/interface/ReconstructerFP420.h"
#include "DataFormats/FP420Cluster/interface/TrackCollectionFP420.h"
#include "DataFormats/FP420Cluster/interface/RecoCollectionFP420.h"

//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
//#include "HepMC/GenEvent.h"

#include <iostream> 
using namespace std;

//
namespace cms
{
  ReconstructerFP420::ReconstructerFP420(const edm::ParameterSet& conf):conf_(conf)     {
    
    edm::LogInfo ("ReconstructerFP420 ") <<"Enter the FP420 Reco constructer";


    verbosity = conf_.getUntrackedParameter<int>("VerbosityLevel");
    if (verbosity > 0) {
      std::cout << "Constructor of  ReconstructerFP420" << std::endl;
    }


    std::string alias ( conf.getParameter<std::string>("@module_label") );
    
    produces<RecoCollectionFP420>().setBranchAlias( alias );
    
    trackerContainers.clear();
    trackerContainers = conf.getParameter<std::vector<std::string> >("ROUList");
    VtxFlag                = conf.getParameter<int>("VtxFlagGenRec");
    m_genReadoutName        = conf.getParameter<string>("genReadoutName");
    
    
    // Initialization:
    sFP420RecoMain_ = new FP420RecoMain(conf_);
    
  }
  
  // Virtual destructor needed.
  ReconstructerFP420::~ReconstructerFP420() {
    if (verbosity > 0) {
      std::cout << "ReconstructerFP420:delete FP420RecoMain" << std::endl;
    }
    delete sFP420RecoMain_;
  }  
  
  //Get at the beginning
  void ReconstructerFP420::beginJob() {
    if (verbosity > 0) {
      std::cout << "ReconstructerFP420:BeginJob method " << std::endl;
    }
  }
  
  
  void ReconstructerFP420::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
  {
    //  beginJob;
    // be lazy and include the appropriate namespaces
    using namespace edm; 
    using namespace std;   
    
    // Get input

    // Vtx info:
    
    // define GEN Vtx of Signal
    double vtxGenX = 0.;
    double vtxGenY = 0.;
    double vtxGenZ = 0.;

    /*
    if(VtxFlag == 1) {
      
      Handle<HepMCProduct> EvtHandle;
      try{
	iEvent.getByLabel(m_genReadoutName,EvtHandle);
      }catch(const Exception&){
	if(verbosity>0){
	  std::cout << "no HepMCProduct found"<< std::endl;
	}
      }
      
      const HepMC::GenEvent* evt = EvtHandle->GetEvent() ;
      HepMC::GenParticle* proton1 = 0;
      HepMC::GenParticle* proton2 = 0;	
      double partmomcut=4000.;
      double pz1max = 0.;
      double pz2min = 0.;
      for ( HepMC::GenEvent::particle_const_iterator p = evt->particles_begin(); p != evt->particles_end(); ++p ) {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	double pz = (*p)->momentum().pz();
	//	if (((*p)->pdg_id() == ipdgproton)&&((*p)->status() == 1)&&(pz > partmomcut)){
	if( pz > partmomcut){
	  if(pz > pz1max){
	    proton1 = *p;pz1max=pz;
	  }
	} 
	//	else if(( (*p)->pdg_id() == ipdgproton)&&((*p)->status() == 1)&&(pz < -1.*partmomcut)) {
	else if(pz < -1.*partmomcut) {
	  if(pz < pz2min){
	    proton2 = *p;pz2min=pz;
	  }
	}
	
      }// for
      if(proton1 && !proton2){
	vtxGenX = (proton1)->production_vertex()->position().x();
	vtxGenY = (proton1)->production_vertex()->position().y();
	vtxGenZ = (proton1)->production_vertex()->position().z();
      }
      else if(proton2 && !proton1){
	vtxGenX = (proton2)->production_vertex()->position().x();
	vtxGenY = (proton2)->production_vertex()->position().y();
	vtxGenZ = (proton2)->production_vertex()->position().z();
      }
      else if(proton1 && proton2){
	if(abs((proton1)->momentum().pz()) >= abs((proton2)->momentum().pz()) ) {
	  vtxGenX = (proton1)->production_vertex()->position().x();
	  vtxGenY = (proton1)->production_vertex()->position().y();
	  vtxGenZ = (proton1)->production_vertex()->position().z();
	}
	else {
	  vtxGenX = (proton2)->production_vertex()->position().x();
	  vtxGenY = (proton2)->production_vertex()->position().y();
	  vtxGenZ = (proton2)->production_vertex()->position().z();
	}
      }
    }// if(VtxFlag == 1 

*/
    
    double VtxX = 0.;
    double VtxY = 0.;
    double VtxZ = 0.;
    if(VtxFlag == 1) {
      VtxX = vtxGenX;// mm
      VtxY = vtxGenY;// mm
      VtxZ = vtxGenZ;// mm
    }
    
    


    // track collection:
    //A
    //   edm::Handle<ClusterCollectionFP420> icf_simhit;
    /*
    Handle<ClusterCollectionFP420> cf_simhit;
    std::vector<const ClusterCollectionFP420 *> cf_simhitvec;
    for(uint32_t i = 0; i< trackerContainers.size();i++){
      iEvent.getByLabel( trackerContainers[i], cf_simhit);
      cf_simhitvec.push_back(cf_simhit.product());   }
    std::unique_ptr<ClusterCollectionFP420 > input(new DigiCollectionFP420(cf_simhitvec));
    */   
    
    //B
    
      Handle<TrackCollectionFP420> input;
      iEvent.getByLabel( trackerContainers[0] , input);


       
    
    // Step C: create empty output collection
    auto toutput = std::make_unique<RecoCollectionFP420>();
    
    
    
    //    put zero to container info from the beginning (important! because not any detID is updated with coming of new event     !!!!!!   
    // clean info of container from previous event
    
    std::vector<RecoFP420> collector;
    collector.clear();
    RecoCollectionFP420::Range inputRange;
    inputRange.first = collector.begin();
    inputRange.second = collector.end();
    
    unsigned int detID = 0;
    toutput->putclear(inputRange,detID);
    
    unsigned  int StID = 1;
    toutput->putclear(inputRange,StID);
    StID = 2;
    toutput->putclear(inputRange,StID);
    
    
    //                                                                                                                      !!!!!!   
    // if we want to keep Reco container/Collection for one event --->   uncomment the line below and vice versa
    toutput->clear();   //container_.clear() --> start from the beginning of the container
    
    //                                RUN now:                                                                                 !!!!!!     
    //   startFP420RecoMain_.run(input, toutput);
    sFP420RecoMain_->run(input, toutput.get(), VtxX, VtxY, VtxZ);
    // std::cout <<"=======           ReconstructerFP420:                    end of produce     " << endl;
    
	// Step D: write output to file
    if (verbosity > 0) {
      std::cout << "ReconstructerFP420: iEvent.put(std::move(toutput)" << std::endl;
    }
	iEvent.put(std::move(toutput));
    if (verbosity > 0) {
      std::cout << "ReconstructerFP420: iEvent.put(std::move(toutput) DONE" << std::endl;
    }
  }//produce
  
} // namespace cms


