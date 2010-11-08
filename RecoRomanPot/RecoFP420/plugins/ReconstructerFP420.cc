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
    //A
    //   edm::Handle<ClusterCollectionFP420> icf_simhit;
    /*
    Handle<ClusterCollectionFP420> cf_simhit;
    std::vector<const ClusterCollectionFP420 *> cf_simhitvec;
    for(uint32_t i = 0; i< trackerContainers.size();i++){
      iEvent.getByLabel( trackerContainers[i], cf_simhit);
      cf_simhitvec.push_back(cf_simhit.product());   }
    std::auto_ptr<ClusterCollectionFP420 > input(new DigiCollectionFP420(cf_simhitvec));
    */   
    
    //B
    
      Handle<TrackCollectionFP420> input;
      try{
      iEvent.getByLabel( trackerContainers[0] , input);
      } catch(...){;}


       
    
    // Step C: create empty output collection
    std::auto_ptr<RecoCollectionFP420> toutput(new RecoCollectionFP420);
    
    
    
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
    sFP420RecoMain_->run(input, toutput);
    // cout <<"=======           ReconstructerFP420:                    end of produce     " << endl;
    
	// Step D: write output to file
    if (verbosity > 0) {
      std::cout << "ReconstructerFP420: iEvent.put(toutput)" << std::endl;
    }
	iEvent.put(toutput);
    if (verbosity > 0) {
      std::cout << "ReconstructerFP420: iEvent.put(toutput) DONE" << std::endl;
    }
  }//produce
  
} // namespace cms


