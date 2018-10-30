///////////////////////////////////////////////////////////////////////////////
// File: TrackerizerFP420.cc
// Date: 12.2006
// Description: TrackerizerFP420 for FP420
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

#include "RecoRomanPot/RecoFP420/interface/TrackerizerFP420.h"
#include "DataFormats/FP420Cluster/interface/ClusterCollectionFP420.h"
#include "DataFormats/FP420Cluster/interface/TrackCollectionFP420.h"

#include <iostream> 
using namespace std;

//
namespace cms
{
  TrackerizerFP420::TrackerizerFP420(const edm::ParameterSet& conf):conf_(conf)     {
    
    std::string alias ( conf.getParameter<std::string>("@module_label") );
    
    produces<TrackCollectionFP420>().setBranchAlias( alias );
    
    trackerContainers.clear();
    trackerContainers = conf.getParameter<std::vector<std::string> >("ROUList");
    
    verbosity = conf_.getUntrackedParameter<int>("VerbosityLevel");
    if (verbosity > 0) {
      std::cout << "Creating a TrackerizerFP420" << std::endl;
    }
    
    // Initialization:
    sFP420TrackMain_ = new FP420TrackMain(conf_);
    
  }
  
  // Virtual destructor needed.
  TrackerizerFP420::~TrackerizerFP420() {
    delete sFP420TrackMain_;
  }  
  
  //Get at the beginning
  void TrackerizerFP420::beginJob() {
    if (verbosity > 0) {
      std::cout << "BeginJob method " << std::endl;
    }
  }
  
  
  void TrackerizerFP420::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
    std::unique_ptr<ClusterCollectionFP420 > input(new DigiCollectionFP420(cf_simhitvec));
    */   
    
    //B
    
      Handle<ClusterCollectionFP420> input;
      iEvent.getByLabel( trackerContainers[0] , input);


       
    
    // Step C: create empty output collection
    auto toutput = std::make_unique<TrackCollectionFP420>();
    
    
    
    //    put zero to container info from the beginning (important! because not any detID is updated with coming of new event     !!!!!!   
    // clean info of container from previous event
    
    std::vector<TrackFP420> collector;
    collector.clear();
    TrackCollectionFP420::Range inputRange;
    inputRange.first = collector.begin();
    inputRange.second = collector.end();
    
    unsigned int detID = 0;
    toutput->putclear(inputRange,detID);
    
    unsigned  int StID = 1111;
    toutput->putclear(inputRange,StID);
    StID = 2222;
    toutput->putclear(inputRange,StID);
    
    
    //                                                                                                                      !!!!!!   
    // if we want to keep Track container/Collection for one event --->   uncomment the line below and vice versa
    toutput->clear();   //container_.clear() --> start from the beginning of the container
    
    //                                RUN now:                                                                                 !!!!!!     
    //   startFP420TrackMain_.run(input, toutput);
    sFP420TrackMain_->run(input, toutput.get());
    // std::cout <<"=======           TrackerizerFP420:                    end of produce     " << std::endl;
    
	// Step D: write output to file
	iEvent.put(std::move(toutput));
  }//produce
  
} // namespace cms


