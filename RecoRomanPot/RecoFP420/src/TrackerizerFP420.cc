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
#include "RecoRomanPot/RecoFP420/interface/ClusterCollectionFP420.h"
#include "RecoRomanPot/RecoFP420/interface/TrackCollectionFP420.h"

#include <iostream> 
using namespace std;

//
TrackerizerFP420::TrackerizerFP420(const edm::ParameterSet& conf):conf_(conf)     {
  
  edm::ParameterSet m_Anal = conf.getParameter<edm::ParameterSet>("TrackerizerFP420");
  verbosity    = m_Anal.getParameter<int>("Verbosity");
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


void TrackerizerFP420::produce(ClusterCollectionFP420 & input, TrackCollectionFP420 & toutput)
{
  //  beginJob;
  
  //    put zero to container info from the beginning (important! because not any detID is updated with coming of new event     !!!!!!   
  // clean info of container from previous event
  
  std::vector<TrackFP420> collector;
  collector.clear();
  TrackCollectionFP420::Range inputRange;
  inputRange.first = collector.begin();
  inputRange.second = collector.end();
  
  unsigned int detID = 0;
  toutput.putclear(inputRange,detID);
  
  unsigned  int StID = 1111;
  toutput.putclear(inputRange,StID);
  
  
  //                                                                                                                      !!!!!!   
  // if we want to keep Track container/Collection for one event --->   uncomment the line below and vice versa
  toutput.clear();   //container_.clear() --> start from the beginning of the container
  
  //                                RUN now:                                                                                 !!!!!!     
  //   startFP420TrackMain_.run(input, toutput);
  sFP420TrackMain_->run(input, toutput);
  // cout <<"=======           TrackerizerFP420:                    end of produce     " << endl;
  
}

