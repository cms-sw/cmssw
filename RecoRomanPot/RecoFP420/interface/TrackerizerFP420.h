#ifndef TrackerizerFP420_h
#define TrackerizerFP420_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "RecoRomanPot/RecoFP420/interface/FP420TrackMain.h"

#include "DataFormats/FP420Cluster/interface/ClusterCollectionFP420.h"

#include "DataFormats/FP420Cluster/interface/TrackFP420.h"
#include "DataFormats/FP420Cluster/interface/TrackCollectionFP420.h"

#include <string>
#include<vector>
#include<map>
#include<iostream>




namespace cms
{
  class TrackerizerFP420: public edm::EDProducer
  {
  public:
    
    explicit TrackerizerFP420(const edm::ParameterSet& conf);
    //TrackerizerFP420();
    
    virtual ~TrackerizerFP420();
    
    virtual void beginJob();
    
    //  virtual void produce(ClusterCollectionFP420 &, TrackCollectionFP420 &);
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
    
  private:
    typedef std::vector<std::string> vstring;
    edm::ParameterSet conf_;
    vstring trackerContainers;

    FP420TrackMain* sFP420TrackMain_;
    //  FP420TrackMain startFP420TrackMain_;
    //bool UseNoiseBadElectrodeFlagFromDB_;
    int verbosity;
  };
}
#endif
