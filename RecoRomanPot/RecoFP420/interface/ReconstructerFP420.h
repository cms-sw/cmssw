#ifndef ReconstructerFP420_h
#define ReconstructerFP420_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoRomanPot/RecoFP420/interface/FP420RecoMain.h"

#include "DataFormats/FP420Cluster/interface/TrackCollectionFP420.h"
#include "DataFormats/FP420Cluster/interface/RecoFP420.h"
#include "DataFormats/FP420Cluster/interface/RecoCollectionFP420.h"

#include <string>
#include<vector>
#include<map>
#include<iostream>



namespace cms
{
  class ReconstructerFP420: public edm::EDProducer
  {
  public:
    
    explicit ReconstructerFP420(const edm::ParameterSet& conf);
    //ReconstructerFP420();
    
    virtual ~ReconstructerFP420();
    
    virtual void beginJob();
    
    //  virtual void produce(ClusterCollectionFP420 &, RecoCollectionFP420 &);
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
    
  private:
    typedef std::vector<std::string> vstring;
    edm::ParameterSet conf_;
    vstring trackerContainers;

    FP420RecoMain* sFP420RecoMain_;
    int verbosity;
    int VtxFlag;
    std::string m_genReadoutName;

  };
}
#endif
