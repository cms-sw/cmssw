#ifndef ClusterizerFP420_h
#define ClusterizerFP420_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
//#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "RecoRomanPot/RecoFP420/interface/FP420ClusterMain.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterNoiseFP420.h"

#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"

#include "DataFormats/FP420Cluster/interface/ClusterFP420.h"
#include "DataFormats/FP420Cluster/interface/ClusterCollectionFP420.h"

#include <CLHEP/Vector/ThreeVector.h>
#include <string>
#include<vector>
#include<map>
#include<iostream>




namespace cms
{
  class ClusterizerFP420: public edm::EDProducer
  {
  public:
    
    explicit ClusterizerFP420(const edm::ParameterSet& conf);
    
    virtual ~ClusterizerFP420();
    
    virtual void beginJob();
    
    //  virtual void produce(DigiCollectionFP420*, ClusterCollectionFP420 &);
    // virtual void produce(DigiCollectionFP420 &, ClusterCollectionFP420 &);
    
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
    
  private:
    typedef std::vector<std::string> vstring;




    edm::ParameterSet conf_;
    vstring trackerContainers;

    FP420ClusterMain* sClusterizerFP420_;

    ClusterCollectionFP420* soutput;
   
    FP420NumberingScheme * theFP420NumberingScheme;
    
    std::vector<ClusterNoiseFP420> noise;
    bool UseNoiseBadElectrodeFlagFromDB_;
    int sn0, pn0, dn0, rn0;
    int verbosity;
  };
}
#endif
