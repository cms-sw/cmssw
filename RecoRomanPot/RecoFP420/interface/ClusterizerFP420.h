#ifndef ClusterizerFP420_h
#define ClusterizerFP420_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "RecoRomanPot/RecoFP420/interface/FP420ClusterMain.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterNoiseFP420.h"

#include "SimRomanPot/SimFP420/interface/DigiCollectionFP420.h"

#include "RecoRomanPot/RecoFP420/interface/ClusterFP420.h"
#include "RecoRomanPot/RecoFP420/interface/ClusterCollectionFP420.h"

#include<vector>
#include<map>
#include<iostream>
using namespace std;



//namespace cms
//{
  class ClusterizerFP420: public SimWatcher 
  {
  public:

     explicit ClusterizerFP420(const edm::ParameterSet& conf);

    virtual ~ClusterizerFP420();

    virtual void beginJob();

    //  virtual void produce(DigiCollectionFP420*, ClusterCollectionFP420 &);
    virtual void produce(DigiCollectionFP420 &, ClusterCollectionFP420 &);

  private:
    edm::ParameterSet conf_;
    FP420ClusterMain* sClusterizerFP420_;


    std::vector<ClusterNoiseFP420> noise;
    bool UseNoiseBadElectrodeFlagFromDB_;
    int sn0, pn0;
    int verbosity;
  };
//}
#endif
