#ifndef TrackerizerFP420_h
#define TrackerizerFP420_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "RecoRomanPot/RecoFP420/interface/FP420TrackMain.h"

#include "RecoRomanPot/RecoFP420/interface/ClusterCollectionFP420.h"

#include "RecoRomanPot/RecoFP420/interface/TrackFP420.h"
#include "RecoRomanPot/RecoFP420/interface/TrackCollectionFP420.h"

#include<vector>
#include<map>
#include<iostream>
using namespace std;



class TrackerizerFP420 : public SimWatcher 
{
 public:
  
  explicit TrackerizerFP420(const edm::ParameterSet& conf);
  //TrackerizerFP420();
  
  virtual ~TrackerizerFP420();
  
  virtual void beginJob();
  
  virtual void produce(ClusterCollectionFP420 &, TrackCollectionFP420 &);
  
 private:
  edm::ParameterSet conf_;
  FP420TrackMain* sFP420TrackMain_;
  //  FP420TrackMain startFP420TrackMain_;
  //bool UseNoiseBadElectrodeFlagFromDB_;
  int verbosity;
};

#endif
