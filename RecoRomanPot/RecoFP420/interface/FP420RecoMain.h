#ifndef FP420RecoMain_h
#define FP420RecoMain_h
   
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/FP420Cluster/interface/TrackFP420.h"
#include "DataFormats/FP420Cluster/interface/TrackCollectionFP420.h"
#include "DataFormats/FP420Cluster/interface/RecoCollectionFP420.h"
#include "RecoRomanPot/RecoFP420/interface/RecoProducerFP420.h"

class RecoProducerFP420;

class FP420RecoMain 
{
 public:
  
  FP420RecoMain(const edm::ParameterSet& conf);
  ~FP420RecoMain();

  /// Runs the algorithm
  void run(edm::Handle<TrackCollectionFP420> &input,
	   std::auto_ptr<RecoCollectionFP420> &toutput,
	   double VtxX, double VtxY, double VtxZ
	   );

 private:


  edm::ParameterSet conf_;
  RecoProducerFP420 *finderParameters_;

  int verbosity;

  double m_rpp420_f;
  double m_rpp420_b;
  double m_zreff;
  double m_zrefb;
  int dn0;

  double zinibeg_;

};

#endif
