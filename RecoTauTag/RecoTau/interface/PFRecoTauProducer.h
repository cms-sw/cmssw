#ifndef RecoTauTag_PFRecoTau
#define RecoTauTag_PFRecoTau

/* class PFRecoTau
 * EDProducer of the tagged TauJet with the PFConeIsolationAlgorithm, 
 * authors: Simone Gennai, Ludovic Houchu
 */

#include "DataFormats/BTauReco/interface/PFIsolatedTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithm.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFRecoTauProducer : public EDProducer {
 public:
  explicit PFRecoTauProducer(const ParameterSet& iConfig){
    PFTagInfo_  = iConfig.getParameter<InputTag>("PFTagInfo");
    PVProducer_ = iConfig.getParameter<string>("PVProducer");
    JetMinPt_   = iConfig.getParameter<double>("JetPtMin");
    PFRecoTauAlgo_ = new PFRecoTauAlgorithm(iConfig);
    produces<TauCollection>();      
  }
  ~PFRecoTauProducer(){
    delete PFRecoTauAlgo_;
  }
  virtual void produce(Event&,const EventSetup&);
 private:
  InputTag PFTagInfo_;
  string PVProducer_;
  double JetMinPt_;
  PFRecoTauAlgorithm* PFRecoTauAlgo_;
};
#endif

