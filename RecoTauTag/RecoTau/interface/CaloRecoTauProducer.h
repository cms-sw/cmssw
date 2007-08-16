#ifndef RecoTauTag_CaloRecoTau
#define RecoTauTag_CaloRecoTau

/* class CaloRecoTau
 * EDProducer of the tagged TauJet with the PFConeIsolationAlgorithm, 
 * authors: Simone Gennai, Ludovic Houchu
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/BTauReco/interface/CombinedTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoTauTag/RecoTau/interface/CaloRecoTauAlgorithm.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class CaloRecoTauProducer : public EDProducer {
 public:
  explicit CaloRecoTauProducer(const ParameterSet& iConfig){
    CaloTagInfo_  = iConfig.getParameter<InputTag>("CombinedTauTagInfo");
    PVProducer_ = iConfig.getParameter<string>("PVProducer");
    JetMinPt_  = iConfig.getParameter<double>("JetPtMin");
    CaloRecoTauAlgo_=new CaloRecoTauAlgorithm(iConfig);
    produces<TauCollection>();      
  }
  ~CaloRecoTauProducer(){
    delete CaloRecoTauAlgo_;
  }
  virtual void produce(Event&,const EventSetup&);
 private:
  InputTag CaloTagInfo_;
  string PVProducer_;
  double JetMinPt_;
  CaloRecoTauAlgorithm* CaloRecoTauAlgo_;
};
#endif

