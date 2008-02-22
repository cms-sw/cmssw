#ifndef RecoTauTag_RecoTau_CaloRecoTauProducer
#define RecoTauTag_RecoTau_CaloRecoTauProducer

/* class CaloRecoTauProducer
 * EDProducer of the CaloTauCollection, starting from the CaloTauTagInfoCollection, 
 * authors: Simone Gennai (simone.gennai@cern.ch), Ludovic Houchu (Ludovic.Houchu@cern.ch)
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoTauTag/RecoTau/interface/CaloRecoTauAlgorithm.h"

#include "CLHEP/Random/RandGauss.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class CaloRecoTauProducer : public EDProducer {
 public:
  explicit CaloRecoTauProducer(const ParameterSet& iConfig);
  ~CaloRecoTauProducer();
  virtual void produce(Event&,const EventSetup&);
 private:
  InputTag CaloRecoTauTagInfoProducer_;
  string PVProducer_;
  double smearedPVsigmaX_;
  double smearedPVsigmaY_;
  double smearedPVsigmaZ_;
  double JetMinPt_;
  CaloRecoTauAlgorithm* CaloRecoTauAlgo_;
};
#endif

