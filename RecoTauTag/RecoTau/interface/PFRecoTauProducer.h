#ifndef RecoTauTag_RecoTau_PFRecoTauProducer
#define RecoTauTag_RecoTau_PFRecoTauProducer

/* class PFRecoTauProducer
 * EDProducer of the PFTauCollection, starting from the PFTauTagInfoCollection, 
 * authors: Simone Gennai (simone.gennai@cern.ch), Ludovic Houchu (Ludovic.Houchu@cern.ch)
 */

#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

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

#include "CLHEP/Random/RandGauss.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFRecoTauProducer : public EDProducer {
 public:
  explicit PFRecoTauProducer(const ParameterSet& iConfig);
  ~PFRecoTauProducer();
  virtual void produce(Event&,const EventSetup&);
 private:
  InputTag PFTauTagInfoProducer_;
  InputTag ElectronPreIDProducer_;
  string PVProducer_;
  double smearedPVsigmaX_;
  double smearedPVsigmaY_;
  double smearedPVsigmaZ_;
  double JetMinPt_;
  PFRecoTauAlgorithm* PFRecoTauAlgo_;
};
#endif

