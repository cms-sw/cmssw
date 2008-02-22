#ifndef RecoTauTag_PFRecoTauTagInfoProducer
#define RecoTauTag_PFRecoTauTagInfoProducer

/* class PFRecoTauTagInfoProducer
 * returns a PFTauTagInfo collection starting from a JetTrackAssociations <a PFJet,a list of Tracks> collection,
 * created: Aug 28 2007,
 * revised: ,
 * authors: Ludovic Houchu
 */

#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoTauTag/RecoTau/interface/PFRecoTauTagInfoAlgorithm.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandGauss.h"

#include "Math/GenVector/VectorUtil.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFRecoTauTagInfoProducer : public EDProducer {
 public:
  explicit PFRecoTauTagInfoProducer(const ParameterSet& iConfig);
  ~PFRecoTauTagInfoProducer();
  virtual void produce(Event&,const EventSetup&);
 private:
  PFRecoTauTagInfoAlgorithm* PFRecoTauTagInfoAlgo_;
  string PFCandidateProducer_;
  string PFJetTracksAssociatorProducer_;
  string PVProducer_;
  double smearedPVsigmaX_;
  double smearedPVsigmaY_;
  double smearedPVsigmaZ_;  
};
#endif

