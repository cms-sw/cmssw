#ifndef RecoTauTag_CaloRecoTauTagInfoProducer
#define RecoTauTag_CaloRecoTauTagInfoProducer

/* class CaloRecoTauTagInfoProducer 
 * returns a CaloTauTagInfo collection starting from a JetTrackAssociations <a CaloJet,a list of Track's> collection,
 * created: Aug 28 2007,
 * revised: ,
 * authors: Ludovic Houchu
 */

#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/TauReco/interface/CaloTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoTauTag/RecoTau/interface/CaloRecoTauTagInfoAlgorithm.h"

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

class CaloRecoTauTagInfoProducer : public EDProducer {
 public:
  explicit CaloRecoTauTagInfoProducer(const ParameterSet&);
  ~CaloRecoTauTagInfoProducer();
  virtual void produce(Event&,const EventSetup&);
 private:
  CaloRecoTauTagInfoAlgorithm* CaloRecoTauTagInfoAlgo_;
  string CaloJetTracksAssociatorProducer_;
  string PVProducer_;
  double smearedPVsigmaX_;
  double smearedPVsigmaY_;
  double smearedPVsigmaZ_;  
};
#endif

