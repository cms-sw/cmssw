#ifndef RecoTauTag_PFRecoTauTagInfoProducer
#define RecoTauTag_PFRecoTauTagInfoProducer

/* class PFRecoTauTagInfoProducer
 * returns a PFTauTagInfo collection starting from a JetTrackAssociations <a PFJet,a list of Track's> collection,
 * created: Aug 28 2007,
 * revised: ,
 * authors: Ludovic Houchu
 */

#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/TauReco/interface/TauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

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

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFRecoTauTagInfoProducer : public EDProducer {
 public:
  explicit PFRecoTauTagInfoProducer(const ParameterSet& iConfig){
    PFJetTracksAssociatormodule_        = iConfig.getParameter<string>("PFJetTracksAssociatormodule");
    PVmodule_                           = iConfig.getParameter<string>("PVmodule");
    smearedPVsigmaX_                    = iConfig.getParameter<double>("smearedPVsigmaX");
    smearedPVsigmaY_                    = iConfig.getParameter<double>("smearedPVsigmaY");
    smearedPVsigmaZ_                    = iConfig.getParameter<double>("smearedPVsigmaZ");	
    PFRecoTauTagInfoAlgo_=new PFRecoTauTagInfoAlgorithm(iConfig);
    produces<TauTagInfoCollection>();      
  }
  ~PFRecoTauTagInfoProducer(){
    delete PFRecoTauTagInfoAlgo_;
  }
  virtual void produce(Event&,const EventSetup&);
 private:
  PFRecoTauTagInfoAlgorithm* PFRecoTauTagInfoAlgo_;
  string PFJetTracksAssociatormodule_;
  string PVmodule_;
  double smearedPVsigmaX_;
  double smearedPVsigmaY_;
  double smearedPVsigmaZ_;  
};
#endif

