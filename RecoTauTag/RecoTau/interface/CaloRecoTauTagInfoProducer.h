#ifndef RecoTauTag_CaloRecoTauTagInfoProducer
#define RecoTauTag_CaloRecoTauTagInfoProducer

/* class CaloRecoTauTagInfoProducer 
 * returns a TauTagInfo collection starting from a JetTrackAssociations <a CaloJet,a list of Track's> collection,
 * created: Aug 28 2007,
 * revised: ,
 * authors: Ludovic Houchu
 */

#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/TauReco/interface/TauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

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

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class CaloRecoTauTagInfoProducer : public EDProducer {
 public:
  explicit CaloRecoTauTagInfoProducer(const ParameterSet& iConfig){
    CaloJetTracksAssociatormodule_ = iConfig.getParameter<string>("CaloJetTracksAssociatormodule");
    PVmodule_                      = iConfig.getParameter<string>("PVmodule");
    smearedPVsigmaX_               = iConfig.getParameter<double>("smearedPVsigmaX");
    smearedPVsigmaY_               = iConfig.getParameter<double>("smearedPVsigmaY");
    smearedPVsigmaZ_               = iConfig.getParameter<double>("smearedPVsigmaZ");	
    CaloRecoTauTagInfoAlgo_=new CaloRecoTauTagInfoAlgorithm(iConfig);
    produces<TauTagInfoCollection>();      
  }
  ~CaloRecoTauTagInfoProducer(){
    delete CaloRecoTauTagInfoAlgo_;
  }
  virtual void produce(Event&,const EventSetup&);
 private:
  CaloRecoTauTagInfoAlgorithm* CaloRecoTauTagInfoAlgo_;
  string CaloJetTracksAssociatormodule_;
  string PVmodule_;
  double smearedPVsigmaX_;
  double smearedPVsigmaY_;
  double smearedPVsigmaZ_;  
};
#endif

