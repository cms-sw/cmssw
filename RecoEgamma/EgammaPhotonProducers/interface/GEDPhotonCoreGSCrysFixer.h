#ifndef RecoEgamma_EgammaPhotonProducers_GEDPhotonCoreGSCrysFixer_h
#define RecoEgamma_EgammaPhotonProducers_GEDPhotonCoreGSCrysFixer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h" 
#include "FWCore/Framework/interface/ESHandle.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/Handle.h" 
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

#include <iostream>
#include <string>

class GEDPhotonCoreGSCrysFixer : public edm::stream::EDProducer<> {
public:
  explicit GEDPhotonCoreGSCrysFixer(const edm::ParameterSet& );
  virtual ~GEDPhotonCoreGSCrysFixer(){}
  
  void produce(edm::Event&, const edm::EventSetup& ) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, 
			    edm::EventSetup const&) override;
  

  template<typename T>
  void getToken(edm::EDGetTokenT<T>& token,const edm::ParameterSet& pset,const std::string& label){
    token=consumes<T>(pset.getParameter<edm::InputTag>(label));
  }
private:
  edm::EDGetTokenT<reco::PhotonCoreCollection> orgCoresToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ebRecHitsToken_;
  edm::EDGetTokenT<edm::ValueMap<reco::SuperClusterRef> > oldRefinedSCToNewMapToken_;
  edm::EDGetTokenT<edm::ValueMap<reco::SuperClusterRef> > oldSCToNewMapToken_;
  const CaloTopology* topology_;
  
};
#endif
