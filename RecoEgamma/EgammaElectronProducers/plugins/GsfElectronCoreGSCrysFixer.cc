#ifndef RecoEgamma_EgammaElectronProducers_GsfElectronCoreGSCrysFixer_h
#define RecoEgamma_EgammaElectronProducers_GsfElectronCoreGSCrysFixer_h

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
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

#include <iostream>
#include <string>

class GsfElectronCoreGSCrysFixer : public edm::stream::EDProducer<> {
public:
  explicit GsfElectronCoreGSCrysFixer(const edm::ParameterSet& );
  virtual ~GsfElectronCoreGSCrysFixer(){}
  
  void produce(edm::Event&, const edm::EventSetup& ) override;

  template<typename T>
  void getToken(edm::EDGetTokenT<T>& token,const edm::ParameterSet& pset,const std::string& label, const std::string& instance = ""){
    auto tag(pset.getParameter<edm::InputTag>(label));
    if (!instance.empty())
      tag = edm::InputTag(tag.label(), instance, tag.process());

    token = consumes<T>(tag);
  }
private:
  typedef edm::ValueMap<reco::SuperClusterRef> SCRefMap;

  edm::EDGetTokenT<reco::GsfElectronCoreCollection> orgCoresToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> refinedSCsToken_; // new
  edm::EDGetTokenT<SCRefMap> refinedSCMapToken_; // new->old
  edm::EDGetTokenT<reco::SuperClusterCollection> ebSCsToken_; // new
  edm::EDGetTokenT<SCRefMap> ebSCMapToken_; // new->old
  edm::EDGetTokenT<reco::SuperClusterCollection> eeSCsToken_; // new
  edm::EDGetTokenT<SCRefMap> eeSCMapToken_; // new->old
};

namespace {
  template<typename T> edm::Handle<T> getHandle(const edm::Event& iEvent,const edm::EDGetTokenT<T>& token){
    edm::Handle<T> handle;
    iEvent.getByToken(token,handle);
    return handle;
  }
}


GsfElectronCoreGSCrysFixer::GsfElectronCoreGSCrysFixer( const edm::ParameterSet & pset )
{
  getToken(orgCoresToken_,pset,"orgCores");
  getToken(refinedSCsToken_, pset, "refinedSCs");
  getToken(refinedSCMapToken_, pset, "refinedSCs");
  getToken(ebSCsToken_, pset, "scs", "particleFlowSuperClusterECALBarrel");
  getToken(ebSCMapToken_, pset, "refinedSCs", "parentSCsEB");
  getToken(eeSCsToken_, pset, "scs", "particleFlowSuperClusterECALEndcapWithPreshower");
  getToken(eeSCMapToken_, pset, "refinedSCs", "parentSCsEE");
  
  produces<reco::GsfElectronCoreCollection >();
  produces<SCRefMap>(); // new core to old SC
}


void GsfElectronCoreGSCrysFixer::produce(edm::Event & iEvent, const edm::EventSetup &)
{
  auto outCores = std::make_unique<reco::GsfElectronCoreCollection>();
  
  auto eleCoresHandle = getHandle(iEvent,orgCoresToken_);
  auto refinedSCs(getHandle(iEvent, refinedSCsToken_));
  auto& refinedSCMap(*getHandle(iEvent, refinedSCMapToken_));
  auto ebSCs(getHandle(iEvent, ebSCsToken_));
  auto& ebSCMap(*getHandle(iEvent, ebSCMapToken_));
  auto eeSCs(getHandle(iEvent, eeSCsToken_));
  auto& eeSCMap(*getHandle(iEvent, eeSCMapToken_));

  std::vector<reco::SuperClusterRef> oldSCRefs;
  
  for (auto& inCore : *eleCoresHandle) {
    outCores->emplace_back(inCore);
    auto& outCore(outCores->back());

    // NOTE: These mappings can result in NULL superclusters!
    auto& oldRefinedSC(inCore.superCluster());
    outCore.setSuperCluster(GainSwitchTools::findNewRef(oldRefinedSC, refinedSCs, refinedSCMap));

    oldSCRefs.push_back(oldRefinedSC);

    auto& parentSC(inCore.parentSuperCluster());
    if (parentSC.isNonnull()) {
      if (parentSC->seed()->seed().subdetId() == EcalBarrel)
        outCore.setParentSuperCluster(GainSwitchTools::findNewRef(parentSC, ebSCs, ebSCMap));
      else
        outCore.setParentSuperCluster(GainSwitchTools::findNewRef(parentSC, eeSCs, eeSCMap));
    }
  }
  
  auto newCoresHandle(iEvent.put(std::move(outCores)));

  std::auto_ptr<SCRefMap> pRefMap(new SCRefMap);
  SCRefMap::Filler refMapFiller(*pRefMap);
  refMapFiller.insert(newCoresHandle, oldSCRefs.begin(), oldSCRefs.end());
  refMapFiller.fill();
  iEvent.put(pRefMap);
}

DEFINE_FWK_MODULE(GsfElectronCoreGSCrysFixer);
#endif
