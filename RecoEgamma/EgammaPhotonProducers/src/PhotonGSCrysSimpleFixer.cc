#ifndef RecoEgamma_EgammaElectronProducers_PhotonGSCrysSimpleFixer_h
#define RecoEgamma_EgammaElectronProducers_PhotonGSCrysSimpleFixer_h

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
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

#include <iostream>
#include <string>

class PhotonGSCrysSimpleFixer : public edm::stream::EDProducer<> {
public:
  explicit PhotonGSCrysSimpleFixer(const edm::ParameterSet& );
  virtual ~PhotonGSCrysSimpleFixer(){}
  
  void produce(edm::Event&, const edm::EventSetup& ) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, 
			    edm::EventSetup const&) override;
  

  template<typename T>
  void getToken(edm::EDGetTokenT<T>& token,const edm::ParameterSet& pset,const std::string& label){
    token=consumes<T>(pset.getParameter<edm::InputTag>(label));
  }
private:
  edm::EDGetTokenT<reco::PhotonCollection> oldPhosToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ebMultiAndWeightsRecHitsToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ebMultiRecHitsToken_;

  const std::vector<int> energyTypesToFix_;
  const reco::Photon::P4type energyTypeForP4_;
  
  const CaloTopology* topology_;  
  const CaloGeometry* geometry_;
};


namespace {
  template<typename T> edm::Handle<T> getHandle(const edm::Event& iEvent,const edm::EDGetTokenT<T>& token){
    edm::Handle<T> handle;
    iEvent.getByToken(token,handle);
    return handle;
  }
}



PhotonGSCrysSimpleFixer::PhotonGSCrysSimpleFixer( const edm::ParameterSet & pset ):
  energyTypesToFix_(StringToEnumValue<reco::Photon::P4type>(pset.getParameter<std::vector<std::string> >("energyTypesToFix"))),
  energyTypeForP4_(static_cast<reco::Photon::P4type>(StringToEnumValue<reco::Photon::P4type>(pset.getParameter<std::string>("energyTypeForP4"))))
{ 
  getToken(oldPhosToken_,pset,"oldPhos");
  getToken(ebMultiAndWeightsRecHitsToken_,pset,"ebMultiAndWeightsRecHits");
  getToken(ebMultiRecHitsToken_,pset,"ebMultiRecHits");
  

  produces<reco::PhotonCollection >();
}

void PhotonGSCrysSimpleFixer::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  auto outPhos = std::make_unique<reco::PhotonCollection>();
 

  auto phosHandle = getHandle(iEvent,oldPhosToken_);
  auto& ebMultiRecHits = *getHandle(iEvent,ebMultiRecHitsToken_);
  auto& ebMultiAndWeightsRecHits = *getHandle(iEvent,ebMultiAndWeightsRecHitsToken_);

  
  for(size_t phoNr=0;phoNr<phosHandle->size();phoNr++){
    reco::PhotonRef phoRef(phosHandle,phoNr);
    if(GainSwitchTools::hasEBGainSwitchIn5x5(*phoRef->superCluster(),&ebMultiRecHits,topology_)){
      
      reco::Photon newPho(*phoRef);
      
      std::vector<DetId> gsIds = GainSwitchTools::gainSwitchedIdsIn5x5(phoRef->superCluster()->seed()->seed(),
								       &ebMultiRecHits,topology_);
      float newRawEnergy = GainSwitchTools::newRawEnergyNoFracs(*phoRef->superCluster(),gsIds,
							      &ebMultiRecHits,&ebMultiAndWeightsRecHits);
      float energyCorr = newRawEnergy / phoRef->superCluster()->rawEnergy();
      
      reco::Photon::ShowerShape full5x5ShowerShape = GainSwitchTools::redoEcalShowerShape<true>(newPho.full5x5_showerShapeVariables(),newPho.superCluster(),&ebMultiAndWeightsRecHits,topology_,geometry_);
      reco::Photon::ShowerShape showerShape = GainSwitchTools::redoEcalShowerShape<false>(newPho.showerShapeVariables(),newPho.superCluster(),&ebMultiAndWeightsRecHits,topology_,geometry_);

      
      GainSwitchTools::correctHadem(showerShape,energyCorr);
      GainSwitchTools::correctHadem(full5x5ShowerShape,energyCorr);

      newPho.full5x5_setShowerShapeVariables(full5x5ShowerShape);   
      newPho.setShowerShapeVariables(showerShape);   
      
      for(int typeAsInt : energyTypesToFix_){
	auto type = static_cast<reco::Photon::P4type>(typeAsInt);
	float oldEnergy = newPho.getCorrectedEnergy(type);
	float oldEnergyErr = newPho.getCorrectedEnergyError(type);	
	newPho.setCorrectedEnergy(type,oldEnergy*energyCorr,oldEnergyErr*energyCorr,false);
      }
      
      //now we set the P4 of the object as appropriate
      newPho.setCorrectedEnergy(energyTypeForP4_,newPho.getCorrectedEnergy(energyTypeForP4_),
				newPho.getCorrectedEnergyError(energyTypeForP4_),true);


      outPhos->push_back(newPho);
    }else{
      outPhos->push_back(*phoRef);
    }
  }
  
  iEvent.put(std::move(outPhos));
}

void PhotonGSCrysSimpleFixer::beginLuminosityBlock(edm::LuminosityBlock const& lb, 
						 edm::EventSetup const& es) {
  edm::ESHandle<CaloGeometry> caloGeom ;
  edm::ESHandle<CaloTopology> caloTopo ;
  es.get<CaloGeometryRecord>().get(caloGeom);
  es.get<CaloTopologyRecord>().get(caloTopo);
  geometry_ = caloGeom.product();
  topology_ = caloTopo.product();
}




DEFINE_FWK_MODULE(PhotonGSCrysSimpleFixer);
#endif

