#ifndef RecoEgamma_EgammaElectronProducers_GsfEleGSCrysSimpleFixer_h
#define RecoEgamma_EgammaElectronProducers_GsfEleGSCrysSimpleFixer_h

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
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

#include <iostream>
#include <string>

class GsfEleGSCrysSimpleFixer : public edm::stream::EDProducer<> {
public:
  explicit GsfEleGSCrysSimpleFixer(const edm::ParameterSet& );
  virtual ~GsfEleGSCrysSimpleFixer(){}
  
  void produce(edm::Event&, const edm::EventSetup& ) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, 
			    edm::EventSetup const&) override;
  

  template<typename T>
  void getToken(edm::EDGetTokenT<T>& token,const edm::ParameterSet& pset,const std::string& label){
    token=consumes<T>(pset.getParameter<edm::InputTag>(label));
  }
private:
  edm::EDGetTokenT<reco::GsfElectronCollection> oldGsfElesToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ebMultiAndWeightsRecHitsToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ebMultiRecHitsToken_;
  
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



GsfEleGSCrysSimpleFixer::GsfEleGSCrysSimpleFixer( const edm::ParameterSet & pset )
{

  getToken(oldGsfElesToken_,pset,"oldEles");
  getToken(ebMultiAndWeightsRecHitsToken_,pset,"ebMultiAndWeightsRecHits");
  getToken(ebMultiRecHitsToken_,pset,"ebMultiRecHits");
  

  produces<reco::GsfElectronCollection >();
}

void GsfEleGSCrysSimpleFixer::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  auto outEles = std::make_unique<reco::GsfElectronCollection>();
 

  auto elesHandle = getHandle(iEvent,oldGsfElesToken_);
  auto& ebMultiRecHits = *getHandle(iEvent,ebMultiRecHitsToken_);
  auto& ebMultiAndWeightsRecHits = *getHandle(iEvent,ebMultiAndWeightsRecHitsToken_);

  
  for(size_t eleNr=0;eleNr<elesHandle->size();eleNr++){
    reco::GsfElectronRef eleRef(elesHandle,eleNr);
    if(GainSwitchTools::hasEBGainSwitchIn5x5(*eleRef->superCluster(),&ebMultiRecHits,topology_)){
      
      reco::GsfElectron newEle(*eleRef);
      
      std::vector<DetId> gsIds = GainSwitchTools::gainSwitchedIdsIn5x5(eleRef->superCluster()->seed()->seed(),
								       &ebMultiRecHits,topology_);
      float newRawEnergy = GainSwitchTools::newRawEnergyNoFracs(*eleRef->superCluster(),gsIds,
							      &ebMultiRecHits,&ebMultiAndWeightsRecHits);
      float energyCorr = newRawEnergy / eleRef->superCluster()->rawEnergy();
      
      reco::GsfElectron::ShowerShape full5x5ShowerShape = GainSwitchTools::redoEcalShowerShape<true>(newEle.full5x5_showerShape(),newEle.superCluster(),&ebMultiAndWeightsRecHits,topology_,geometry_);
      reco::GsfElectron::ShowerShape showerShape = GainSwitchTools::redoEcalShowerShape<false>(newEle.showerShape(),newEle.superCluster(),&ebMultiAndWeightsRecHits,topology_,geometry_);
      //so the no fractions showershape had hcalDepth1/2 corrected by the regression energy, hence we need to know the type
      GainSwitchTools::correctHadem(showerShape,energyCorr,GainSwitchTools::ShowerShapeType::Fractions);
      GainSwitchTools::correctHadem(full5x5ShowerShape,energyCorr,GainSwitchTools::ShowerShapeType::Full5x5);
      newEle.full5x5_setShowerShape(full5x5ShowerShape);
      newEle.setShowerShape(showerShape);   
      
      newEle.setCorrectedEcalEnergy(newEle.ecalEnergy()*energyCorr);
      newEle.setCorrectedEcalEnergyError(newEle.ecalEnergyError()*energyCorr);

      //meh, somebody else can sort this out
      //the energy should for gain switch electrons be all from the ECAL so we will assume that
      //to make my life easier
      math::XYZTLorentzVector newMom(newEle.p4().x()/newEle.p4().t()*newEle.ecalEnergy(),
				     newEle.p4().y()/newEle.p4().t()*newEle.ecalEnergy(),
				     newEle.p4().z()/newEle.p4().t()*newEle.ecalEnergy(),
				     newEle.ecalEnergy());
      newEle.correctMomentum(newMom,newEle.trackMomentumError(),newEle.correctedEcalEnergyError());


      outEles->push_back(newEle);
    }else{
      outEles->push_back(*eleRef);
    }
  }
  
  iEvent.put(std::move(outEles));
}

void GsfEleGSCrysSimpleFixer::beginLuminosityBlock(edm::LuminosityBlock const& lb, 
						 edm::EventSetup const& es) {
  edm::ESHandle<CaloGeometry> caloGeom ;
  edm::ESHandle<CaloTopology> caloTopo ;
  es.get<CaloGeometryRecord>().get(caloGeom);
  es.get<CaloTopologyRecord>().get(caloTopo);
  geometry_ = caloGeom.product();
  topology_ = caloTopo.product();
}




DEFINE_FWK_MODULE(GsfEleGSCrysSimpleFixer);
#endif

