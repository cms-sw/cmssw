
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowerCandidateCreator.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"

#include <vector>
#include <cmath>
#include "TMath.h"

class HIhfFilter_miniAOD : public edm::stream::EDProducer<> {
   public:
      explicit HIhfFilter_miniAOD(const edm::ParameterSet&);
      ~HIhfFilter_miniAOD() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      edm::EDGetTokenT<CaloTowerCollection>  srcTowers_;

};

HIhfFilter_miniAOD::HIhfFilter_miniAOD(const edm::ParameterSet& iConfig):
  srcTowers_(consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("srcTowers")))
{
   produces<std::vector<int>>("HIhfFilters");
}


HIhfFilter_miniAOD::~HIhfFilter_miniAOD()
{
}


void
HIhfFilter_miniAOD::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std; 
  
  Handle<CaloTowerCollection> towers;
  iEvent.getByToken(srcTowers_, towers); 

  int nTowersTh2HFplus_ =0;
  int nTowersTh2HFminus_=0;
  int nTowersTh3HFplus_ =0;
  int nTowersTh3HFminus_=0;
  int nTowersTh4HFplus_ =0;
  int nTowersTh4HFminus_=0;
  int nTowersTh5HFplus_ =0;
  int nTowersTh5HFminus_=0;

  std::size_t size = 4;
  auto HIhfFiltersOut = std::make_unique<std::vector<int>>(size);
  
  for( unsigned int towerIndx = 0; towerIndx<towers->size(); ++towerIndx){
    const CaloTower & tower = (*towers)[ towerIndx ];
    if(tower.et() >= 0.0){
	if(tower.energy() >= 2.0 && tower.eta() > 3.0 && tower.eta() < 6.0) nTowersTh2HFplus_ += 1;
        if(tower.energy() >= 3.0 && tower.eta() > 3.0 && tower.eta() < 6.0) nTowersTh3HFplus_ += 1;
        if(tower.energy() >= 4.0 && tower.eta() > 3.0 && tower.eta() < 6.0) nTowersTh4HFplus_ += 1;
        if(tower.energy() >= 5.0 && tower.eta() > 3.0 && tower.eta() < 6.0) nTowersTh5HFplus_ += 1;
        if(tower.energy() >= 2.0 && tower.eta() < -3.0 && tower.eta() > -6.0) nTowersTh2HFminus_ += 1;
        if(tower.energy() >= 3.0 && tower.eta() < -3.0 && tower.eta() > -6.0) nTowersTh3HFminus_ += 1;
        if(tower.energy() >= 4.0 && tower.eta() < -3.0 && tower.eta() > -6.0) nTowersTh4HFminus_ += 1;
        if(tower.energy() >= 5.0 && tower.eta() < -3.0 && tower.eta() > -6.0) nTowersTh5HFminus_ += 1;
    }
  }

  (*HIhfFiltersOut)[0] = TMath::Min(nTowersTh2HFplus_,nTowersTh2HFminus_);
  (*HIhfFiltersOut)[1] = TMath::Min(nTowersTh3HFplus_,nTowersTh3HFminus_);
  (*HIhfFiltersOut)[2] = TMath::Min(nTowersTh4HFplus_,nTowersTh4HFminus_);
  (*HIhfFiltersOut)[3] = TMath::Min(nTowersTh5HFplus_,nTowersTh5HFminus_);

  iEvent.put(std::move(HIhfFiltersOut),"HIhfFilters");
}

void
HIhfFilter_miniAOD::beginStream(edm::StreamID)
{
}

void
HIhfFilter_miniAOD::endStream()
{
}

void
HIhfFilter_miniAOD::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcTowers",edm::InputTag("towerMaker"));
  descriptions.add("HIhfFilter_miniAOD", desc);
}

DEFINE_FWK_MODULE(HIhfFilter_miniAOD);
