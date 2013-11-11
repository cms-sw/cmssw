// Author:  Alfredo Gurrola
//(20/11/08 make MET and JET logic independent   /Grigory Safronov)

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HLTrigger/special/interface/HLTHcalNoiseFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

HLTHcalNoiseFilter::HLTHcalNoiseFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
  JetSource_ = iConfig.getParameter<edm::InputTag>("JetSource");
  MetSource_ = iConfig.getParameter<edm::InputTag>("MetSource");
  TowerSource_ = iConfig.getParameter<edm::InputTag>("TowerSource");
  useMet_ = iConfig.getParameter<bool>("UseMET");
  useJet_ = iConfig.getParameter<bool>("UseJet");
  MetCut_ = iConfig.getParameter<double>("MetCut");
  JetMinE_ = iConfig.getParameter<double>("JetMinE");
  JetHCALminEnergyFraction_ = iConfig.getParameter<double>("JetHCALminEnergyFraction");

  if (useMet_) {
    MetSourceToken_ = consumes<reco::CaloMETCollection>(MetSource_);
  }
  if (useJet_) {
    JetSourceToken_ = consumes<reco::CaloJetCollection>(JetSource_);
    TowerSourceToken_ = consumes<CaloTowerCollection>(TowerSource_);
  }
}

HLTHcalNoiseFilter::~HLTHcalNoiseFilter() { }

void
HLTHcalNoiseFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("JetSource",edm::InputTag("iterativeCone5CaloJets"));
  desc.add<edm::InputTag>("MetSource",edm::InputTag("met"));
  desc.add<edm::InputTag>("TowerSource",edm::InputTag("towerMaker"));
  desc.add<bool>("UseJet",true);
  desc.add<bool>("UseMET",false);
  desc.add<double>("MetCut",0.);
  desc.add<double>("JetMinE",20.);
  desc.add<double>("JetHCALminEnergyFraction",0.98);
  descriptions.add("hltHcalNoiseFilter",desc);
}

//
// member functions
//

bool HLTHcalNoiseFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
   using namespace edm;
   using namespace reco;

   bool isAnomalous_BasedOnMET = false;
   bool isAnomalous_BasedOnEnergyFraction=false;

   if (useMet_)
     {
       Handle <CaloMETCollection> metHandle;
       iEvent.getByToken(MetSourceToken_, metHandle);
       const CaloMETCollection *metCol = metHandle.product();
       const CaloMET met = metCol->front();

       if(met.pt() > MetCut_) isAnomalous_BasedOnMET=true;
     }

   if (useJet_)
     {
       Handle<CaloJetCollection> calojetHandle;
       iEvent.getByToken(JetSourceToken_,calojetHandle);

       Handle<CaloTowerCollection> towerHandle;
       iEvent.getByToken(TowerSourceToken_, towerHandle);

       std::vector<CaloTower> TowerContainer;
       std::vector<CaloJet> JetContainer;
       TowerContainer.clear();
       JetContainer.clear();
       CaloTower seedTower;
       for(CaloJetCollection::const_iterator calojetIter = calojetHandle->begin();calojetIter != calojetHandle->end();++calojetIter) {
	 if( ((calojetIter->et())*cosh(calojetIter->eta()) > JetMinE_) and (calojetIter->energyFractionHadronic() > JetHCALminEnergyFraction_) ) {
	   JetContainer.push_back(*calojetIter);
	   double maxTowerE = 0.0;
	   for(CaloTowerCollection::const_iterator kal = towerHandle->begin(); kal != towerHandle->end(); kal++) {
	     double dR = deltaR((*calojetIter).eta(),(*calojetIter).phi(),(*kal).eta(),(*kal).phi());
	     if( (dR < 0.50) and (kal->p() > maxTowerE) ) {
	       maxTowerE = kal->p();
	       seedTower = *kal;
	     }
	   }
	   TowerContainer.push_back(seedTower);
	 }
	
       }
       if(JetContainer.size() > 0) {
	 isAnomalous_BasedOnEnergyFraction = true;
       }
     }

   return ((useMet_ and isAnomalous_BasedOnMET) or (useJet_ and isAnomalous_BasedOnEnergyFraction));
}
