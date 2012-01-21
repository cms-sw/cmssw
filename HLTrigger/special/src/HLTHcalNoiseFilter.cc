// Author:  Alfredo Gurrola 
//(20/11/08 make MET and JET logic independent   /Grigory Safronov)

#include "HLTrigger/special/interface/HLTHcalNoiseFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
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

  nAnomalousEvents=0;
  nEvents=0;
}

HLTHcalNoiseFilter::~HLTHcalNoiseFilter() { }

bool HLTHcalNoiseFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
   using namespace edm;
   using namespace reco;

   bool isAnomalous_BasedOnMET = false;
   bool isAnomalous_BasedOnEnergyFraction=false; 

   if (useMet_)
     {
       Handle <CaloMETCollection> metHandle;
       iEvent.getByLabel(MetSource_, metHandle);
       const CaloMETCollection *metCol = metHandle.product();
       const CaloMET met = metCol->front();
    
       if(met.pt() > MetCut_) isAnomalous_BasedOnMET=true;
     }
       
   if (useJet_)
     {
       Handle<CaloJetCollection> calojetHandle;
       iEvent.getByLabel(JetSource_,calojetHandle);
       
       Handle<CaloTowerCollection> towerHandle;
       iEvent.getByLabel(TowerSource_, towerHandle);

       std::vector<CaloTower> TowerContainer;
       std::vector<CaloJet> JetContainer;
       TowerContainer.clear();
       JetContainer.clear();
       CaloTower seedTower;
       nEvents++;
       for(CaloJetCollection::const_iterator calojetIter = calojetHandle->begin();calojetIter != calojetHandle->end();++calojetIter) {
	 if( ((calojetIter->et())*cosh(calojetIter->eta()) > JetMinE_) && (calojetIter->energyFractionHadronic() > JetHCALminEnergyFraction_) ) {
	   JetContainer.push_back(*calojetIter);
	   double maxTowerE = 0.0;
	   for(CaloTowerCollection::const_iterator kal = towerHandle->begin(); kal != towerHandle->end(); kal++) {
	     double dR = deltaR((*calojetIter).eta(),(*calojetIter).phi(),(*kal).eta(),(*kal).phi());
	     if( (dR < 0.50) && (kal->p() > maxTowerE) ) {
	       maxTowerE = kal->p();
	       seedTower = *kal;
	     }
	   }
	   TowerContainer.push_back(seedTower);
	 }
	 
       }
       if(JetContainer.size() > 0) {
	 nAnomalousEvents++;
	 isAnomalous_BasedOnEnergyFraction = true;
       }
     }
   
   return ((useMet_&&isAnomalous_BasedOnMET)||(useJet_&&isAnomalous_BasedOnEnergyFraction));
}
