//*****************************************************************************
// File:      EgammaEcalRecHitIsolationProducer.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************


#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaEcalRecHitIsolationProducer.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

EgammaEcalRecHitIsolationProducer::EgammaEcalRecHitIsolationProducer(const edm::ParameterSet& config) : conf_(config)
{
 // use configuration file to setup input/output collection names
 //inputs 
  emObjectProducer_               = conf_.getParameter<edm::InputTag>("emObjectProducer");
  ecalBarrelRecHitProducer_       = conf_.getParameter<edm::InputTag>("ecalBarrelRecHitProducer");
  ecalBarrelRecHitCollection_     = conf_.getParameter<edm::InputTag>("ecalBarrelRecHitCollection");
  ecalEndcapRecHitProducer_       = conf_.getParameter<edm::InputTag>("ecalEndcapRecHitProducer");
  ecalEndcapRecHitCollection_     = conf_.getParameter<edm::InputTag>("ecalEndcapRecHitCollection");

  //vetos
  egIsoPtMinBarrel_               = conf_.getParameter<double>("etMinBarrel");
  egIsoEMinBarrel_                = conf_.getParameter<double>("eMinBarrel");
  egIsoPtMinEndcap_               = conf_.getParameter<double>("etMinEndcap");
  egIsoEMinEndcap_                = conf_.getParameter<double>("eMinEndcap");
  egIsoConeSizeInBarrel_          = conf_.getParameter<double>("intRadiusBarrel");
  egIsoConeSizeInEndcap_          = conf_.getParameter<double>("intRadiusEndcap");
  egIsoConeSizeOut_               = conf_.getParameter<double>("extRadius");
  egIsoJurassicWidth_             = conf_.getParameter<double>("jurassicWidth");



  // options
  useIsolEt_      = conf_.getParameter<bool>("useIsolEt");
  tryBoth_        = conf_.getParameter<bool>("tryBoth");
  subtract_       = conf_.getParameter<bool>("subtract");
  useNumCrystals_ = conf_.getParameter<bool>("useNumCrystals");
  vetoClustered_  = conf_.getParameter<bool>("vetoClustered");

  //register your products
  produces < edm::ValueMap<double> >();
}


EgammaEcalRecHitIsolationProducer::~EgammaEcalRecHitIsolationProducer(){}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EgammaEcalRecHitIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  // Get the  filtered objects
  edm::Handle< edm::View<reco::Candidate> > emObjectHandle;
  iEvent.getByLabel(emObjectProducer_,emObjectHandle);
    
  // Next get Ecal hits barrel
  edm::Handle<EcalRecHitCollection> ecalBarrelRecHitHandle; //EcalRecHitCollection is a typedef to 
  iEvent.getByLabel(ecalBarrelRecHitProducer_.label(),ecalBarrelRecHitCollection_.label(), ecalBarrelRecHitHandle);

  // Next get Ecal hits endcap
  edm::Handle<EcalRecHitCollection> ecalEndcapRecHitHandle;
  iEvent.getByLabel(ecalEndcapRecHitProducer_.label(), ecalEndcapRecHitCollection_.label(),ecalEndcapRecHitHandle);

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);
  const EcalSeverityLevelAlgo* sevLevel = sevlv.product();

  //Get Calo Geometry
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* caloGeom = pG.product();

  //reco::CandViewDoubleAssociations* isoMap = new reco::CandViewDoubleAssociations( reco::CandidateBaseRefProd( emObjectHandle ) );
  std::auto_ptr<edm::ValueMap<double> > isoMap(new edm::ValueMap<double>());
  edm::ValueMap<double>::Filler filler(*isoMap);
  std::vector<double> retV(emObjectHandle->size(),0);

  EgammaRecHitIsolation ecalBarrelIsol(egIsoConeSizeOut_,egIsoConeSizeInBarrel_,egIsoJurassicWidth_,egIsoPtMinBarrel_,egIsoEMinBarrel_,caloGeom,*ecalBarrelRecHitHandle,sevLevel,DetId::Ecal);
  ecalBarrelIsol.setUseNumCrystals(useNumCrystals_);
  ecalBarrelIsol.setVetoClustered(vetoClustered_);

  EgammaRecHitIsolation ecalEndcapIsol(egIsoConeSizeOut_,egIsoConeSizeInEndcap_,egIsoJurassicWidth_,egIsoPtMinEndcap_,egIsoEMinEndcap_,caloGeom,*ecalEndcapRecHitHandle,sevLevel,DetId::Ecal);
  ecalEndcapIsol.setUseNumCrystals(useNumCrystals_);
  ecalEndcapIsol.setVetoClustered(vetoClustered_);
  
  
  for( size_t i = 0 ; i < emObjectHandle->size(); ++i) {
    
    //i need to know if its in the barrel/endcap so I get the supercluster handle to find out the detector eta
    //this might not be the best way, are we guaranteed that eta<1.5 is barrel
    //this can be safely replaced by another method which determines where the emobject is
    //then we either get the isolation Et or isolation Energy depending on user selection 
    double isoValue =0.;
    
    reco::SuperClusterRef superClus = emObjectHandle->at(i).get<reco::SuperClusterRef>();

    if(tryBoth_){ //barrel + endcap
      if(useIsolEt_) isoValue =  ecalBarrelIsol.getEtSum(&(emObjectHandle->at(i))) + ecalEndcapIsol.getEtSum(&(emObjectHandle->at(i)));
      else           isoValue =  ecalBarrelIsol.getEnergySum(&(emObjectHandle->at(i))) + ecalEndcapIsol.getEnergySum(&(emObjectHandle->at(i)));
    }
    else if( fabs(superClus->eta())<1.479) { //barrel
      if(useIsolEt_) isoValue =  ecalBarrelIsol.getEtSum(&(emObjectHandle->at(i)));
      else           isoValue =  ecalBarrelIsol.getEnergySum(&(emObjectHandle->at(i)));
    }
    else{ //endcap
      if(useIsolEt_) isoValue =  ecalEndcapIsol.getEtSum(&(emObjectHandle->at(i)));
      else           isoValue =  ecalEndcapIsol.getEnergySum(&(emObjectHandle->at(i)));
    }

    //we subtract off the electron energy here as well
    double subtractVal=0;

    if(useIsolEt_) subtractVal = superClus.get()->rawEnergy()*sin(2*atan(exp(-superClus.get()->eta())));
    else           subtractVal = superClus.get()->rawEnergy();
    
    if(subtract_) isoValue-= subtractVal;
    
    retV[i]=isoValue;
    //all done, isolation is now in the map

  }//end of loop over em objects
  
  filler.insert(emObjectHandle,retV.begin(),retV.end());
  filler.fill();

  //std::auto_ptr<reco::CandViewDoubleAssociations> isolMap(isoMap);
  iEvent.put(isoMap);

}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaRecHitIsolation,Producer);
