#include "RecoTauTag/RecoTau/interface/CaloRecoTauDiscriminationAgainstElectron.h"

void CaloRecoTauDiscriminationAgainstElectron::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<CaloTauCollection> theCaloTauCollection;
  iEvent.getByLabel(CaloTauProducer_,theCaloTauCollection);

  // fill the AssociationVector object
  auto_ptr<CaloTauDiscriminator> theCaloTauDiscriminatorAgainstElectron(new CaloTauDiscriminator(CaloTauRefProd(theCaloTauCollection)));

  for(size_t iCaloTau=0;iCaloTau<theCaloTauCollection->size();++iCaloTau) {
    CaloTauRef theCaloTauRef(theCaloTauCollection,iCaloTau);
    if (!(*theCaloTauRef).leadTrack()){
      theCaloTauDiscriminatorAgainstElectron->setValue(iCaloTau,0);
      continue;
    }
    if (ApplyCut_maxleadTrackHCAL3x3hottesthitDEta_){
      // optional selection : ask for small |deta| between direction of propag. leading Track - ECAL inner surf. contact point and direction of highest Et hit among HCAL hits inside a 3x3 calo. tower matrix centered on direction of propag. leading Track - ECAL inner surf. contact point
      if (isnan((*theCaloTauRef).leadTrackHCAL3x3hottesthitDEta()) || (*theCaloTauRef).leadTrackHCAL3x3hottesthitDEta()>maxleadTrackHCAL3x3hottesthitDEta_){
	theCaloTauDiscriminatorAgainstElectron->setValue(iCaloTau,0);
	continue;
      }
    }
    if (ApplyCut_leadTrackavoidsECALcrack_){
      // optional selection : ask that leading track - ECAL inner surface contact point does not fall inside any ECAL eta crack 
      ESHandle<MagneticField> theMagneticField;
      iEventSetup.get<IdealMagneticFieldRecord>().get(theMagneticField);
      math::XYZPoint thepropagleadTrackECALSurfContactPoint=TauTagTools::propagTrackECALSurfContactPoint(theMagneticField.product(),(*theCaloTauRef).leadTrack());
      if(thepropagleadTrackECALSurfContactPoint.R()==0. ||
	 fabs(thepropagleadTrackECALSurfContactPoint.eta())<ECALBounds::crack_absEtaIntervalA().second || 
	 (fabs(thepropagleadTrackECALSurfContactPoint.eta())>ECALBounds::crack_absEtaIntervalB().first && fabs(thepropagleadTrackECALSurfContactPoint.eta())<ECALBounds::crack_absEtaIntervalB().second) ||
	 (fabs(thepropagleadTrackECALSurfContactPoint.eta())>ECALBounds::crack_absEtaIntervalC().first && fabs(thepropagleadTrackECALSurfContactPoint.eta())<ECALBounds::crack_absEtaIntervalC().second) ||
	 (fabs(thepropagleadTrackECALSurfContactPoint.eta())>ECALBounds::crack_absEtaIntervalD().first && fabs(thepropagleadTrackECALSurfContactPoint.eta())<ECALBounds::crack_absEtaIntervalD().second) ||
	 (fabs(thepropagleadTrackECALSurfContactPoint.eta())>ECALBounds::crack_absEtaIntervalE().first && fabs(thepropagleadTrackECALSurfContactPoint.eta())<ECALBounds::crack_absEtaIntervalE().second)
	 ){
	theCaloTauDiscriminatorAgainstElectron->setValue(iCaloTau,0);
	continue;
      }     
    }
    if (isnan((*theCaloTauRef).leadTrackHCAL3x3hitsEtSum())){
      theCaloTauDiscriminatorAgainstElectron->setValue(iCaloTau,0);
    }else{
      if ((*theCaloTauRef).leadTrackHCAL3x3hitsEtSum()/(*theCaloTauRef).leadTrack()->pt()<=leadTrack_HCAL3x3hitsEtSumOverPt_minvalue_) theCaloTauDiscriminatorAgainstElectron->setValue(iCaloTau,0);
      else theCaloTauDiscriminatorAgainstElectron->setValue(iCaloTau,1);
    }
  }
   
  iEvent.put(theCaloTauDiscriminatorAgainstElectron);
}
