#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"
#include "FWCore/Utilities/interface/isFinite.h"

/* class CaloRecoTauDiscriminationAgainstElectron
 * created : Feb 17 2008,
 * revised : ,
 * contributors : Konstantinos Petridis, Sebastien Greder, 
 *                Maiko Takahashi, Alexandre Nikitenko (Imperial College, London), 
 *                Evan Friis (UC Davis)
 */

using namespace reco;

class CaloRecoTauDiscriminationAgainstElectron : public  CaloTauDiscriminationProducerBase {
   public:
      explicit CaloRecoTauDiscriminationAgainstElectron(const edm::ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){   
         CaloTauProducer_                            = iConfig.getParameter<edm::InputTag>("CaloTauProducer");
         leadTrack_HCAL3x3hitsEtSumOverPt_minvalue_  = iConfig.getParameter<double>("leadTrack_HCAL3x3hitsEtSumOverPt_minvalue");  
         ApplyCut_maxleadTrackHCAL3x3hottesthitDEta_ = iConfig.getParameter<bool>("ApplyCut_maxleadTrackHCAL3x3hottesthitDEta");
         maxleadTrackHCAL3x3hottesthitDEta_          = iConfig.getParameter<double>("maxleadTrackHCAL3x3hottesthitDEta");
         ApplyCut_leadTrackavoidsECALcrack_          = iConfig.getParameter<bool>("ApplyCut_leadTrackavoidsECALcrack");
      }
      ~CaloRecoTauDiscriminationAgainstElectron(){} 
      double discriminate(const CaloTauRef& theCaloTauRef);
      void beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup);
   private:  
      edm::ESHandle<MagneticField> theMagneticField;
      edm::InputTag CaloTauProducer_;
      double leadTrack_HCAL3x3hitsEtSumOverPt_minvalue_;   
      bool ApplyCut_maxleadTrackHCAL3x3hottesthitDEta_;
      double maxleadTrackHCAL3x3hottesthitDEta_;
      bool ApplyCut_leadTrackavoidsECALcrack_;
};

void CaloRecoTauDiscriminationAgainstElectron::beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup)
{
   if (ApplyCut_leadTrackavoidsECALcrack_)
   {
      // get the magnetic field, if we need it
      eventSetup.get<IdealMagneticFieldRecord>().get(theMagneticField);
   }
}


double CaloRecoTauDiscriminationAgainstElectron::discriminate(const CaloTauRef& theCaloTauRef)
{
   if (ApplyCut_maxleadTrackHCAL3x3hottesthitDEta_){
      // optional selection : ask for small |deta| between direction of propag. leading Track - ECAL inner surf. contact point and direction of highest Et hit among HCAL hits inside a 3x3 calo. tower matrix centered on direction of propag. leading Track - ECAL inner surf. contact point
      if (edm::isNotFinite((*theCaloTauRef).leadTrackHCAL3x3hottesthitDEta()) || (*theCaloTauRef).leadTrackHCAL3x3hottesthitDEta()>maxleadTrackHCAL3x3hottesthitDEta_) return 0.;
   }
   if (ApplyCut_leadTrackavoidsECALcrack_){
      // optional selection : ask that leading track - ECAL inner surface contact point does not fall inside any ECAL eta crack 
      math::XYZPoint thepropagleadTrackECALSurfContactPoint = TauTagTools::propagTrackECALSurfContactPoint(theMagneticField.product(),(*theCaloTauRef).leadTrack());
      if(thepropagleadTrackECALSurfContactPoint.R()==0. ||
            fabs(thepropagleadTrackECALSurfContactPoint.eta())<ECALBounds::crack_absEtaIntervalA().second || 
            (fabs(thepropagleadTrackECALSurfContactPoint.eta())>ECALBounds::crack_absEtaIntervalB().first && fabs(thepropagleadTrackECALSurfContactPoint.eta())<ECALBounds::crack_absEtaIntervalB().second) ||
            (fabs(thepropagleadTrackECALSurfContactPoint.eta())>ECALBounds::crack_absEtaIntervalC().first && fabs(thepropagleadTrackECALSurfContactPoint.eta())<ECALBounds::crack_absEtaIntervalC().second) ||
            (fabs(thepropagleadTrackECALSurfContactPoint.eta())>ECALBounds::crack_absEtaIntervalD().first && fabs(thepropagleadTrackECALSurfContactPoint.eta())<ECALBounds::crack_absEtaIntervalD().second) ||
            (fabs(thepropagleadTrackECALSurfContactPoint.eta())>ECALBounds::crack_absEtaIntervalE().first && fabs(thepropagleadTrackECALSurfContactPoint.eta())<ECALBounds::crack_absEtaIntervalE().second))
      {
         return 0.;
      }     
   }
   if (edm::isNotFinite((*theCaloTauRef).leadTrackHCAL3x3hitsEtSum()))
   {
      return 0.;
   } else
   {
      if ((*theCaloTauRef).leadTrackHCAL3x3hitsEtSum()/(*theCaloTauRef).leadTrack()->pt()<=leadTrack_HCAL3x3hitsEtSumOverPt_minvalue_) return 0.;
      else return 1.;
   }
}

   /*
void CaloRecoTauDiscriminationAgainstElectron::produce(edm::Event& iEvent,const edm::EventSetup& iEventSetup){
  edm::Handle<CaloTauCollection> theCaloTauCollection;
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
      if (edm::isNotFinite((*theCaloTauRef).leadTrackHCAL3x3hottesthitDEta()) || (*theCaloTauRef).leadTrackHCAL3x3hottesthitDEta()>maxleadTrackHCAL3x3hottesthitDEta_){
	theCaloTauDiscriminatorAgainstElectron->setValue(iCaloTau,0);
	continue;
      }
    }
    if (ApplyCut_leadTrackavoidsECALcrack_){
      // optional selection : ask that leading track - ECAL inner surface contact point does not fall inside any ECAL eta crack 
      edm::ESHandle<MagneticField> theMagneticField;
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
    if (edm::isNotFinite((*theCaloTauRef).leadTrackHCAL3x3hitsEtSum())){
      theCaloTauDiscriminatorAgainstElectron->setValue(iCaloTau,0);
    }else{
      if ((*theCaloTauRef).leadTrackHCAL3x3hitsEtSum()/(*theCaloTauRef).leadTrack()->pt()<=leadTrack_HCAL3x3hitsEtSumOverPt_minvalue_) theCaloTauDiscriminatorAgainstElectron->setValue(iCaloTau,0);
      else theCaloTauDiscriminatorAgainstElectron->setValue(iCaloTau,1);
    }
  }
   
  iEvent.put(theCaloTauDiscriminatorAgainstElectron);
}
*/
DEFINE_FWK_MODULE(CaloRecoTauDiscriminationAgainstElectron);
