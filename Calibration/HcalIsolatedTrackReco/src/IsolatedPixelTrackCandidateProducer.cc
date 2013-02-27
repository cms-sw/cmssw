#include <vector>
#include <memory>
#include <algorithm>

// Class header file
#include "Calibration/HcalIsolatedTrackReco/interface/IsolatedPixelTrackCandidateProducer.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
// Framework
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/Common/interface/TriggerResults.h"
// L1Extra
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
///

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
//#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "DataFormats/Math/interface/deltaR.h"

//vertices
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

//magF
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

//for ECAL geometry
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"


IsolatedPixelTrackCandidateProducer::IsolatedPixelTrackCandidateProducer(const edm::ParameterSet& config){
   
  l1eTauJetsSource_           = config.getParameter<edm::InputTag>("L1eTauJetsSource");
  tauAssocCone_               = config.getParameter<double>("tauAssociationCone"); 
  tauUnbiasCone_              = config.getParameter<double>("tauUnbiasCone");
  pixelTracksSources_         = config.getParameter<std::vector<edm::InputTag> >("PixelTracksSources");
  prelimCone_                 = config.getParameter<double>("ExtrapolationConeSize");
  pixelIsolationConeSizeAtEC_ = config.getParameter<double>("PixelIsolationConeSizeAtEC");
  hltGTseedlabel_             = config.getParameter<edm::InputTag>("L1GTSeedLabel");
  vtxCutSeed_                 = config.getParameter<double>("MaxVtxDXYSeed");
  vtxCutIsol_                 = config.getParameter<double>("MaxVtxDXYIsol");
  vertexLabel_                = config.getParameter<edm::InputTag>("VertexLabel");
  bfield_                     = config.getParameter<std::string>("MagFieldRecordName");
  minPTrackValue_             = config.getParameter<double>("minPTrack");
  maxPForIsolationValue_      = config.getParameter<double>("maxPTrackForIsolation");
  ebEtaBoundary_              = config.getParameter<double>("EBEtaBoundary");
  rEB_ = zEE_ = -1;

  // Register the product
  produces< reco::IsolatedPixelTrackCandidateCollection >();
}

IsolatedPixelTrackCandidateProducer::~IsolatedPixelTrackCandidateProducer() {

}

void IsolatedPixelTrackCandidateProducer::beginRun(const edm::Run &run, const edm::EventSetup &theEventSetup)
{

  edm::ESHandle<CaloGeometry> pG;
  theEventSetup.get<CaloGeometryRecord>().get(pG);   
  
  const double rad (dynamic_cast<const EcalBarrelGeometry*>( pG->getSubdetectorGeometry(DetId::Ecal, EcalBarrel ))->avgRadiusXYFrontFaceCenter() ) ;
  
  const double zz (dynamic_cast<const EcalEndcapGeometry*>( pG->getSubdetectorGeometry(DetId::Ecal, EcalEndcap ))->avgAbsZFrontFaceCenter() ) ;

  rEB_=rad;
  zEE_=zz;

}

void IsolatedPixelTrackCandidateProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  reco::IsolatedPixelTrackCandidateCollection* trackCollection=new reco::IsolatedPixelTrackCandidateCollection;

  //create vector of refs from input collections
  std::vector<reco::TrackRef> pixelTrackRefs;

  for (unsigned int iPix=0; iPix<pixelTracksSources_.size(); iPix++)
    {
      edm::Handle<reco::TrackCollection> iPixCol;
      theEvent.getByLabel(pixelTracksSources_[iPix],iPixCol);
      for (reco::TrackCollection::const_iterator pit=iPixCol->begin(); pit!=iPixCol->end(); pit++)
        {
          pixelTrackRefs.push_back(reco::TrackRef(iPixCol,pit-iPixCol->begin()));
        }
    }

  edm::Handle<l1extra::L1JetParticleCollection> l1eTauJets;
  theEvent.getByLabel(l1eTauJetsSource_,l1eTauJets);

  edm::Handle<reco::VertexCollection> pVert;
  theEvent.getByLabel(vertexLabel_,pVert);

  double ptTriggered  = -10;
  double etaTriggered = -100;
  double phiTriggered = -100;
  
  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1trigobj;
  theEvent.getByLabel(hltGTseedlabel_, l1trigobj);
  
  std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1tauobjref;
  std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1jetobjref;
  std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1forjetobjref;
  
  l1trigobj->getObjects(trigger::TriggerL1TauJet, l1tauobjref);
  l1trigobj->getObjects(trigger::TriggerL1CenJet, l1jetobjref);
  l1trigobj->getObjects(trigger::TriggerL1ForJet, l1forjetobjref);
  
  for (unsigned int p=0; p<l1tauobjref.size(); p++)
    {
      if (l1tauobjref[p]->pt()>ptTriggered)
	{
	  ptTriggered  = l1tauobjref[p]->pt(); 
	  phiTriggered = l1tauobjref[p]->phi();
	  etaTriggered = l1tauobjref[p]->eta();
	}
    }
  for (unsigned int p=0; p<l1jetobjref.size(); p++)
    {
      if (l1jetobjref[p]->pt()>ptTriggered)
	{
	  ptTriggered  = l1jetobjref[p]->pt();
	  phiTriggered = l1jetobjref[p]->phi();
	  etaTriggered = l1jetobjref[p]->eta();
	}
    }
  for (unsigned int p=0; p<l1forjetobjref.size(); p++)
    {
      if (l1forjetobjref[p]->pt()>ptTriggered)
        {
          ptTriggered=l1forjetobjref[p]->pt();
          phiTriggered=l1forjetobjref[p]->phi();
          etaTriggered=l1forjetobjref[p]->eta();
        }
    }

  double minPTrack_    = minPTrackValue_;
  double drMaxL1Track_ = tauAssocCone_;
  
  int ntr = 0;
  
  //loop to select isolated tracks
  for (unsigned iSeed=0; iSeed<pixelTrackRefs.size(); iSeed++)
    {
      if(pixelTrackRefs[iSeed]->p()<minPTrack_) continue;

      bool good     = false;
      bool vtxMatch = false;

      //associate to vertex (in Z) 
      reco::VertexCollection::const_iterator vitSel;
      double minDZ = 100;
      for (reco::VertexCollection::const_iterator vit=pVert->begin(); vit!=pVert->end(); vit++)
	{
	  if (fabs(pixelTrackRefs[iSeed]->dz(vit->position()))<minDZ)
	    {
	      minDZ  = fabs(pixelTrackRefs[iSeed]->dz(vit->position()));
	      vitSel = vit;
	    }
	}
      //cut on dYX:
      if (minDZ!=100&&fabs(pixelTrackRefs[iSeed]->dxy(vitSel->position()))<vtxCutSeed_) vtxMatch=true;
      if (minDZ==100) vtxMatch=true;

      //select tracks not matched to triggered L1 jet
      double R=deltaR(etaTriggered, phiTriggered, pixelTrackRefs[iSeed]->eta(), pixelTrackRefs[iSeed]->phi());
      if (R<tauUnbiasCone_) continue;
      
      //check taujet matching
      bool tmatch=false;
      l1extra::L1JetParticleCollection::const_iterator selj;
      for (l1extra::L1JetParticleCollection::const_iterator tj=l1eTauJets->begin(); tj!=l1eTauJets->end(); tj++) 
	{
	  if(ROOT::Math::VectorUtil::DeltaR(pixelTrackRefs[iSeed]->momentum(),tj->momentum()) > drMaxL1Track_) continue;
	  selj   = tj;
	  tmatch = true;
	} //loop over L1 tau
      
      //propagate seed track to ECAL surface:
      std::pair<double,double> seedCooAtEC;
      // in case vertex is found:
      if (minDZ!=100) seedCooAtEC=GetEtaPhiAtEcal(theEventSetup, pixelTrackRefs[iSeed]->eta(), pixelTrackRefs[iSeed]->phi(), pixelTrackRefs[iSeed]->pt(), pixelTrackRefs[iSeed]->charge(), vitSel->z());
      //in case vertex is not found:
      else seedCooAtEC=GetEtaPhiAtEcal(theEventSetup, pixelTrackRefs[iSeed]->eta(), pixelTrackRefs[iSeed]->phi(), pixelTrackRefs[iSeed]->pt(), pixelTrackRefs[iSeed]->charge(), 0);

      //calculate isolation
      double maxP = 0;
      double sumP = 0;
      for (unsigned iSurr=0; iSurr<pixelTrackRefs.size(); iSurr++)
        {
	  if(iSeed==iSurr) continue;
	  //define preliminary cone around seed track impact point from which tracks will be extrapolated:
          if (deltaR(seedCooAtEC.first, seedCooAtEC.second, pixelTrackRefs[iSurr]->eta(), pixelTrackRefs[iSurr]->phi())>prelimCone_) continue;
	  //associate to vertex (in Z):
	  double minDZ2=100;
	  reco::VertexCollection::const_iterator vitSel2;
	  for (reco::VertexCollection::const_iterator vit=pVert->begin(); vit!=pVert->end(); vit++)
	    {
	      if (fabs(pixelTrackRefs[iSurr]->dz(vit->position()))<minDZ2)
		{
		  minDZ2  = fabs(pixelTrackRefs[iSurr]->dz(vit->position()));
		  vitSel2 = vit;
		}
	    }
	  //cut ot dXY:
	  if (minDZ2!=100&&fabs(pixelTrackRefs[iSurr]->dxy(vitSel2->position()))>vtxCutIsol_) continue;
	  //propagate to ECAL surface:
	  std::pair<double,double> cooAtEC;
	  // in case vertex is found:
	  if (minDZ2!=100) cooAtEC=GetEtaPhiAtEcal(theEventSetup, pixelTrackRefs[iSurr]->eta(), pixelTrackRefs[iSurr]->phi(), pixelTrackRefs[iSurr]->pt(), pixelTrackRefs[iSurr]->charge(), vitSel2->z());
	  // in case vertex is not found:
	  else cooAtEC=GetEtaPhiAtEcal(theEventSetup, pixelTrackRefs[iSurr]->eta(), pixelTrackRefs[iSurr]->phi(), pixelTrackRefs[iSurr]->pt(), pixelTrackRefs[iSurr]->charge(), 0);
	  
	  //calculate distance at ECAL surface and update isolation: 
	  if (getDistInCM(seedCooAtEC.first, seedCooAtEC.second, cooAtEC.first, cooAtEC.second)<pixelIsolationConeSizeAtEC_)
	    {
	      sumP+=pixelTrackRefs[iSurr]->p();
	      if(pixelTrackRefs[iSurr]->p()>maxP) maxP=pixelTrackRefs[iSurr]->p();
	    }
	}

      if (tmatch||vtxMatch) good=true;

      if (good&&maxP<maxPForIsolationValue_)
	{
	  reco::IsolatedPixelTrackCandidate newCandidate(pixelTrackRefs[iSeed], l1extra::L1JetParticleRef(l1eTauJets,selj-l1eTauJets->begin()), maxP, sumP);
	  trackCollection->push_back(newCandidate);
	  ntr++;
	}
    }//loop over pixel tracks

  // put the product in the event
  std::auto_ptr< reco::IsolatedPixelTrackCandidateCollection > outCollection(trackCollection);
  theEvent.put(outCollection);

}


double IsolatedPixelTrackCandidateProducer::getDistInCM(double eta1, double phi1, double eta2, double phi2)
{
  double Rec;
  double theta1=2*atan(exp(-eta1));
  double theta2=2*atan(exp(-eta2));
  if (fabs(eta1)<1.479) Rec=129; //radius of ECAL barrel
  else if (fabs(eta1)>1.479&&fabs(eta1)<7.0) Rec=tan(theta1)*317; //distance from IP to ECAL endcap
  else return 1000;

  //|vect| times tg of acos(scalar product)
  double angle=acos((sin(theta1)*sin(theta2)*(sin(phi1)*sin(phi2)+cos(phi1)*cos(phi2))+cos(theta1)*cos(theta2)));
  if (angle<acos(-1)/2) return fabs((Rec/sin(theta1))*tan(angle));
  else return 1000;
}


std::pair<double,double>
IsolatedPixelTrackCandidateProducer::GetEtaPhiAtEcal(const edm::EventSetup& iSetup, double etaIP, double phiIP, double pT, int charge, double vtxZ)
{
  edm::ESHandle<MagneticField> vbfField;
  iSetup.get<IdealMagneticFieldRecord>().get(vbfField);
  const VolumeBasedMagneticField* vbfCPtr = dynamic_cast<const VolumeBasedMagneticField*>(&(*vbfField));
  GlobalVector BField=vbfCPtr->inTesla(GlobalPoint(0,0,0));
 //test
 //int curvSgn=int(BField.z()/fabs(BField.z())); 

  double bfVal=BField.mag();

  double deltaPhi=0;
  double etaEC = 100;
  double phiEC = 100;

  double Rcurv = 9999999;
  if (bfVal!=0) Rcurv=pT*33.3*100/(bfVal*10); //r(m)=pT(GeV)*33.3/B(kG)

  double ecDist = zEE_;  //distance to ECAL andcap from IP (cm), 317 - ecal (not preshower), preshower -300
  double ecRad  = rEB_;  //radius of ECAL barrel (cm)
  double theta=2*atan(exp(-etaIP));
  double zNew=0;
  if (theta>0.5*acos(-1)) theta=acos(-1)-theta;
  if (fabs(etaIP)<ebEtaBoundary_)
    {
      if ((0.5*ecRad/Rcurv)>1)
	{
	  etaEC=10000;
	  deltaPhi=0;
	}
      else
	{
	  deltaPhi      =-charge*asin(0.5*ecRad/Rcurv);
	  double alpha1 = 2*asin(0.5*ecRad/Rcurv);
	  double z      = ecRad/tan(theta);
	  if (etaIP>0) zNew = z*(Rcurv*alpha1)/ecRad+vtxZ; //new z-coordinate of track
	  else         zNew =-z*(Rcurv*alpha1)/ecRad+vtxZ; //new z-coordinate of track
	  double zAbs=fabs(zNew);
	  if (zAbs<ecDist)
	    {
	      etaEC    = -log(tan(0.5*atan(ecRad/zAbs)));
	      deltaPhi = -charge*asin(0.5*ecRad/Rcurv);
	    }
	  if (zAbs>ecDist)
	    {
	      zAbs           = (fabs(etaIP)/etaIP)*ecDist;
	      double Zflight = fabs(zAbs-vtxZ);
	      double alpha   = (Zflight*ecRad)/(z*Rcurv);
	      double Rec     = 2*Rcurv*sin(alpha/2);
	      deltaPhi       =-charge*alpha/2;
	      etaEC          =-log(tan(0.5*atan(Rec/ecDist)));
	    }
	}
    }
  else
    {
      zNew           = (fabs(etaIP)/etaIP)*ecDist;
      double Zflight = fabs(zNew-vtxZ);
      double Rvirt   = fabs(Zflight*tan(theta));
      double Rec     = 2*Rcurv*sin(Rvirt/(2*Rcurv));
      deltaPhi       =-(charge)*(Rvirt/(2*Rcurv));
      etaEC          =-log(tan(0.5*atan(Rec/ecDist)));
    }

  if (zNew<0) etaEC=-etaEC;
  phiEC            = phiIP+deltaPhi;

  if (phiEC<-acos(-1)) phiEC = 2*acos(-1)+phiEC;
  if (phiEC>acos(-1))  phiEC =-2*acos(-1)+phiEC;

  std::pair<double,double> retVal(etaEC,phiEC);
  return retVal;
}


