#include "RecoHI/HiEgammaAlgos/interface/HcalRechitIsoCalculator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"


using namespace edm;
using namespace reco;

HcalRechitIsoCalculator::HcalRechitIsoCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::Handle<HBHERecHitCollection> hbhe, const edm::Handle<HFRecHitCollection> hf, const edm::Handle<HORecHitCollection> ho)
{
  if(hf.isValid())
    fHFRecHits_ = hf.product();
  else
    fHFRecHits_ = nullptr;

  if(ho.isValid())
    fHORecHits_ = ho.product();
  else
    fHORecHits_ = nullptr;

  if(hbhe.isValid())
    fHBHERecHits_ = hbhe.product();
  else
    fHBHERecHits_ = nullptr;

  ESHandle<CaloGeometry> geometryHandle;
  iSetup.get<CaloGeometryRecord>().get(geometryHandle);
  if(geometryHandle.isValid())
    geometry_ = geometryHandle.product();
  else
    geometry_ = nullptr;

}


double HcalRechitIsoCalculator::getHcalRechitIso(const reco::SuperClusterRef cluster, const double x, const double threshold, const double innerR )
{
   if(!fHBHERecHits_) {
     return -100;
   }

   double TotalEt = 0;

   for(size_t index = 0; index < fHBHERecHits_->size(); index++) {
      const HBHERecHit &rechit = (*fHBHERecHits_)[index];
      const DetId &detid = rechit.id();
      const GlobalPoint& hitpoint = geometry_->getPosition(detid);
      double eta = hitpoint.eta();

      double dR2 = reco::deltaR2(*cluster, hitpoint);
      // veto inner cone///////////////
      if ( dR2 < innerR*innerR )  continue;
      /////////////////////////////////
      if (dR2< (x*x*0.01)) {
	 double et = rechit.energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      }
   }

   return TotalEt;
}

double HcalRechitIsoCalculator::getBkgSubHcalRechitIso(const reco::SuperClusterRef cluster, const double x, const double threshold, const double innerR )
{
   if(!fHBHERecHits_) {
      return -100;
   }

   double SClusterEta = cluster->eta();
   double TotalEt = 0;

   for(size_t index = 0; index < fHBHERecHits_->size(); index++) {
      const HBHERecHit &rechit = (*fHBHERecHits_)[index];
      const DetId &detid = rechit.id();
      const GlobalPoint& hitpoint = geometry_->getPosition(detid);
      double eta = hitpoint.eta();
      double dEta = fabs(eta-SClusterEta);

      if (dEta<x*0.1) {
         double et = rechit.energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      }
   }

   double Rx = getHcalRechitIso(cluster,x,threshold,innerR);
   double CRx = (Rx - TotalEt * (0.01*x*x - innerR*innerR) / (2 * 2 * 0.1 * x))*(1/(1-x/40.)) ;

   return CRx;
}
