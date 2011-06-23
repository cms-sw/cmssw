#include "RecoHI/HiEgammaAlgos/interface/RxCalculator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"


using namespace edm;
using namespace reco;

#define PI 3.141592653589793238462643383279502884197169399375105820974945


RxCalculator::RxCalculator (const edm::Event &iEvent, const edm::EventSetup &iSetup, edm::InputTag hbheLabel, edm::InputTag hfLabel, edm::InputTag hoLabel)
{
   Handle<HFRecHitCollection> hfhandle;
   iEvent.getByLabel(hfLabel, hfhandle);
   fHFRecHits_ = hfhandle.product();

   Handle<HORecHitCollection> hohandle;
   iEvent.getByLabel(hoLabel, hohandle);
   fHORecHits_ = hohandle.product();

   Handle<HBHERecHitCollection> hehbhandle;
   iEvent.getByLabel(hbheLabel, hehbhandle);
   fHBHERecHits_ = hehbhandle.product();

   ESHandle<CaloGeometry> geometryHandle;
   iSetup.get<CaloGeometryRecord>().get(geometryHandle);
   geometry_ = geometryHandle.product();

} 


double RxCalculator::getRx(const reco::SuperClusterRef cluster, double x, double threshold )
{
   using namespace edm;
   using namespace reco;


   if(!fHBHERecHits_) {       
      LogError("RxCalculator") << "Error! Can't get HBHERecHits for event.";
      return -100;
   }

   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();
   double TotalEt = 0;

   for(size_t index = 0; index < fHBHERecHits_->size(); index++) {
      const HBHERecHit &rechit = (*fHBHERecHits_)[index];
      const DetId &detid = rechit.id();
      const GlobalPoint& hitpoint = geometry_->getPosition(detid);
      double eta = hitpoint.eta();
      double phi = hitpoint.phi();
      double dEta = fabs(eta-SClusterEta);
      double dPhi = fabs(phi-SClusterPhi);
      while (dPhi>2*PI) dPhi-=2*PI;
      if (dPhi>PI) dPhi=2*PI-dPhi;

      if (dPhi>PI) dPhi=2*PI-dPhi;

      double dR = sqrt(dEta * dEta + dPhi * dPhi);

      if (dR<x*0.1) {
         double et = rechit.energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      }
   }

   return TotalEt;
}

double RxCalculator::getROx(const reco::SuperClusterRef cluster, double x,double threshold)
{
   using namespace edm;
   using namespace reco;

   if(!fHORecHits_) {       
      LogError("RxCalculator") << "Error! Can't get HORecHits for event.";
      return -100;
   }

   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();
   double TotalEt = 0;

   for(size_t index = 0; index < fHORecHits_->size(); index++) {
      const HORecHit &rechit = (*fHORecHits_)[index];
      const DetId &detid = rechit.id();
      const GlobalPoint& hitpoint = geometry_->getPosition(detid);
      double eta = hitpoint.eta();
      double phi = hitpoint.phi();
      double dEta = fabs(eta-SClusterEta);
      double dPhi = fabs(phi-SClusterPhi);
      while (dPhi>2*PI) dPhi-=2*PI;
      if (dPhi>PI) dPhi=2*PI-dPhi;

      double dR = sqrt(dEta * dEta + dPhi * dPhi);
      if (dR<x*0.1) {
         double et = rechit.energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      }
   }
   return TotalEt;
}

double RxCalculator::getRFx(const reco::SuperClusterRef cluster, double x, double threshold)
{
   using namespace edm;
   using namespace reco;

   if(!fHFRecHits_) {       
      LogError("RxCalculator") << "Error! Can't get HFRecHits for event.";
      return -100;
   }

   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();
   double TotalEt = 0;

   for(size_t index = 0; index < fHFRecHits_->size(); index++) {
      const HFRecHit &rechit = (*fHFRecHits_)[index];
      const DetId &detid = rechit.id();
      const GlobalPoint& hitpoint = geometry_->getPosition(detid);
      double eta = hitpoint.eta();
      double phi = hitpoint.phi();
      double dEta = fabs(eta-SClusterEta);
      double dPhi = fabs(phi-SClusterPhi);
      while (dPhi>2*PI) dPhi-=2*PI;
      if (dPhi>PI) dPhi=2*PI-dPhi;


      double dR = sqrt(dEta * dEta + dPhi * dPhi);
      if (dR<x*0.1) {
         double et = rechit.energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      }
   }



   return TotalEt;
}


double RxCalculator::getCRx(const reco::SuperClusterRef cluster, double x, double threshold )
{
   using namespace edm;
   using namespace reco;


   if(!fHBHERecHits_) {       
      LogError("RxCalculator") << "Error! Can't get HBHERecHits for event.";
      return -100;
   }

   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();
   double TotalEt = 0;

   for(size_t index = 0; index < fHBHERecHits_->size(); index++) {
      const HBHERecHit &rechit = (*fHBHERecHits_)[index];
      const DetId &detid = rechit.id();
      const GlobalPoint& hitpoint = geometry_->getPosition(detid);
      double eta = hitpoint.eta();
      double phi = hitpoint.phi();
      double dEta = fabs(eta-SClusterEta);
      double dPhi = fabs(phi-SClusterPhi);
      while (dPhi>2*PI) dPhi-=2*PI;
      if (dPhi>PI) dPhi=2*PI-dPhi;

      if (dEta<x*0.1) {
         double et = rechit.energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      }
   }

   double Rx = getRx(cluster,x,threshold);
   double CRx = Rx - TotalEt / 40.0 * x;

   return CRx;
}
