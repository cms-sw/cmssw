// ROOT includes
#include <Math/VectorUtil.h>

#include "RecoHI/HiEgammaAlgos/interface/CxCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Math/interface/Vector3D.h"


using namespace edm;
using namespace reco;
using namespace std;
using namespace ROOT::Math::VectorUtil; 

#define PI 3.141592653589793238462643383279502884197169399375105820974945

CxCalculator::CxCalculator (const edm::Event &iEvent, const edm::EventSetup &iSetup, edm::InputTag barrelLabel, edm::InputTag endcapLabel)
{
//InputTag("islandBasicClusters:islandBarrelBasicClusters")
//InputTag("islandBasicClusters:islandEndcapBasicClusters")
   Handle<BasicClusterCollection> pEBclusters;
   iEvent.getByLabel(barrelLabel, pEBclusters);
   fEBclusters_ = pEBclusters.product(); 

   Handle<BasicClusterCollection> pEEclusters;
   iEvent.getByLabel(endcapLabel, pEEclusters);
   fEEclusters_ = pEEclusters.product(); 

   ESHandle<CaloGeometry> geometryHandle;
   iSetup.get<CaloGeometryRecord>().get(geometryHandle);

   geometry_ = geometryHandle.product();
} 

double CxCalculator::getCx(const reco::SuperClusterRef cluster, double x, double threshold)
{
   using namespace edm;
   using namespace reco;

   if(!fEBclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   math::XYZVector SClusPoint(cluster->position().x(),
                          cluster->position().y(),
                          cluster->position().z());

   double TotalEt = 0;

   TotalEt = - cluster->rawEnergy()/cosh(cluster->eta());

   // Loop over barrel basic clusters 
   for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
       iclu != fEBclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();

      double dR = ROOT::Math::VectorUtil::DeltaR(ClusPoint,SClusPoint);
      
      if (dR<x*0.1) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
       iclu != fEEclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      const GlobalPoint clusPoint(clu->x(),clu->y(),clu->z());
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();

      double dR = ROOT::Math::VectorUtil::DeltaR(ClusPoint,SClusPoint);

      if (dR<x*0.1) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   return TotalEt;
}

double CxCalculator::getCxRemoveSC(const reco::SuperClusterRef cluster, double x, double threshold)
{
   // Calculate Cx and remove the basicClusters used by superCluster

   using namespace edm;
   using namespace reco;

   if(!fEBclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   math::XYZVector SClusPoint(cluster->position().x(),
                          cluster->position().y(),
                          cluster->position().z());

   double TotalEt = 0;

   TotalEt = 0;

   // Loop over barrel basic clusters 
   for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
       iclu != fEBclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();

      double dR = ROOT::Math::VectorUtil::DeltaR(ClusPoint,SClusPoint);
      
      // check if this basic cluster is used in the target supercluster
      bool inSuperCluster = checkUsed(cluster,clu);

      if (dR<x*0.1&&inSuperCluster==false) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
     
   }

   for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
       iclu != fEEclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      const GlobalPoint clusPoint(clu->x(),clu->y(),clu->z());
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();

      double dR = ROOT::Math::VectorUtil::DeltaR(ClusPoint,SClusPoint);

      // check if this basic cluster is used in the target supercluster
      bool inSuperCluster = checkUsed(cluster,clu);

      if (dR<x*0.1&&inSuperCluster==false) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   return TotalEt;
}

double CxCalculator::getCCx(const reco::SuperClusterRef cluster, double x, double threshold)
{
   using namespace edm;
   using namespace reco;


   if(!fEBclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();
   double TotalEt = 0;

   TotalEt = - cluster->rawEnergy()/cosh(cluster->eta());

   for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
       iclu != fEBclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();

      double dEta = fabs(eta-SClusterEta);
 
     if (dEta<x*0.1) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
       iclu != fEEclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();
      double phi = ClusPoint.phi();

      double dEta = fabs(eta-SClusterEta);
      double dPhi = fabs(phi-SClusterPhi);
      while (dPhi>2*PI) dPhi-=2*PI;
      if (dPhi>PI) dPhi=2*PI-dPhi;

      if (dEta<x*0.1) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   double Cx = getCx(cluster,x,threshold);
   double CCx = Cx - TotalEt / 40.0 * x;

   return CCx;
}


double CxCalculator::getCCxRemoveSC(const reco::SuperClusterRef cluster, double x, double threshold)
{
   using namespace edm;
   using namespace reco;


   if(!fEBclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();
   double TotalEt = 0;

   TotalEt = 0;

   for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
       iclu != fEBclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();

      double dEta = fabs(eta-SClusterEta);

      // check if this basic cluster is used in the target supercluster
      bool inSuperCluster = checkUsed(cluster,clu);
 
     if (dEta<x*0.1&&inSuperCluster==false) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
       iclu != fEEclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();
      double phi = ClusPoint.phi();

      double dEta = fabs(eta-SClusterEta);
      double dPhi = fabs(phi-SClusterPhi);
      while (dPhi>2*PI) dPhi-=2*PI;
      if (dPhi>PI) dPhi=2*PI-dPhi;

      // check if this basic cluster is used in the target supercluster
      bool inSuperCluster = checkUsed(cluster,clu);

      if (dEta<x*0.1&&inSuperCluster==false) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   double Cx = getCxRemoveSC(cluster,x,threshold);
   double CCx = Cx - TotalEt / 40.0 * x;

   return CCx;
}


bool CxCalculator::checkUsed(const reco::SuperClusterRef sc, const reco::BasicCluster* bc)
{
   reco::CaloCluster_iterator theEclust = sc->clustersBegin();

   // Loop over the basicClusters inside the target superCluster
   for(;theEclust != sc->clustersEnd(); theEclust++) {
     if ((**theEclust) == (*bc) ) return  true; //matched, so it's used.
   }
   return false;
}

double CxCalculator::getBCMax(const reco::SuperClusterRef cluster,int i)
{
   reco::CaloCluster_iterator theEclust = cluster->clustersBegin();

   double energyMax=0,energySecond=0;
   // Loop over the basicClusters inside the target superCluster
   for(;theEclust != cluster->clustersEnd(); theEclust++) {
     if ((*theEclust)->energy()>energyMax ) {
        energySecond=energyMax;
        energyMax=(*theEclust)->energy();
     } else if ((*theEclust)->energy()>energySecond) {
        energySecond=(*theEclust)->energy();
     }
   }
   if (i==1) return energyMax;
   return energySecond;
}


double CxCalculator::getCorrection(const reco::SuperClusterRef cluster, double x, double y,double threshold)
{
   using namespace edm;
   using namespace reco;

   // doesn't really work now ^^; (Yen-Jie)
   if(!fEBclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();
   double TotalEnergy = 0;
   double TotalBC = 0;

   TotalEnergy = 0;

   double Area = PI * (-x*x+y*y) / 100.0;
   double nCrystal = Area / 0.0174 / 0.0174; // ignore the difference between endcap and barrel for the moment....

   for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
       iclu != fEBclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      const GlobalPoint clusPoint(clu->x(),clu->y(),clu->z());
      double eta = clusPoint.eta();
      double phi = clusPoint.phi();
      double dEta = fabs(eta-SClusterEta);
      double dPhi = fabs(phi-SClusterPhi);
      while (dPhi>2*PI) dPhi-=2*PI;
      if (dPhi>PI) dPhi=2*PI-dPhi;
      double dR = sqrt(dEta*dEta+dPhi*dPhi);
 
     if (dR>x*0.1&&dR<y*0.1) {
         double e = clu->energy();
         if (e<threshold) e=0;
         TotalEnergy += e;
         if (e!=0) TotalBC+=clu->size();  // number of crystals
   
      } 
   }

   for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
       iclu != fEEclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      const GlobalPoint clusPoint(clu->x(),clu->y(),clu->z());
      double eta = clusPoint.eta();
      double phi = clusPoint.phi();
      double dEta = fabs(eta-SClusterEta);
      double dPhi = fabs(phi-SClusterPhi);
      while (dPhi>2*PI) dPhi-=2*PI;
      if (dPhi>PI) dPhi=2*PI-dPhi;
      double dR = sqrt(dEta*dEta+dPhi*dPhi);
 
     if (dR>x*0.1&&dR<y*0.1) {
         double e = clu->energy();
         if (e<threshold) e=0;
         TotalEnergy += e;
         if (e!=0) TotalBC += clu->size(); // number of crystals
      } 
   }


  if (TotalBC==0) return 0;
  return TotalEnergy/nCrystal;
}

double CxCalculator::getAvgBCEt(const reco::SuperClusterRef cluster, double x,double phi1,double phi2, double threshold)
// x: eta cut, phi1: deltaPhiMin cut, phi2: deltaPhiMax
{
   using namespace edm;
   using namespace reco;


   if(!fEBclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();

   double TotalEt = 0;    // Total E
   double TotalN = 0;     // Total N

   TotalEt = - cluster->rawEnergy()/cosh(cluster->eta());

   if (fabs(SClusterEta) < 1.479) {
      //Barrel    
      for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
          iclu != fEBclusters_->end(); ++iclu) {
         const BasicCluster *clu = &(*iclu);
         math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
         double eta = ClusPoint.eta();
         double phi = ClusPoint.phi();  

         double dEta = fabs(eta-SClusterEta);
         double dPhi = fabs(phi-SClusterPhi);
         while (dPhi>2*PI) dPhi-=2*PI;

         bool inSuperCluster = checkUsed(cluster,clu);

         if (dEta<x*0.1&&inSuperCluster==false&&dPhi>phi1*0.1&&dPhi<phi2*0.1) {
            double et = clu->energy()/cosh(eta);
            if (et<threshold) et=0;
            TotalEt += et;
            TotalN ++;
         } 
      }   
   } else {
      //Endcap
      for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
          iclu != fEEclusters_->end(); ++iclu) {
         const BasicCluster *clu = &(*iclu);
         math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
         double eta = ClusPoint.eta();
         double phi = ClusPoint.phi();  
   
         double dEta = fabs(eta-SClusterEta);
         double dPhi = fabs(phi-SClusterPhi);
         while (dPhi>2*PI) dPhi-=2*PI;

         bool inSuperCluster = checkUsed(cluster,clu);

         if (dEta<x*0.1&&inSuperCluster==false&&dPhi>phi1*0.1&&dPhi<phi2*0.1) {
            double et = clu->energy()/cosh(eta);
            if (et<threshold) et=0;
            TotalEt += et;
            TotalN ++;
         } 
      }
   }
   return TotalEt / TotalN;
}

double CxCalculator::getNBC(const reco::SuperClusterRef cluster, double x,double phi1,double phi2, double threshold)
// x: eta cut, phi1: deltaPhiMin cut, phi2: deltaPhiMax
{
   using namespace edm;
   using namespace reco;


   if(!fEBclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();

   double TotalEt = 0;    // Total E
   double TotalN = 0;     // Total N

   TotalEt = - cluster->rawEnergy()/cosh(cluster->eta());

   

   if (fabs(SClusterEta) < 1.479) {
      //Barrel    
      for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
          iclu != fEBclusters_->end(); ++iclu) {
         const BasicCluster *clu = &(*iclu);
         math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
         double eta = ClusPoint.eta();
         double phi = ClusPoint.phi();  

         double dEta = fabs(eta-SClusterEta);
         double dPhi = fabs(phi-SClusterPhi);
         while (dPhi>2*PI) dPhi-=2*PI;

         bool inSuperCluster = checkUsed(cluster,clu);

         if (dEta<x*0.1&&inSuperCluster==false&&dPhi>phi1*0.1&&dPhi<phi2*0.1) {
            double et = clu->energy()/cosh(eta);
            if (et<threshold) et=0;
            TotalEt += et;
            TotalN ++;
         } 
      }   
   } else {
      //Endcap
      for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
          iclu != fEEclusters_->end(); ++iclu) {
         const BasicCluster *clu = &(*iclu);
         math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
         double eta = ClusPoint.eta();
         double phi = ClusPoint.phi();  
   
         double dEta = fabs(eta-SClusterEta);
         double dPhi = fabs(phi-SClusterPhi);
         while (dPhi>2*PI) dPhi-=2*PI;

         bool inSuperCluster = checkUsed(cluster,clu);

         if (dEta<x*0.1&&inSuperCluster==false&&dPhi>phi1*0.1&&dPhi<phi2*0.1) {
            double et = clu->energy()/cosh(eta);
            if (et<threshold) et=0;
            TotalEt += et;
            TotalN ++;
         } 
      }
   }
   return TotalN;
}
