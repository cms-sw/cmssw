// ROOT includes
#include <Math/VectorUtil.h>

#include "HeavyIonsAnalysis/PhotonAnalysis/plugins/CxCalculator.h"
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


double CxCalculator::getJurassicArea( double r1, double r2, double width) {
   
   float theta1 = asin( width / r1);
   float theta2 = asin( width / r2);
   float theA   = sqrt ( r1*r1 + r2*r2 - 2 * r1 * r2 * cos ( theta1 - theta2) );
   float area1 =  0.5 * r1*r1 * ( 3.141592 - 2 * theta1 )   ;
   float area2 =  0.5 * r2*r2 * ( 3.141592 - 2 * theta2 )   ;
   float area3 =  width * theA;
   float finalArea = 2 * ( area1 - area2 - area3);
   return finalArea;
}



CxCalculator::CxCalculator (const edm::Event &iEvent, const edm::EventSetup &iSetup, edm::InputTag barrelLabel, edm::InputTag endcapLabel)
{
//InputTag("islandBasicClusters:islandBarrelBasicClusters")
//InputTag("islandBasicClusters:islandEndcapBasicClusters")
   Handle<BasicClusterCollection> pEBclusters;
   iEvent.getByLabel(barrelLabel, pEBclusters);
   if(pEBclusters.isValid())
     fEBclusters_ = pEBclusters.product(); 
   else
     fEBclusters_ = NULL;

   Handle<BasicClusterCollection> pEEclusters;
   iEvent.getByLabel(endcapLabel, pEEclusters);
   if(pEEclusters.isValid())
     fEEclusters_ = pEEclusters.product(); 
   else
     fEEclusters_ = NULL;

   ESHandle<CaloGeometry> geometryHandle;
   iSetup.get<CaloGeometryRecord>().get(geometryHandle);
   if(geometryHandle.isValid())
     geometry_ = geometryHandle.product();
   else
     geometry_ = NULL;

} 

double CxCalculator::getCx(const reco::SuperClusterRef cluster, double x, double threshold)
{
   using namespace edm;
   using namespace reco;

   if(!fEBclusters_) {       
//       LogError("CxCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
//       LogError("CxCalculator") << "Error! Can't get EEclusters for event.";
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
      //       LogError("CxCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }
   
   if(!fEEclusters_) {       
      //       LogError("CxCalculator") << "Error! Can't get EEclusters for event.";
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




double CxCalculator::getJc(const reco::SuperClusterRef cluster, double r1, double r2, double jWidth, double threshold)
{
   using namespace edm;
   using namespace reco;
   if(!fEBclusters_) {
      //       LogError("CxCalculator") << "Error! Can't get EBclusters for event.";                                                    
      return -100;
   }
   if(!fEEclusters_) {
      return -100;
   }
   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();
   double TotalEt = 0;
   
   for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
       iclu != fEBclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();
      double phi = ClusPoint.phi();
      
      double dEta = fabs(eta-SClusterEta);
      double dPhi = phi-SClusterPhi;
      if ( dPhi < -PI )    dPhi = dPhi + 2*PI ;
      if ( dPhi >  PI )    dPhi = dPhi - 2*PI ;
      if ( fabs(dPhi) > PI )   cout << " error!!! dphi > 2pi   : " << dPhi << endl;
      double dR = sqrt(dEta*dEta+dPhi*dPhi);
      
      // Jurassic Cone /////
      if ( dR > r1 ) continue;
      if ( dR < r2 ) continue;
      if ( fabs(dEta) <  jWidth)  continue;
      //////////////////////
      double theEt = clu->energy()/cosh(eta);
      if (theEt<threshold) continue;
      TotalEt += theEt;
   }
   
   for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
       iclu != fEEclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();
      double phi = ClusPoint.phi();
      double dEta = fabs(eta-SClusterEta);
      double dPhi = phi-SClusterPhi;
      if ( dPhi < -PI )    dPhi = dPhi + 2*PI ;
      if ( dPhi >  PI )    dPhi = dPhi - 2*PI ;
      if ( fabs(dPhi) >PI )   cout << " error!!! dphi > 2pi   : " << dPhi << endl;
      double dR = sqrt(dEta*dEta+dPhi*dPhi);
      // Jurassic Cone /////                                                                                                  
      if ( dR > r1 ) continue;
      if ( dR < r2 ) continue;
      if ( fabs(dEta) <  jWidth)  continue;
      //////////////////////          
      double theEt = clu->energy()/cosh(eta);
      if (theEt<threshold) continue;
      TotalEt += theEt;
   }
   return TotalEt;
}


double CxCalculator::getJcc(const reco::SuperClusterRef cluster, double r1, double r2, double jWidth, double threshold)
{
                   
   using namespace edm;
   using namespace reco;
   if(!fEBclusters_) {
      //       LogError("CxCalculator") << "Error! Can't get EBclusters for event.";                                                         
      return -100;
   }
   if(!fEEclusters_) {
      return -100;
   }
   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();
   double TotalEt = 0;

   for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
       iclu != fEBclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();
      double phi = ClusPoint.phi();

      double dEta = fabs(eta-SClusterEta);
      double dPhi = phi-SClusterPhi;
      if ( dPhi < -PI )    dPhi = dPhi + 2*PI ;
      if ( dPhi >  PI )    dPhi = dPhi - 2*PI ;
      //      double dR = sqrt(dEta*dEta+dPhi*dPhi);

      //////// phi strip /////////                                                                                               
      if ( fabs(dEta) > r1 ) continue;
      if ( fabs(dPhi) <r1 ) continue;
      //////////////////////                                                                                                      
      
      double theEt = clu->energy()/cosh(eta);
      if (theEt<threshold) continue;
      TotalEt += theEt;
   }
   for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
       iclu != fEEclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();
      double phi = ClusPoint.phi();
      double dEta = fabs(eta-SClusterEta);
      double dPhi = phi-SClusterPhi;
      if ( dPhi < -PI )    dPhi = dPhi + 2*PI ;
      if ( dPhi >  PI )    dPhi = dPhi - 2*PI ;
      //      double dR = sqrt(dEta*dEta+dPhi*dPhi);
      
      //////// phi strip /////////                                                                                       
      if ( fabs(dEta) > r1 ) continue;
      if ( fabs(dPhi) < r1 ) continue;
      //////////////////////  
      
      double theEt = clu->energy()/cosh(eta);
      if (theEt<threshold) continue;
      TotalEt += theEt;
   }
   
   double areaStrip = 4*PI*r1 -  4*r1*r1; 
   double areaJura  = getJurassicArea(r1,r2, jWidth) ;
   double theCJ     = getJc(cluster,r1,  r2, jWidth, threshold);
   //   cout << "areJura = " << areaJura << endl;
   //  cout << "areaStrip " << areaStrip << endl;
   double theCCJ   = theCJ - TotalEt * areaJura / areaStrip ;
   return theCCJ;
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
