// ROOT includes
#include <Math/VectorUtil.h>

#include "RecoHI/HiEgammaAlgos/interface/EcalClusterIsoCalculator.h"
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

EcalClusterIsoCalculator::EcalClusterIsoCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::Handle<BasicClusterCollection> pEBclusters, const edm::Handle<BasicClusterCollection> pEEclusters)
{
  if(pEBclusters.isValid())
     fEBclusters_ = pEBclusters.product();
   else
     fEBclusters_ = nullptr;

   if(pEEclusters.isValid())
     fEEclusters_ = pEEclusters.product();
   else
     fEEclusters_ = nullptr;

   ESHandle<CaloGeometry> geometryHandle;
   iSetup.get<CaloGeometryRecord>().get(geometryHandle);
   if(geometryHandle.isValid())
     geometry_ = geometryHandle.product();
   else
     geometry_ = nullptr;
}

double EcalClusterIsoCalculator::getEcalClusterIso(const reco::SuperClusterRef cluster, const double x, const double threshold)
{
   if(!fEBclusters_) {
      return -100;
   }

   if(!fEEclusters_) {
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

      double dR2 = reco::deltaR2(*clu, *cluster);

      if (dR2 < (x*x*0.01)) {
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

      double dR2 = reco::deltaR2(*clu, *cluster);

      if (dR2 < (x*x*0.01)) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      }
   }

   return TotalEt;
}

double EcalClusterIsoCalculator::getBkgSubEcalClusterIso(const reco::SuperClusterRef cluster, const double x, double const threshold)
{
   if(!fEBclusters_) {
      return -100;
   }

   if(!fEEclusters_) {
      return -100;
   }

   double SClusterEta = cluster->eta();
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
      double dEta = fabs(eta-SClusterEta);

      if (dEta<x*0.1) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      }
   }

   double Cx = getEcalClusterIso(cluster,x,threshold);
   double CCx = (Cx - TotalEt / 40.0 * x) * (1/(1-x/40.));

   return CCx;
}
