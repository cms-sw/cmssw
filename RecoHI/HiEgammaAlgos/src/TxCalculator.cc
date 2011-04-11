// ROOT includes
#include <Math/VectorUtil.h>

#include "RecoHI/HiEgammaAlgos/interface/TxCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/Math/interface/Vector3D.h"


using namespace edm;
using namespace reco;
using namespace std;
using namespace ROOT::Math::VectorUtil; 


TxCalculator::TxCalculator (const edm::Event &iEvent, const edm::EventSetup &iSetup, edm::InputTag trackLabel)
{
   iEvent.getByLabel(trackLabel, recCollection); 
} 



double TxCalculator::getMPT( double ptCut     ,   double etaCut  )
{
   using namespace edm;
   using namespace reco;
   
   double sumpx(0), sumpy(0);
   
   for(reco::TrackCollection::const_iterator
          recTrack = recCollection->begin(); recTrack!= recCollection->end(); recTrack++)
      {
	 double pt = recTrack->pt();
         double eta = recTrack->eta();
	 
	 if(pt < ptCut ) 
	    continue;
	 if ( fabs( eta) > etaCut ) 
	    continue;
	 
	 double pxTemp = recTrack->px();
	 double pyTemp = recTrack->py();
	 
	 sumpx = sumpx + pxTemp;
	 sumpy = sumpy + pyTemp;
	 //	 cout << " pt  = " << recTrack->pt() <<  "    and px = " << pxTemp << " and  py = " << pyTemp << endl;
      }
   //   cout << " square = " << sumpx*sumpx + sumpy*sumpy << endl;
   double theMPT = sqrt(sumpx*sumpx + sumpy*sumpy) ;
   //  cout << " mpt    = "<< theMPT << endl;
   
   return theMPT;
}


double TxCalculator::getTx(const reco::Photon cluster, double x, double threshold, double innerDR)
{

   using namespace edm;
   using namespace reco;


   double SClusterEta = cluster.eta();
   double SClusterPhi = cluster.phi();
   double TotalPt = 0;

   for(reco::TrackCollection::const_iterator
   	  recTrack = recCollection->begin(); recTrack!= recCollection->end(); recTrack++)
      {
	 double pt = recTrack->pt();
	 double eta2 = recTrack->eta();
	 double phi2 = recTrack->phi();
	 
      if(dRDistance(SClusterEta,SClusterPhi,eta2,phi2) >= 0.1 * x)
         continue;
      if(dRDistance(SClusterEta,SClusterPhi,eta2,phi2) < innerDR)
	 continue;
      if(pt > threshold)
         TotalPt = TotalPt + pt;
   }

   return TotalPt;
}

double TxCalculator::getCTx(const reco::Photon cluster, double x, double threshold, double innerDR)
{
   using namespace edm;
   using namespace reco;

   double SClusterEta = cluster.eta();
   double SClusterPhi = cluster.phi();
   double TotalPt = 0;

   TotalPt = 0;

   for(reco::TrackCollection::const_iterator
   	  recTrack = recCollection->begin(); recTrack!= recCollection->end(); recTrack++)
   {
      double pt = recTrack->pt();
      double eta2 = recTrack->eta();
      double phi2 = recTrack->phi();
      double dEta = fabs(eta2-SClusterEta);

      if(dEta >= 0.1 * x)
         continue;
      if(dRDistance(SClusterEta,SClusterPhi,eta2,phi2) < innerDR)
         continue;
      
      if(pt > threshold)
         TotalPt = TotalPt + pt;
   }
   
   double Tx = getTx(cluster,x,threshold);
   double CTx = Tx - TotalPt / 40.0 * x;

   return CTx;
}


