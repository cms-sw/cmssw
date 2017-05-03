// ROOT includes
#include <Math/VectorUtil.h>

#include "HeavyIonsAnalysis/PhotonAnalysis/plugins/TxCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

using namespace edm;
using namespace reco;
using namespace std;
using namespace ROOT::Math::VectorUtil;


TxCalculator::TxCalculator (const edm::Event &iEvent, const edm::EventSetup &iSetup, edm::InputTag trackLabel,std::string trackQuality)
{
   iEvent.getByLabel(trackLabel, recCollection);
   edm::Service<edm::RandomNumberGenerator> rng;
   trackQuality_=trackQuality;
   if ( ! rng.isAvailable()) {
      throw cms::Exception("Configuration")
         << "XXXXXXX requires the RandomNumberGeneratorService\n"
         "which is not present in the configuration file.  You must add the service\n"
         "in the configuration file or remove the modules that require it.";
   }
   CLHEP::HepRandomEngine& engine = rng->getEngine(iEvent.streamID());
   theDice = new CLHEP::RandFlat(engine, 0, 1);

}


double TxCalculator::getJurassicArea( double r1, double r2, double width) {

   float theta1 = asin( width / r1);
   float theta2 = asin( width / r2);
   float theA   = sqrt ( r1*r1 + r2*r2 - 2 * r1 * r2 * cos ( theta1 - theta2) );
   float area1 =  0.5 * r1*r1 * ( 3.141592 - 2 * theta1 )   ;
   float area2 =  0.5 * r2*r2 * ( 3.141592 - 2 * theta2 )   ;
   float area3 =  width * theA;
   float finalArea = 2 * ( area1 - area2 - area3);
   return finalArea;
}


double TxCalculator::getMPT( double ptCut     ,   double etaCut  )
{
   using namespace edm;
   using namespace reco;

   double sumpx(0), sumpy(0);

   for(reco::TrackCollection::const_iterator
          recTrack = recCollection->begin(); recTrack!= recCollection->end(); recTrack++)
      {
      
      bool goodtrack = recTrack->quality(reco::TrackBase::qualityByName(trackQuality_));
	    if(!goodtrack) continue;

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


double TxCalculator::getTx(const reco::Photon cluster, double x, double threshold, double innerDR, double effRatio)
{

   using namespace edm;
   using namespace reco;



   double SClusterEta = cluster.eta();
   double SClusterPhi = cluster.phi();
   double TotalPt = 0;

   for(reco::TrackCollection::const_iterator
   	  recTrack = recCollection->begin(); recTrack!= recCollection->end(); recTrack++)
      {
	 double diceNum = theDice->fire();
	 if ( (effRatio < 1 ) &&  ( diceNum > effRatio))
	    continue;

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





double TxCalculator::getCTx(const reco::Photon cluster, double x, double threshold, double innerDR,double effRatio)
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
	 double diceNum = theDice->fire();
         if ( (effRatio < 1 ) &&  ( diceNum > effRatio))
            continue;


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

   double Tx = getTx(cluster,x,threshold,innerDR,effRatio);
   double CTx = Tx - TotalPt / 40.0 * x;

   return CTx;
}



double TxCalculator::getJt(const reco::Photon cluster, double r1, double r2, double jWidth, double threshold)
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
	 double eta = recTrack->eta();
         double phi = recTrack->phi();
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
	 // stupid bug if ( fabs(dPhi) >  jWidth)  continue;
	 //////////////////////

	 if(pt > threshold)
	    TotalPt = TotalPt + pt;
      }

   return TotalPt;
}


double TxCalculator::getJct(const reco::Photon cluster, double r1, double r2, double jWidth, double threshold)
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
         double eta = recTrack->eta();
         double phi = recTrack->phi();
         double dEta = fabs(eta-SClusterEta);
         double dPhi = phi-SClusterPhi;
         if ( dPhi < -PI )    dPhi = dPhi + 2*PI ;
         if ( dPhi >  PI )    dPhi = dPhi - 2*PI ;
	 if ( fabs(dPhi) >PI )   cout << " error!!! dphi > 2pi   : " << dPhi << endl;
	 //         double dR = sqrt(dEta*dEta+dPhi*dPhi);


	 //////// phi strip /////////
	 if ( fabs(dEta) > r1 ) continue;
	 if ( fabs(dPhi) <r1 ) continue;
	 //////////////////////

	 if(pt > threshold)
            TotalPt = TotalPt + pt;
      }

   double areaStrip = 4*PI*r1 -  4*r1*r1;
   double areaJura  = getJurassicArea(r1,r2, jWidth) ;
   double theCJ     = getJt(cluster,r1,  r2, jWidth, threshold);

   double theCCJ   = theCJ - TotalPt * areaJura / areaStrip ;
   return theCCJ;

}
