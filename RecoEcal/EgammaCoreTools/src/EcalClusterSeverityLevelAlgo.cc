#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterSeverityLevelAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

float EcalClusterSeverityLevelAlgo::goodFraction( const reco::CaloCluster & cluster, 
						  const EcalRecHitCollection & recHits, const EcalSeverityLevelAlgo& sevlv) 
{
        float fraction = 0.;
        std::vector< std::pair<DetId, float> > hitsAndFracs = cluster.hitsAndFractions();
        std::vector< std::pair<DetId, float> >::const_iterator it;
        for ( it = hitsAndFracs.begin(); it != hitsAndFracs.end(); ++it ) {
                DetId id = (*it).first;
                EcalRecHitCollection::const_iterator jrh = recHits.find( id );
                if ( jrh == recHits.end() ) {
                        edm::LogError("EcalClusterSeverityLevelAlgo") << "The cluster DetId " << id.rawId() << " is not in the recHit collection!!";
                        return -1;
                }
		      
                uint32_t sev = sevlv.severityLevel( id, recHits);
		//                if ( sev == EcalSeverityLevelAlgo::kBad ) ++recoveryFailed;
                if ( sev == EcalSeverityLevel::kProblematic 
                     || sev == EcalSeverityLevel::kRecovered || sev == EcalSeverityLevel::kBad ) 
		  {
// 		    std::cout << "[goodFraction] Found a problematic channel " << EBDetId(id) << " " << flag << " energy: " <<  (*jrh).energy() << std::endl;
		    fraction += (*jrh).energy() * (*it).second / cluster.energy();
		  }
        }
        return 1. - fraction;
}

float EcalClusterSeverityLevelAlgo::fractionAroundClosestProblematic( const reco::CaloCluster & cluster, 
								      const EcalRecHitCollection & recHits, const CaloTopology* topology, const EcalSeverityLevelAlgo& sevlv )
{
  DetId closestProb = closestProblematic(cluster , recHits, topology, sevlv);
  //  std::cout << "%%%%%%%%%%% Closest prob is " << EBDetId(closestProb) << std::endl;
  if (closestProb.null())
    return 0.;
  
  std::vector<DetId> neighbours = topology->getWindow(closestProb,3,3);
  std::vector<DetId>::const_iterator itn;

  std::vector< std::pair<DetId, float> > hitsAndFracs = cluster.hitsAndFractions();
  std::vector< std::pair<DetId, float> >::const_iterator it;

  float fraction = 0.;  

  for ( itn = neighbours.begin(); itn != neighbours.end(); ++itn )
    {
      //      std::cout << "Checking detId " << EBDetId((*itn)) << std::endl;
      for ( it = hitsAndFracs.begin(); it != hitsAndFracs.end(); ++it ) 
	{
	  DetId id = (*it).first;
	  if ( id != (*itn) )
	    continue;
	  //	  std::cout << "Is in cluster detId " << EBDetId(id) << std::endl;
	  EcalRecHitCollection::const_iterator jrh = recHits.find( id );
	  if ( jrh == recHits.end() ) 
	    {
	      edm::LogError("EcalClusterSeverityLevelAlgo") << "The cluster DetId " << id.rawId() << " is not in the recHit collection!!";
	      return -1;
	    }

	  fraction += (*jrh).energy() * (*it).second  / cluster.energy();
	}
    }
  //  std::cout << "%%%%%%%%%%% Fraction is " << fraction << std::endl;
  return fraction;
}

DetId EcalClusterSeverityLevelAlgo::closestProblematic(const reco::CaloCluster & cluster, 
						     const EcalRecHitCollection & recHits, 
						       const CaloTopology* topology ,  const EcalSeverityLevelAlgo& sevlv)
{
  DetId seed=EcalClusterTools::getMaximum(cluster,&recHits).first;
  if ( (seed.det() != DetId::Ecal) || 
       (EcalSubdetector(seed.subdetId()) != EcalBarrel) )
    {
      //method not supported if not in Barrel
      edm::LogError("EcalClusterSeverityLevelAlgo") << "The cluster seed is not in the BARREL";
      return DetId(0);
    }

  int minDist=9999; DetId closestProb(0);   
  //Get a window of DetId around the seed crystal
  std::vector<DetId> neighbours = topology->getWindow(seed,51,11);

  for ( std::vector<DetId>::const_iterator it = neighbours.begin(); it != neighbours.end(); ++it ) 
    {
      EcalRecHitCollection::const_iterator jrh = recHits.find(*it);
      if ( jrh == recHits.end() ) 
	continue;
      //Now checking rh flag   
      uint32_t sev = sevlv.severityLevel( *it, recHits);
      if (sev == EcalSeverityLevel::kGood)
	continue;
      //      std::cout << "[closestProblematic] Found a problematic channel " << EBDetId(*it) << " " << flag << std::endl;
      //Find the closest DetId in eta,phi space (distance defined by deta^2 + dphi^2)
      int deta=EBDetId::distanceEta(EBDetId(seed),EBDetId(*it));
      int dphi=EBDetId::distancePhi(EBDetId(seed),EBDetId(*it));
      double r = sqrt(deta*deta + dphi*dphi);
      if (r < minDist){
	closestProb = *it;
	minDist = r;
      }
    }
      
  return closestProb;
}

std::pair<int,int> EcalClusterSeverityLevelAlgo::etaphiDistanceClosestProblematic( const reco::CaloCluster & cluster, 
						     const EcalRecHitCollection & recHits, 
										   const CaloTopology* topology,  const EcalSeverityLevelAlgo& sevlv )
{
  DetId seed=EcalClusterTools::getMaximum(cluster,&recHits).first;
  if ( (seed.det() != DetId::Ecal) || 
       (EcalSubdetector(seed.subdetId()) != EcalBarrel) )
    {
      edm::LogError("EcalClusterSeverityLevelAlgo") << "The cluster seed is not in the BARREL";
      //method not supported if not in Barrel
      return std::pair<int,int>(-1,-1);
    }

  DetId closestProb = closestProblematic(cluster , recHits, topology, sevlv);

  if (! closestProb.null())
    return std::pair<int,int>(EBDetId::distanceEta(EBDetId(seed),EBDetId(closestProb)),EBDetId::distancePhi(EBDetId(seed),EBDetId(closestProb)));
  else
    return std::pair<int,int>(-1,-1);
} 







