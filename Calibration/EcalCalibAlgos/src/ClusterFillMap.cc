#include "Calibration/EcalCalibAlgos/interface/ClusterFillMap.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

	ClusterFillMap::ClusterFillMap (int WindowX,
			int WindowY,
			std::map<int,int> xtalReg,
			double minE,
			double maxE, 
			std::map<int,int> IndexReg,
			EcalIntercalibConstantMap * barrelMap,
			EcalIntercalibConstantMap * endcapMap):
   	      VFillMap (WindowX,WindowY,xtalReg,minE,
			maxE, IndexReg,
			barrelMap,endcapMap)

	      {
 	      }

ClusterFillMap::~ClusterFillMap()
{
}

DetId 
ClusterFillMap::fillMap (const std::vector<DetId> & v1,
                const EcalRecHitCollection * barrelHitsCollection,
		const EcalRecHitCollection * endcapHitsCollection,
                std::map<int,double> & xtlMap,
                double & pSubtract )
{
 
  DetId Max = findMaxHit (v1, barrelHitsCollection, endcapHitsCollection);
  for (std::vector<DetId>::const_iterator idsIt = v1.begin();
       idsIt != v1.end () ;
       ++idsIt)
 {
   int RegionNumber = m_xtalRegionId[Max.rawId()];
   EcalRecHitCollection::const_iterator itrechit;
   double dummy;
   if(idsIt->subdetId()==EcalBarrel)
   {
     itrechit=barrelHitsCollection->find(*idsIt);
     dummy=itrechit->energy();
     dummy*= (*m_barrelMap)[*idsIt];
   }
   if(idsIt->subdetId()==EcalEndcap){
     itrechit=endcapHitsCollection->find(*idsIt);
     dummy=itrechit->energy();
     dummy*= (*m_endcapMap)[*idsIt];
   }
   int ID=idsIt->rawId();
   if (isnan(dummy)) {
     dummy=0;
   }
   if ( dummy < m_minEnergyPerCrystal ) continue; //return 1; 
   if ( dummy > m_maxEnergyPerCrystal ) {
	dummy=0;
	continue;
   }
   if (m_xtalRegionId[ID]==RegionNumber)
      xtlMap[m_IndexInRegion[ID]] += dummy;
   else pSubtract +=dummy; 
 }
 return Max;
}

