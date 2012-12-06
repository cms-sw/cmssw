#include "Calibration/EcalCalibAlgos/interface/ClusterFillMap.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/Utilities/interface/isFinite.h"

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

void 
ClusterFillMap::fillMap (const std::vector<std::pair<DetId,float> > & v1,
		const DetId Max,
                const EcalRecHitCollection * barrelHitsCollection,
		const EcalRecHitCollection * endcapHitsCollection,
                std::map<int,double> & xtlMap,
                double & pSubtract )
{
  for (std::vector<std::pair<DetId,float> >::const_iterator idsIt = v1.begin();
       idsIt != v1.end () ;
       ++idsIt)
 {
   int RegionNumber = m_xtalRegionId[Max.rawId()];
   EcalRecHitCollection::const_iterator itrechit;
   double dummy=0.;
   if(idsIt->first.subdetId()==EcalBarrel)
   {
     itrechit=barrelHitsCollection->find(idsIt->first);
     dummy=itrechit->energy();
     dummy*= (*m_barrelMap)[idsIt->first];
   }
   if(idsIt->first.subdetId()==EcalEndcap){
     itrechit=endcapHitsCollection->find(idsIt->first);
     dummy=itrechit->energy();
     dummy*= (*m_endcapMap)[idsIt->first];
   }
   int ID=idsIt->first.rawId();
   if (edm::isNotFinite(dummy)) {
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
}

