#include "Calibration/EcalCalibAlgos/interface/MatrixFillMap.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/Utilities/interface/isFinite.h"

MatrixFillMap::MatrixFillMap (int WindowX,
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

MatrixFillMap::~MatrixFillMap ()
	{
	}


void
MatrixFillMap::fillMap (const std::vector<std::pair<DetId,float> >  & v1,
	const DetId Max,
	 const EcalRecHitCollection * barrelHitsCollection,
	 const EcalRecHitCollection * endcapHitsCollection,
	 std::map<int, double> & xtlMap,
	 double & pSubtract)
{
	if (Max.subdetId() == EcalBarrel ){
		EBDetId EBMax = Max;
                fillEBMap (EBMax, barrelHitsCollection, xtlMap,
                     m_xtalRegionId[Max.rawId()], pSubtract ) ;
         }
        else if (Max.subdetId()== EcalEndcap){
               EEDetId EEMax = Max;
               fillEEMap (EEMax, endcapHitsCollection, xtlMap,
                     m_xtalRegionId[Max.rawId()],pSubtract ) ;
	}
}



void 
MatrixFillMap::fillEBMap (EBDetId EBmax,
	 const EcalRecHitCollection * barrelHitsCollection,
	 std::map<int, double> & EBRegionMap,
	 int EBNumberOfRegion, double & pSubtract)
{
  int curr_eta;
  int curr_phi;
  //reads the hits in a recoWindowSide^2 wide region around the MOX
  for (int ii = 0 ; ii< m_recoWindowSidex ; ++ii)
   for (int ij =0 ; ij< m_recoWindowSidey ; ++ij) 
   {
    curr_eta=EBmax.ieta() + ii - (m_recoWindowSidex/2);
    curr_phi=EBmax.iphi() + ij - (m_recoWindowSidey/2);
    //skips if the xtals matrix falls over the border
    if (abs(curr_eta)>85) continue;
    //Couples with the zero gap in the barrel eta index
    if (curr_eta * EBmax.ieta() <= 0) {
	    if (EBmax.ieta() > 0) curr_eta--; 
	      else curr_eta++;
        } // JUMP over 0
    //The following 2 couples with the ciclicity of the phiIndex
    if (curr_phi < 1) curr_phi += 360;
    if (curr_phi >= 360) curr_phi -= 360;
    //checks if the detId is valid
    if(EBDetId::validDetId(curr_eta,curr_phi))
     {
      EBDetId det = EBDetId(curr_eta,curr_phi,EBDetId::ETAPHIMODE);
      int ID= det.rawId();
      //finds the hit corresponding to the cell
      EcalRecHitCollection::const_iterator curr_recHit = barrelHitsCollection->find(det) ;
      double dummy = 0;
      dummy = curr_recHit->energy () ;
      //checks if the reading of the xtal is in a sensible range
      if (edm::isNotFinite(dummy)){
	  dummy=0;
       }	
      if ( dummy < m_minEnergyPerCrystal) continue; 
      if (dummy > m_maxEnergyPerCrystal)  continue;
     //corrects the energy with the calibration coeff of the ring
      dummy *= (*m_barrelMap)[det];
      //sums the energy of the xtal to the appropiate ring
      if (m_xtalRegionId[ID]==EBNumberOfRegion)
        EBRegionMap[m_IndexInRegion[ID]]+= dummy;
      //adds the reading to pSubtract when part of the matrix is outside the region
      else pSubtract +=dummy; 
     }
   }
}

void MatrixFillMap::fillEEMap (EEDetId EEmax,
                const EcalRecHitCollection * endcapHitsCollection,
                std::map<int,double> & EExtlMap,
                int EENumberOfRegion, double & pSubtract )
{
 int curr_x;
 int curr_y;
 for (int ii = 0 ; ii< m_recoWindowSidex ; ++ii)
  for (int ij =0 ; ij< m_recoWindowSidey ; ++ij) 
 {
  //Works as fillEBMap
  curr_x = EEmax.ix() - m_recoWindowSidex/2 +ii;
  curr_y = EEmax.iy() - m_recoWindowSidey /2 +ij;
  if(EEDetId::validDetId(curr_x,curr_y,EEmax.zside()))
  {
   EEDetId det = EEDetId(curr_x,curr_y,EEmax.zside(),EEDetId::XYMODE);
   int ID=det.rawId();
   EcalRecHitCollection::const_iterator curr_recHit = endcapHitsCollection->find(det) ;
   double dummy = curr_recHit->energy () ;
   if (edm::isNotFinite(dummy)) {
     dummy=0;
   }
   if ( dummy < m_minEnergyPerCrystal ) continue; 
   if ( dummy > m_maxEnergyPerCrystal ) {
	dummy=0;
	continue;
   }
   dummy *= (*m_endcapMap)[det];
   if (m_xtalRegionId[ID]==EENumberOfRegion)
      EExtlMap[m_IndexInRegion[ID]] += dummy;
   else pSubtract +=dummy; 
  }
 }
}
