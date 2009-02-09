#include "Calibration/EcalCalibAlgos/interface/VFillMap.h"

	VFillMap::VFillMap (int WindowX,int WindowY,
			std::map<int,int> xtalReg,double minE,
			double maxE, std::map<int,int> RingReg,
			EcalIntercalibConstantMap * barrelMap,
			EcalIntercalibConstantMap * endcapMap):
	m_recoWindowSidex (WindowX),
	m_recoWindowSidey (WindowY),
        m_xtalRegionId (xtalReg),
        m_minEnergyPerCrystal (minE),
        m_maxEnergyPerCrystal (maxE),
        m_IndexInRegion (RingReg),
        m_barrelMap (barrelMap),
        m_endcapMap (endcapMap)
	
	{}


DetId
 VFillMap::findMaxHit (const std::vector<std::pair<DetId,float> > & v1,
		            const EcalRecHitCollection * EBhits,
			    const EcalRecHitCollection * EEhits)
{
 double currEnergy = 0.;
 //creates an empty DetId 
 DetId maxHit;
 //Loops over the vector of the recHits
 for (std::vector<std::pair<DetId,float> >::const_iterator idsIt = v1.begin (); 
      idsIt != v1.end () ; ++idsIt)
   {
    if (idsIt->first.subdetId () == EcalBarrel) 
       {              
         EBRecHitCollection::const_iterator itrechit;
         itrechit = EBhits->find (idsIt->first) ;
	       //not really understood why this should happen, but it happens
         if (itrechit == EBhits->end () )
           {
            continue;
           }
	       //If the energy is greater than the currently stored energy 
         //sets maxHits to the current recHit
         if (itrechit->energy () > currEnergy)
           {
             currEnergy = itrechit->energy () ;
             maxHit= idsIt->first ;
           }
       } //barrel part ends
    else 
       {     
	       //as the barrel part
         EERecHitCollection::const_iterator itrechit;
         itrechit = EEhits->find (idsIt->first) ;
         if (itrechit == EEhits->end () )
           {
             continue;
           }              
         if (itrechit->energy () > currEnergy)
           {
            currEnergy=itrechit->energy ();
            maxHit= idsIt->first;
           }
       } //ends the endcap part
    } //end of the loop over the detId
 return maxHit;
}
