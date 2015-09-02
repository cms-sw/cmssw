#include "Calibration/Tools/interface/EcalRingCalibrationTools.h"
#include "DataFormats/DetId/interface/DetId.h"

//Includes need to read from geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

//by default is not initialized, gets initialized at first call
std::atomic<bool> EcalRingCalibrationTools::isInitializedFromGeometry_(false);
short EcalRingCalibrationTools::endcapRingIndex_[EEDetId::IX_MAX][EEDetId::IY_MAX];
std::once_flag EcalRingCalibrationTools::once_;


short EcalRingCalibrationTools::getRingIndex(DetId id) 
{
  if (id.det() != DetId::Ecal)
    return -1;
  
  if (id.subdetId() == EcalBarrel)
    {
      if(EBDetId(id).ieta()<0) 
	return EBDetId(id).ieta() + 85; 
      else 
	return EBDetId(id).ieta() + 84; 
      
    }
  if (id.subdetId() == EcalEndcap)
    {
      //needed only for the EE, it can be replaced at some point with something smarter
      if (not isInitializedFromGeometry_)
        throw std::logic_error("EcalRingCalibrationTools::initializeFromGeometry Ecal Endcap geometry is not initialized");

      EEDetId eid(id);
      short endcapRingIndex = endcapRingIndex_[eid.ix()-1][eid.iy()-1] + N_RING_BARREL;
      if (eid.zside() == 1) endcapRingIndex += N_RING_ENDCAP/2;
      return endcapRingIndex;
    }
  return -1;
}

short EcalRingCalibrationTools::getModuleIndex(DetId id) 
{

  if (id.det() != DetId::Ecal)
    return -1;
  
  if (id.subdetId() == EcalBarrel)
    {

      
      short module = 4*( EBDetId(id).ism() -1 ) + EBDetId(id).im() -1; 

      //      std::cout<<"SM construction # : "<<EBDetId(id).ism() <<" SM installation # : "<<  installationSMNumber[ EBDetId(id).ism() -1 ]<< "Xtal is Module :"<<module<< std::endl;      
      return module;
      
    }
  if (id.subdetId() == EcalEndcap)
    {
      
      return -1; 
    }

  return -1;
}

std::vector<DetId> EcalRingCalibrationTools::getDetIdsInRing(short etaIndex) 
{

  std::vector<DetId> ringIds;
  if (etaIndex < 0)
    return ringIds;
  
  if (etaIndex < N_RING_BARREL)
    {

      int k =0;
      if (etaIndex<85)
	k=-85 + etaIndex;
      else
	k= etaIndex - 84;

	for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) 
	  
	  if (EBDetId::validDetId(k,iphi))
	    ringIds.push_back(EBDetId(k,iphi));
    }

  else if (etaIndex < N_RING_TOTAL)
    {
      //needed only for the EE, it can be replaced at some point maybe with something smarter
      if (not isInitializedFromGeometry_)
        throw std::logic_error("EcalRingCalibrationTools::initializeFromGeometry Ecal Endcap geometry is not initialized");
      
      int zside= (etaIndex < N_RING_BARREL + (N_RING_ENDCAP/2) ) ? -1 : 1;
      short eeEtaIndex = (etaIndex - N_RING_BARREL)%(N_RING_ENDCAP/2); 

      for (int ix=0;ix<EEDetId::IX_MAX;++ix)
	for (int iy=0;iy<EEDetId::IY_MAX;++iy)
	  if (endcapRingIndex_[ix][iy] == eeEtaIndex)
	    ringIds.push_back(EEDetId(ix+1,iy+1,zside));
      
    }
  
  return ringIds;
} 

std::vector<DetId> EcalRingCalibrationTools::getDetIdsInECAL() 
{
  std::vector<DetId> ringIds;
  
  for(int ieta= - EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) 
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) 
      if (EBDetId::validDetId(ieta,iphi))
	ringIds.push_back(EBDetId(ieta,iphi));
  
  //needed only for the EE, it can be replaced at some point maybe with something smarter
  if (not isInitializedFromGeometry_)
    throw std::logic_error("EcalRingCalibrationTools::initializeFromGeometry Ecal Endcap geometry is not initialized");
  
  for (int ix=0;ix<EEDetId::IX_MAX;++ix)
    for (int iy=0;iy<EEDetId::IY_MAX;++iy)
      for(int zside = -1; zside<2; zside += 2)
	if ( EEDetId::validDetId(ix+1,iy+1,zside) )
	  ringIds.push_back( EEDetId(ix+1,iy+1,zside) );
  
  
  //  std::cout<<" [EcalRingCalibrationTools::getDetIdsInECAL()] DetId.size() is "<<ringIds.size()<<std::endl;
  
  return ringIds;
} 

std::vector<DetId> EcalRingCalibrationTools::getDetIdsInModule(short moduleIndex) 
{

  std::vector<DetId> ringIds;
  if (moduleIndex < 0)
    return ringIds;  

  short moduleBound[5] = {1, 26, 46, 66, 86};  
  if ( moduleIndex < EcalRingCalibrationTools::N_MODULES_BARREL)
    {
      ////////////////////////////nuovo ciclo
      short sm, moduleInSm, zsm;

      short minModuleiphi, maxModuleiphi, minModuleieta=360,maxModuleieta=0;

      //	if(moduleIndex%4 != 0 )
	  sm = moduleIndex / 4 + 1;
	  //	else
	  //sm = moduleIndex/4;//i.e. module 8 belongs to sm=3, not sm=3
	
	  //if(moduleIndex%4 != 0 )
	  moduleInSm = moduleIndex%4;
	  //else
	  //moduleInSm = 4;//moduleInSm is [1,2,3,4]

	if(moduleIndex > 71)
	  zsm = -1;
	else 
	  zsm = 1;
	
	minModuleiphi = ( (sm - 1) %18 + 1 ) *20 - 19;
	maxModuleiphi = ( (sm - 1) %18 + 1 ) * 20;
	
	if(zsm == 1)
	  {
	    minModuleieta = moduleBound[ moduleInSm   ];
	    maxModuleieta = moduleBound[ moduleInSm + 1 ] - 1;
	  }
	else if(zsm == -1){
	  minModuleieta = - moduleBound[ moduleInSm + 1 ] + 1;
	  maxModuleieta = - moduleBound[ moduleInSm ];
	} 
	////////////////////////////nuovo ciclo


	std::cout<<"Called moduleIndex "<<moduleIndex<<std::endl;
	std::cout<<"minModuleieta "<<minModuleieta<<" maxModuleieta "<<maxModuleieta<<" minModuleiphi "<<minModuleiphi<<" maxModuleiphi "<<maxModuleiphi<<std::endl;

	for(int ieta = minModuleieta; ieta <= maxModuleieta; ++ieta){ 
	  for(int iphi = minModuleiphi; iphi<= maxModuleiphi; ++iphi){
	    
	    
	      ringIds.push_back(EBDetId(ieta,iphi));
	      
	      //	      std::cout<<"Putting Xtal with ieta: "<<ieta<<" iphi "<<iphi<<" of SM "<<sm<<" into Module "<<moduleIndex<<std::endl;
	      
	    	    
	}//close loop on phi
	}//close loop on eta  
    }//close if ( moduleInstallationNumber < 144)
  
  return ringIds;
} 

void EcalRingCalibrationTools::setCaloGeometry(const CaloGeometry* geometry)
{ 
  std::call_once(once_, EcalRingCalibrationTools::initializeFromGeometry, geometry);
}

void EcalRingCalibrationTools::initializeFromGeometry(CaloGeometry const* geometry)
{
  if (not geometry)
    throw std::invalid_argument("EcalRingCalibrationTools::initializeFromGeometry called with a nullptr argument");

  float cellPosEta[EEDetId::IX_MAX][EEDetId::IY_MAX];
  for (int ix=0; ix<EEDetId::IX_MAX; ++ix) 
    for (int iy=0; iy<EEDetId::IY_MAX; ++iy)
    {
      cellPosEta[ix][iy] = -1.;
      endcapRingIndex_[ix][iy]=-9;
    }
  
  CaloSubdetectorGeometry const* endcapGeometry = geometry->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  if (not endcapGeometry)
    throw std::logic_error("EcalRingCalibrationTools::initializeFromGeometry Ecal Endcap geometry not found");

  std::vector<DetId> const& endcapCells = geometry->getValidDetIds(DetId::Ecal, EcalEndcap);

  for (std::vector<DetId>::const_iterator endcapIt = endcapCells.begin();
       endcapIt!=endcapCells.end();
       ++endcapIt)
    {
      EEDetId ee(*endcapIt);
      if (ee.zside() == -1) continue; //Just using +side to fill absEta x,y map
      const CaloCellGeometry *cellGeometry = endcapGeometry->getGeometry(*endcapIt) ;
      int ics=ee.ix() - 1 ;
      int ips=ee.iy() - 1 ;
      cellPosEta[ics][ips] = fabs(cellGeometry->getPosition().eta());
      //std::cout<<"EE Xtal, |eta| is "<<fabs(cellGeometry->getPosition().eta())<<std::endl;
    }
  
  float eta_ring[N_RING_ENDCAP/2];
  for (int ring=0; ring<N_RING_ENDCAP/2; ++ring)
    eta_ring[ring]=cellPosEta[ring][50];

  double etaBoundary[N_RING_ENDCAP/2 + 1];
  etaBoundary[0]=1.47;
  etaBoundary[N_RING_ENDCAP/2]=4.0;

  for (int ring=1; ring<N_RING_ENDCAP/2; ++ring)
    etaBoundary[ring]=(eta_ring[ring]+eta_ring[ring-1])/2.;
  
  for (int ring=0; ring<N_RING_ENDCAP/2; ++ring){
    // std::cout<<"***********************EE ring: "<<ring<<" eta "<<(etaBoundary[ring] + etaBoundary[ring+1])/2.<<std::endl;
    for (int ix=0; ix<EEDetId::IX_MAX; ix++)
      for (int iy=0; iy<EEDetId::IY_MAX; iy++)
	if (cellPosEta[ix][iy]>etaBoundary[ring] && cellPosEta[ix][iy]<etaBoundary[ring+1])
	{
	  endcapRingIndex_[ix][iy]=ring;
	  //std::cout<<"endcapRing_["<<ix+1<<"]["<<iy+1<<"] = "<<ring<<";"<<std::endl;  
	}
  }

  std::vector<DetId> const& barrelCells = geometry->getValidDetIds(DetId::Ecal, EcalBarrel);

  for (std::vector<DetId>::const_iterator barrelIt = barrelCells.begin();
       barrelIt!=barrelCells.end();
       ++barrelIt)
    {
      EBDetId eb(*barrelIt);
    }

  //EB

  isInitializedFromGeometry_ = true;

}


