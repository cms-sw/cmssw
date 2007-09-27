#include "Calibration/Tools/interface/EcalRingCalibrationTools.h"
#include "DataFormats/DetId/interface/DetId.h"

//Includes need to read from geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//by default is not initialized, gets initialized at first call
bool EcalRingCalibrationTools::isInitializedFromGeometry_ = false;
const CaloGeometry* EcalRingCalibrationTools::caloGeometry_ = 0;
short EcalRingCalibrationTools::endcapRingIndex_[EEDetId::IX_MAX][EEDetId::IY_MAX];


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
      if (!isInitializedFromGeometry_)
	initializeFromGeometry();
      EEDetId eid(id);
      short endcapRingIndex = endcapRingIndex_[eid.ix()-1][eid.iy()-1] + N_RING_BARREL;
      if (eid.zside() == 1) endcapRingIndex += N_RING_ENDCAP/2;
      return endcapRingIndex;
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
      //needed only for the EE, it can be replaced at some point maybe with something samarter
      if (!isInitializedFromGeometry_)
	initializeFromGeometry();
      
      int zside= (etaIndex < N_RING_BARREL + (N_RING_ENDCAP/2) ) ? -1 : 1;
      short eeEtaIndex = (etaIndex - N_RING_BARREL)%(N_RING_ENDCAP/2); 

      for (int ix=0;ix<EEDetId::IX_MAX;++ix)
	for (int iy=0;iy<EEDetId::IY_MAX;++iy)
	  if (endcapRingIndex_[ix][iy] == eeEtaIndex)
	    ringIds.push_back(EEDetId(ix+1,iy+1,zside));
      
    }
  
  return ringIds;
} 

void EcalRingCalibrationTools::initializeFromGeometry()
{

  if (!caloGeometry_)
    {
      edm::LogError("EcalRingCalibrationTools") << "BIG ERROR::Initializing without geometry handle" ;
      return;
    }

  float m_cellPosEta[EEDetId::IX_MAX][EEDetId::IY_MAX];
  for (int ix=0; ix<EEDetId::IX_MAX; ++ix) 
    for (int iy=0; iy<EEDetId::IY_MAX; ++iy) 
      {
	m_cellPosEta[ix][iy] = -1.;
	endcapRingIndex_[ix][iy]=-9;
      }
  
  
  const CaloSubdetectorGeometry *endcapGeometry = caloGeometry_->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

  if (!endcapGeometry)
    {
      edm::LogError("EcalRingCalibrationTools") << "BIG ERROR::Ecal Endcap geometry not found" ;
      return;
    }

  std::vector<DetId> m_endcapCells= caloGeometry_->getValidDetIds(DetId::Ecal, EcalEndcap);

  for (std::vector<DetId>::const_iterator endcapIt = m_endcapCells.begin();
       endcapIt!=m_endcapCells.end();
       ++endcapIt)
    {
      EEDetId ee(*endcapIt);
      if (ee.zside() == -1) continue; //Just using +side to fill absEta x,y map
      const CaloCellGeometry *cellGeometry = endcapGeometry->getGeometry(*endcapIt) ;
      int ics=ee.ix() - 1 ;
      int ips=ee.iy() - 1 ;
      m_cellPosEta[ics][ips] = fabs(cellGeometry->getPosition().eta());
    }
  
  float eta_ring[N_RING_ENDCAP/2];
  for (int ring=0; ring<N_RING_ENDCAP/2; ++ring)
    eta_ring[ring]=m_cellPosEta[ring][50];

  double etaBoundary[N_RING_ENDCAP/2 + 1];
  etaBoundary[0]=1.47;
  etaBoundary[N_RING_ENDCAP/2]=4.0;

  for (int ring=1; ring<N_RING_ENDCAP/2; ++ring)
    etaBoundary[ring]=(eta_ring[ring]+eta_ring[ring-1])/2.;

  for (int ring=0; ring<N_RING_ENDCAP/2; ring++)
    for (int ix=0; ix<EEDetId::IX_MAX; ix++)
      for (int iy=0; iy<EEDetId::IY_MAX; iy++)
	if (m_cellPosEta[ix][iy]>etaBoundary[ring] && m_cellPosEta[ix][iy]<etaBoundary[ring+1])
	  endcapRingIndex_[ix][iy]=ring;

  isInitializedFromGeometry_ = true;
}
