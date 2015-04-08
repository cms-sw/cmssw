/**
 *  @file   CMSPseudoLayerPlugin.cc
 * 
 *  @brief  Implementation of the CMS pseudo layer plugin class.
 * 
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "RecoParticleFlow/PandoraTranslator/interface/CMSPseudoLayerPlugin.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace pandora;

namespace cms_content
{

CMSPseudoLayerPlugin::CMSPseudoLayerPlugin() :
  PseudoLayerPlugin(),
  m_barrelInnerEdgeR(0.f),
  m_barrelOuterEdgeR(0.f),
  m_endCapInnerEdgeZ(0.f),
  m_endCapOuterEdgeZ(0.f),
  m_endCapOuterEdgeR(0.f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CMSPseudoLayerPlugin::Initialize()
{
    try
    {
        this->StoreLayerPositions();
        this->StoreDetectorOuterEdge();
        //this->StorePolygonAngles();
        //this->StoreOverlapCorrectionDetails();
    }
    catch (StatusCodeException &statusCodeException)
    {
        std::cout << "CMSPseudoLayerPlugin: Incomplete geometry - consider using a different PseudoLayerPlugin." << std::endl;
        return statusCodeException.GetStatusCode();
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

unsigned int CMSPseudoLayerPlugin::GetPseudoLayer(const CartesianVector &positionVector) const
{
  const float zCoordinate(std::fabs(positionVector.GetZ()));
  const float rCoordinate(std::sqrt(positionVector.GetX() * positionVector.GetX() + positionVector.GetY() * positionVector.GetY()));
  
    if ((zCoordinate > m_endCapOuterEdgeZ) || (rCoordinate > std::max(m_barrelOuterEdgeR,m_endCapOuterEdgeR))) {
      if( (zCoordinate > m_endCapOuterEdgeZ) ) {
	throw cms::Exception("BadRecHit")
	  << " Rechit exists outside of CMS detector!";
      }
      unsigned int pseudoLayer(0);
      // hack -> since we only use the HGC hits assume we're in the endcap
      PANDORA_THROW_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, this->FindMatchingLayer(zCoordinate, m_endCapLayerPositions, pseudoLayer));
      /*
      std::cout << "You die here because z = " << zCoordinate << " (" << m_endCapOuterEdgeZ << ") and r = " << rCoordinate 
		<< " (" << m_barrelOuterEdgeR << "," << m_endCapOuterEdgeR << ")" << std::endl; 
      */
      return pseudoLayer; // NS FIX temporary <=============!!!!!!!!!!!! 
        throw pandora::StatusCodeException(pandora::STATUS_CODE_NOT_FOUND);
    }

    unsigned int pseudoLayer(0);

    if (zCoordinate < m_endCapInnerEdgeZ)
    {
        PANDORA_THROW_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, this->FindMatchingLayer(rCoordinate, m_barrelLayerPositions, pseudoLayer));
    }
    else if (rCoordinate < m_barrelInnerEdgeR)
    {
        PANDORA_THROW_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, this->FindMatchingLayer(zCoordinate, m_endCapLayerPositions, pseudoLayer));
    }
    else
    {
        unsigned int bestBarrelLayer(0);
        const pandora::StatusCode barrelStatusCode(this->FindMatchingLayer(rCoordinate, m_barrelLayerPositions, bestBarrelLayer));

        unsigned int bestEndCapLayer(0);
        const pandora::StatusCode endCapStatusCode(this->FindMatchingLayer(zCoordinate, m_endCapLayerPositions, bestEndCapLayer));

        if ((pandora::STATUS_CODE_SUCCESS != barrelStatusCode) && (pandora::STATUS_CODE_SUCCESS != endCapStatusCode))
            throw pandora::StatusCodeException(pandora::STATUS_CODE_NOT_FOUND);

        pseudoLayer = std::max(bestBarrelLayer, bestEndCapLayer);
    }

    // Reserve a pseudo layer for track projections, etc.
    return (1 + pseudoLayer);
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CMSPseudoLayerPlugin::FindMatchingLayer(const float position, const LayerPositionList &layerPositionList,
    unsigned int &layer) const
{
    LayerPositionList::const_iterator upperIter = std::upper_bound(layerPositionList.begin(), layerPositionList.end(), position);

    if (layerPositionList.end() == upperIter)
    {
        return STATUS_CODE_NOT_FOUND;
    }

    if (layerPositionList.begin() == upperIter)
    {
        layer = 0;
        return STATUS_CODE_SUCCESS;
    }

    LayerPositionList::const_iterator lowerIter = upperIter - 1;

    if (std::fabs(position - *lowerIter) < std::fabs(position - *upperIter))
    {
        layer = std::distance(layerPositionList.begin(), lowerIter);
    }
    else
    {
        layer = std::distance(layerPositionList.begin(), upperIter);
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void CMSPseudoLayerPlugin::StoreLayerPositions()
{
    const GeometryManager *const pGeometryManager(this->GetPandora().GetGeometry());
    this->StoreLayerPositions(pGeometryManager->GetSubDetector(ECAL_BARREL), m_barrelLayerPositions);
    this->StoreLayerPositions(pGeometryManager->GetSubDetector(ECAL_ENDCAP), m_endCapLayerPositions);
    this->StoreLayerPositions(pGeometryManager->GetSubDetector(HCAL_BARREL), m_barrelLayerPositions);
    this->StoreLayerPositions(pGeometryManager->GetSubDetector(HCAL_ENDCAP), m_endCapLayerPositions);    

    if (m_barrelLayerPositions.empty() || m_endCapLayerPositions.empty())
    {
        std::cout << "CMSPseudoLayerPlugin: No layer positions specified." << std::endl;
        throw StatusCodeException(STATUS_CODE_NOT_INITIALIZED);
    }

    std::sort(m_barrelLayerPositions.begin(), m_barrelLayerPositions.end());
    std::sort(m_endCapLayerPositions.begin(), m_endCapLayerPositions.end());

    LayerPositionList::const_iterator barrelIter = std::unique(m_barrelLayerPositions.begin(), m_barrelLayerPositions.end());
    LayerPositionList::const_iterator endcapIter = std::unique(m_endCapLayerPositions.begin(), m_endCapLayerPositions.end());

    if ((m_barrelLayerPositions.end() != barrelIter) || (m_endCapLayerPositions.end() != endcapIter))
    {
        std::cout << "CMSPseudoLayerPlugin: Duplicate layer position detected." << std::endl;
        throw StatusCodeException(STATUS_CODE_FAILURE);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void CMSPseudoLayerPlugin::StoreLayerPositions(const SubDetector &subDetector, LayerPositionList &layerPositionList)
{
    if (!subDetector.IsMirroredInZ())
    {
        std::cout << "CMSPseudoLayerPlugin: Error, detector must be symmetrical about z=0 plane." << std::endl;
        throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);
    }

    const SubDetector::SubDetectorLayerList &subDetectorLayerList(subDetector.GetSubDetectorLayerList());

    for (SubDetector::SubDetectorLayerList::const_iterator iter = subDetectorLayerList.begin(), iterEnd = subDetectorLayerList.end(); iter != iterEnd; ++iter)
    {
        layerPositionList.push_back(iter->GetClosestDistanceToIp());
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void CMSPseudoLayerPlugin::StoreDetectorOuterEdge()
{
  const GeometryManager *const pGeometryManager(this->GetPandora().GetGeometry());
  
  const SubDetector& ecalBarrel = pGeometryManager->GetSubDetector(ECAL_BARREL);
  const SubDetector& hcalBarrel = pGeometryManager->GetSubDetector(HCAL_BARREL);
  const SubDetector& ecalEndCap = pGeometryManager->GetSubDetector(ECAL_ENDCAP);
  const SubDetector& hcalEndCap = pGeometryManager->GetSubDetector(HCAL_ENDCAP);
  
  m_barrelInnerEdgeR = ecalBarrel.GetInnerRCoordinate(); 
  // m_barrelInnerEdgeR = hcalBarrelParameters.GetInnerRCoordinate(); // debugging
  m_barrelOuterEdgeR = hcalBarrel.GetOuterRCoordinate(); 
  // m_barrelOuterEdgeR = ecalBarrelParameters.GetOuterRCoordinate(); // debugging
  m_endCapOuterEdgeR = std::max(ecalEndCap.GetOuterRCoordinate(),hcalEndCap.GetOuterRCoordinate()); 
  // m_endcapOuterEdgeR = ecalEndCapParameters.GetOuterRCoordinate() ; // debugging
  m_endCapInnerEdgeZ = ecalEndCap.GetInnerZCoordinate(); 
  // m_endCapInnerEdgeZ = hcalEndCapParameters.GetInnerZCoordinate(); // debugging
  m_endCapOuterEdgeZ = hcalEndCap.GetOuterZCoordinate(); // debugging
  
  if ((m_barrelLayerPositions.end() != std::upper_bound(m_barrelLayerPositions.begin(), m_barrelLayerPositions.end(), m_barrelOuterEdgeR)) ||
      (m_endCapLayerPositions.end() != std::upper_bound(m_endCapLayerPositions.begin(), m_endCapLayerPositions.end(), m_endCapOuterEdgeZ))) 
  // if ((m_barrelLayerPositions.end() != std::upper_bound(m_barrelLayerPositions.begin(), m_barrelLayerPositions.end(), m_barrelOuterEdgeR)))
    {
      std::cout << m_barrelOuterEdgeR << ' ' << m_endCapOuterEdgeZ << std::endl;
      std::cout << "FineGranularityPseudoLayerCalculator: Layers specified outside detector edge." << std::endl;
      throw pandora::StatusCodeException(pandora::STATUS_CODE_FAILURE);
    }

  m_barrelLayerPositions.push_back(m_barrelOuterEdgeR);
  m_endCapLayerPositions.push_back(m_endCapOuterEdgeZ);
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CMSPseudoLayerPlugin::ReadSettings(const TiXmlHandle /*xmlHandle*/)
{
    return STATUS_CODE_SUCCESS;
}

} // namespace lc_content
