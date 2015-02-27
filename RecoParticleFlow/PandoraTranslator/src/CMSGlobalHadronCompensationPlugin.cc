/**
 *  @file   RecoParticleFlow/PandoraTranslator/src/CMSGlobalHadronCompensationPlugin.cc
 * 
 *  @brief  Implementation of the Global Hadronic Energy Compensation strategy
 * 
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "LCHelpers/ReclusterHelper.h"

#include "RecoParticleFlow/PandoraTranslator/interface/CMSGlobalHadronCompensationPlugin.h"

#include <DataFormats/ForwardDetId/interface/ForwardSubdetector.h>
#include <DataFormats/ParticleFlowReco/interface/PFRecHit.h>

using namespace pandora;
using namespace cms_content;

//------------------------------------------------------------------------------------------------------------------------------------------

GlobalHadronCompensation::GlobalHadronCompensation() :
  m_nMIPsCut(10.0f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode GlobalHadronCompensation::MakeEnergyCorrections(const Cluster *const pCluster, float &correctedHadronicEnergy) const
{

  //const unsigned int firstPseudoLayer(this->GetPandora().GetPlugins()->GetPseudoLayerPlugin()->GetPseudoLayerAtIp());

  const OrderedCaloHitList& orderedCaloHitList = pCluster->GetOrderedCaloHitList();
  
  OrderedCaloHitList::const_iterator layer = orderedCaloHitList.begin();
  OrderedCaloHitList::const_iterator hits_end = orderedCaloHitList.end();
  
  float en_avg(0.f), nHits(0.f);

  for( ; layer != hits_end; ++ layer ) {
    const CaloHitList& hits_in_layer = *(layer->second);
    for( const auto& hit : hits_in_layer ) {
      // hack so that we can know if we are in the HEF or not (go back to cmssw det id)
      const void* void_hit_ptr = hit->GetParentCaloHitAddress();
      const reco::PFRecHit* original_hit_ptr = static_cast<const reco::PFRecHit*>(void_hit_ptr);
      const uint32_t rawid = original_hit_ptr->detId();
      const int subDetId = (rawid>>25)&0x7;
      if( subDetId != ForwardSubdetector::HGCHEF ) continue;
      
      const float en_mip = hit->GetMipEquivalentEnergy();
      en_avg += en_mip;
      nHits += 1.f;
    }    
  }
  
  if( nHits == 0.f ) {
    correctedHadronicEnergy *= 1.f;
    return STATUS_CODE_SUCCESS;
  }

  en_avg /= nHits;

  float nHits_avg(0.f), nHits_elim(0.f);
  for( ; layer != hits_end; ++ layer ) {
    const CaloHitList& hits_in_layer = *(layer->second);
    for( const auto& hit : hits_in_layer ) {
      // hack so that we can know if we are in the HEF or not (go back to cmssw det id)                             
      const void* void_hit_ptr = hit->GetParentCaloHitAddress();
      const reco::PFRecHit* original_hit_ptr = static_cast<const reco::PFRecHit*>(void_hit_ptr);
      const uint32_t rawid = original_hit_ptr->detId();
      const int subDetId = (rawid>>25)&0x7;
      if( subDetId != ForwardSubdetector::HGCHEF ) continue;

      const float en_mip = hit->GetMipEquivalentEnergy();
      if( en_mip > en_avg ) {
	nHits_avg += 1.f;
      } else {
	nHits_elim += 1.f;
      }
    }
  }

  correctedHadronicEnergy *= (nHits-nHits_elim)/(nHits-nHits_avg);

  return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

float GlobalHadronCompensation::GetHadronicEnergyInLayer(const OrderedCaloHitList &orderedCaloHitList, const unsigned int pseudoLayer) const
{
  OrderedCaloHitList::const_iterator iter = orderedCaloHitList.find(pseudoLayer);

  float hadronicEnergy(0.f);

  if (iter != orderedCaloHitList.end())
    {
      for (CaloHitList::const_iterator hitIter = iter->second->begin(), hitIterEnd = iter->second->end(); hitIter != hitIterEnd; ++hitIter)
        {
	  hadronicEnergy += (*hitIter)->GetHadronicEnergy();
        }
    }

  return hadronicEnergy;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode GlobalHadronCompensation::ReadSettings(const TiXmlHandle xmlHandle)
{
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
												       "MipEnergyThreshold", m_nMIPsCut));
  
  return STATUS_CODE_SUCCESS;
}
