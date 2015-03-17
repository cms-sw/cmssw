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

#include <DataFormats/ParticleFlowReco/interface/PFRecHit.h>

#include <algorithm>

using namespace pandora;
using namespace cms_content;

//------------------------------------------------------------------------------------------------------------------------------------------

GlobalHadronCompensation::GlobalHadronCompensation() :
  m_nMIPsCut(1.0),
  m_e_em_EE(1.0), m_e_em_FH(1.0), m_e_em_BH(1.0)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode GlobalHadronCompensation::MakeEnergyCorrections(const Cluster *const pCluster, float &correctedHadronicEnergy) const
{

  //const unsigned int firstPseudoLayer(this->GetPandora().GetPlugins()->GetPseudoLayerPlugin()->GetPseudoLayerAtIp());

  const OrderedCaloHitList& orderedCaloHitList = pCluster->GetOrderedCaloHitList();
  
  OrderedCaloHitList::const_iterator layer = orderedCaloHitList.begin();
  OrderedCaloHitList::const_iterator hits_end = orderedCaloHitList.end();

  //compute raw energy
  float rawen_EE(0.f),     rawen_FH(0.f), rawen_BH(0.f);
  float avgen_mip_FH(0.f), nHits_FH(0.f);
  for( ; layer != hits_end; ++ layer ) 
    {
      const CaloHitList& hits_in_layer = *(layer->second);
      for( const auto& hit : hits_in_layer ) 
	{
	  //check sub-detector id
	  const void* void_hit_ptr = hit->GetParentCaloHitAddress();
	  const reco::PFRecHit* original_hit_ptr = static_cast<const reco::PFRecHit*>(void_hit_ptr);
	  const uint32_t rawid = original_hit_ptr->detId();
	  const int subDetId = (rawid>>25)&0x7;
      
	  float nLambdas(hit->GetNCellInteractionLengths());
	  float nMIPs(hit->GetMipEquivalentEnergy());

	  if(subDetId == ForwardSubdetector::HGCEE)    
	    {
	      rawen_EE += nLambdas*nMIPs/m_e_em_EE;
	    }
	  else if(subDetId == ForwardSubdetector::HGCHEB)
	    {
	      rawen_BH += nLambdas*nMIPs/m_e_em_FH;
	    }
	  else if( subDetId == ForwardSubdetector::HGCHEF ) 
	    {
	      rawen_FH     += nLambdas*nMIPs/m_e_em_BH;
	      avgen_mip_FH += nMIPs;
	      nHits_FH     += 1.0;
	    }
	}
    }
  
  //further correction only makes sense if enough energy
  float rawen(rawen_EE+rawen_FH+rawen_BH);
  if( rawen<0.01f)
    {
      correctedHadronicEnergy *= 1.f;
      return STATUS_CODE_SUCCESS;
    }


  //global compensation if energy in FH is significant
  float c_FH(1.0);
  if(nHits_FH>0)
    {
      avgen_mip_FH/=nHits_FH;
      layer = orderedCaloHitList.begin();
      float nHits_avg_FH(0.0), nHits_elim_FH(0.0);
      for( ; layer != hits_end; ++ layer ) 
	{
	  const CaloHitList& hits_in_layer = *(layer->second);
	  for( const auto& hit : hits_in_layer ) 
	    {
	      // hack so that we can know if we are in the HEF or not (go back to cmssw det id)                             
	      const void* void_hit_ptr = hit->GetParentCaloHitAddress();
	      const reco::PFRecHit* original_hit_ptr = static_cast<const reco::PFRecHit*>(void_hit_ptr);
	      const uint32_t rawid = original_hit_ptr->detId();
	      const int subDetId = (rawid>>25)&0x7;
	      if( subDetId != ForwardSubdetector::HGCHEF ) continue;
	      
	      const float nMIPs = hit->GetMipEquivalentEnergy();
	      nHits_avg_FH  += ( nMIPs > avgen_mip_FH);
	      nHits_elim_FH += ( nMIPs > m_nMIPsCut );
	    }
	}
      
      c_FH=(nHits_FH-nHits_elim_FH)/(nHits_FH-nHits_avg_FH);
    }


  //pi/e corrected energy
  float en_EE(getByPiOverECorrectedEn(rawen_EE,rawen,int(ForwardSubdetector::HGCEE))); 
  float en_FH(c_FH*getByPiOverECorrectedEn(rawen_FH,rawen,int(ForwardSubdetector::HGCHEF))); 
  float en_BH(getByPiOverECorrectedEn(rawen_BH,rawen,int(ForwardSubdetector::HGCHEB))); 
  float en_HCAL(en_FH+en_BH);
  float en(en_EE+en_HCAL);
  
  //energy sharing (ECAL MIP subtracted)
  float en_m_EEmip(std::max(en_EE-m_IntegMIP_emEn_EE,0.f)+en_HCAL);
  float enFracInHCAL_m_mip(-1);
  if(en_HCAL==0) enFracInHCAL_m_mip=0;
  if(en_m_EEmip>0) enFracInHCAL_m_mip=en_HCAL/en_m_EEmip;
  
  //apply residual correction according to energy sharing
  float residualScale=getResidualScale(en,enFracInHCAL_m_mip);
  

  /*
    std::cout << "Initial: " << correctedHadronicEnergy
	      << " Raw: " << rawen 
	      << " After pi/e: " << en
	      << " GC: " << c_FH
	      << " Residual: " << residualScale 
	      << " Final: " << residualScale*en
	      << std::endl
	      << "\t EE: " << rawen_EE << "->" << en_EE
	      << " FH: " << rawen_FH << "->" << en_FH
	      << " BH: " << rawen_BH << "->" << en_BH
	      << std::endl;
  */  
  //all done here
  correctedHadronicEnergy=residualScale*en;

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
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "MipEnergyThreshold", m_nMIPsCut));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "IntegMIP_emEn_EE",   m_IntegMIP_emEn_EE));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "e_em_EE", m_e_em_EE));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "e_em_FH", m_e_em_FH));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "e_em_BH", m_e_em_BH));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "pioe_EE", m_pioe_EE));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "pioe_FH", m_pioe_FH));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "pioe_BH", m_pioe_BH));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "pioe_bananaParam_0", m_pioe_bananaParam_0 ));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "pioe_bananaParam_1", m_pioe_bananaParam_1 ));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "pioe_bananaParam_2", m_pioe_bananaParam_2 ));
  
  m_IntegMIP_emEn_EE=getByPiOverECorrectedEn(m_IntegMIP_emEn_EE,m_IntegMIP_emEn_EE,(int)(ForwardSubdetector::HGCEE));

  std::cout << "[CMSGlobalHadronCompensationPlugin::ReadSettings]" << std::endl
	    << "MIP threshold for GC: " << m_nMIPsCut << std::endl
	    << "MIP value in EE (after pi/e/):" << m_IntegMIP_emEn_EE << std::endl
	    << "e.m. scales " << 1./m_e_em_EE << " " << 1./m_e_em_FH << " " << 1./m_e_em_BH << std::endl
	    << "# pi/e parameters " << m_pioe_EE.size() << " " << m_pioe_FH.size() << " " << m_pioe_BH.size() <<" " << std::endl
	    << "# res. corr parameters " << m_pioe_bananaParam_0.size() << " " << m_pioe_bananaParam_1.size()<< " " << m_pioe_bananaParam_2.size() << std::endl;
  
  

  return STATUS_CODE_SUCCESS;
}
