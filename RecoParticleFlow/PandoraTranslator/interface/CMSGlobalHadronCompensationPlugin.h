/**
 *  @file   RecoParticleFlow/PandoraTranslator/interface/CMSGlobalHadronCompensationPlugin.h"
 * 
 *  @brief  Header file for hit-based shower compensation
 * 
 *  $Log: $
 */
#ifndef CMS_GLOBAL_HADRON_COMPENSATION_PLUGINS_H
#define CMS_GLOBAL_HADRON_COMPENSATION_PLUGINS_H 1

#include "Plugins/EnergyCorrectionsPlugin.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include <string>

namespace cms_content
{

/**
 *   @brief  GlobalHadronCompensation class. Correct cluster energy by looking at number of hits above 10 MIPs compared to the average energy density in HEF in MIPs
 */
class GlobalHadronCompensation : public pandora::EnergyCorrectionPlugin
{
 public:
  /**
   *  @brief  Default constructor
   */
  GlobalHadronCompensation();

  pandora::StatusCode MakeEnergyCorrections(const pandora::Cluster *const pCluster, float &correctedEnergy) const;

 private:
  /**
   *  @brief  Get the sum of the hadronic energies of all calo hits in a specified layer of an ordered calo hit list
   * 
   *  @param  orderedCaloHitList the ordered calo hit list
   *  @param  pseudoLayer the specified pseudolayer
   */
  float GetHadronicEnergyInLayer(const pandora::OrderedCaloHitList &orderedCaloHitList, const unsigned int pseudoLayer) const;

  /**
   * @brief return the residual hadronic scale as function of the sharing between ECAL and HCAL
   * @param totalEn total energy after pi/e corrections
   * @param backFracEn_m_mip energy fraction in HCAL
   */
  inline float getResidualScale(float totalEn, float backFracEn_m_mip) const
  {
    if(backFracEn_m_mip<0 || backFracEn_m_mip>1) return 1.0;
    if(totalEn==0) return 1.0;
   
    float p0( m_pioe_bananaParam_0.size()==3 ? m_pioe_bananaParam_0[0]+(m_pioe_bananaParam_0[1]*(1-exp(-(m_pioe_bananaParam_0[2]*totalEn)))) : 0 );
    float p1( m_pioe_bananaParam_1.size()==3 ? m_pioe_bananaParam_1[0]+(m_pioe_bananaParam_1[1]*(1-exp(-(m_pioe_bananaParam_1[2]*totalEn)))) : 0 );
    float p2( m_pioe_bananaParam_2.size()==3 ? m_pioe_bananaParam_2[0]+(m_pioe_bananaParam_2[1]*(1-exp(-(m_pioe_bananaParam_2[2]*totalEn)))) : 0 );

    float residualScale(p0+p1*backFracEn_m_mip+p2*backFracEn_m_mip*backFracEn_m_mip);
    return residualScale>0 ? 1./residualScale : 1.0;
  }

  /**
   * @brief returns the pi/e corrected energy
   * @param em_en energy registered in the subdetector subdet after scaling to e.m. 
   * @param totalEM_en total energy registered in the calorimeter after scaling to e.m. 
   * @param subdet the subdetector
   */
  inline float getByPiOverECorrectedEn(float em_en, float totalEM_en,int subdet) const
  {
    float p[5]={0,0,0,0,0};
    for(size_t i=0; i<5; i++)
      {
	if(subdet==ForwardSubdetector::HGCEE  && m_pioe_EE.size()==5) p[i]=m_pioe_EE[i];
	if(subdet==ForwardSubdetector::HGCHEF && m_pioe_FH.size()==5) p[i]=m_pioe_FH[i];
	if(subdet==ForwardSubdetector::HGCHEB && m_pioe_BH.size()==5) p[i]=m_pioe_BH[i];
      }
    float pioe(p[0]*(((1-exp(-(p[1]*totalEM_en)))/(1+exp(-(p[2]*totalEM_en))))*exp(-(p[3]*totalEM_en)))+p[4]);
    return pioe>0 ? em_en/pioe : em_en;
  }

  /**
   * @brief reads the settings from the configuration xml
   */
  pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

  float              m_nMIPsCut;
  float              m_IntegMIP_emEn_EE;
  float              m_e_em_EE,             m_e_em_FH,             m_e_em_BH;
  std::vector<float> m_pioe_EE,             m_pioe_FH,             m_pioe_BH;
  std::vector<float> m_pioe_bananaParam_0,  m_pioe_bananaParam_1,  m_pioe_bananaParam_2;

 };

} //cms_content

#endif
