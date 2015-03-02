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

  pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

  float           m_nMIPsCut;                ///< Min calo energy in MIPs to consider a hit for correction the

 };

} //cms_content

#endif
