/**
 *  @file   CMSPandora/src/TemplateAlgorithm.h
 * 
 *  @brief  Header file for the template algorithm class.
 * 
 *  $Log: $
 */
#ifndef CMS_PANDORA_TEMPLATE_ALGORITHM_H
#define CMS_PANDORA_TEMPLATE_ALGORITHM_H 1

#include "Pandora/Algorithm.h"

/**
 *  @brief  CMSTemplateAlgorithm class
 */
class CMSTemplateAlgorithm : public pandora::Algorithm
{
 public:
 
  /**
   *  @brief  Factory class for instantiating algorithm
   */
  class Factory : public pandora::AlgorithmFactory
    {
    public:
      pandora::Algorithm *CreateAlgorithm() const;
    };

 private:
  pandora::StatusCode Run();
  pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

  // Member variables here
};

//------------------------------------------------------------------------------------------------------------------------------------------

inline pandora::Algorithm *CMSTemplateAlgorithm::Factory::CreateAlgorithm() const
{
  return new CMSTemplateAlgorithm();
}

#endif 
