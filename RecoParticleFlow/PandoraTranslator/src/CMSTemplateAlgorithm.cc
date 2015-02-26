/**
 *  @file    AthenaPandora/src/TemplateAlgorithm.cxx
 * 
 *  @brief  Implementation of the template algorithm class.
 * 
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"
#include "Pandora/TemplateAlgorithm.h"

#include "RecoParticleFlow/PandoraTranslator/interface/CMSTemplateAlgorithm.h"

using namespace pandora;

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CMSTemplateAlgorithm::Run()
{
    // Algorithm code here

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CMSTemplateAlgorithm::ReadSettings(const TiXmlHandle /*xmlHandle*/)
{
    // Read settings from xml file here

    return STATUS_CODE_SUCCESS;
}
