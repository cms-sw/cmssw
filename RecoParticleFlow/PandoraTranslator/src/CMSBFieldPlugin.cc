/**
 *  @file   LCContent/src/LCPlugins/LCBFieldPlugin.cc
 * 
 *  @brief  Implementation of the lc bfield plugin class.
 * 
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "RecoParticleFlow/PandoraTranslator/interface/CMSBFieldPlugin.h"

using namespace pandora;

namespace cms_content
{

CMSBFieldPlugin::CMSBFieldPlugin(const float innerBField) :
    m_innerBField(innerBField)   
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

float CMSBFieldPlugin::GetBField(const CartesianVector &positionVector) const
{
    return m_innerBField;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CMSBFieldPlugin::Initialize()
{
    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CMSBFieldPlugin::ReadSettings(const TiXmlHandle /*xmlHandle*/)
{
    return STATUS_CODE_SUCCESS;
}

} // namespace cms_content
