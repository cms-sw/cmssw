/**
 *  @file   LCContent/include/LCPlugins/LCBFieldPlugin.h
 * 
 *  @brief  Header file for the lc bfield plugin class.
 * 
 *  $Log: $
 */

#ifndef CMS_BFIELD_PLUGIN_H
#define CMS_BFIELD_PLUGIN_H 1

#include "Plugins/BFieldPlugin.h"

namespace cms_content
{

/**
 *  @brief  CMSBFieldPlugin class
 */
class CMSBFieldPlugin : public pandora::BFieldPlugin
{
public:
    /**
     *  @brief  Default constructor
     * 
     *  @param  innerBField the bfield in the main tracker, ecal and hcal, units Tesla     
     */
    CMSBFieldPlugin(const float innerBField = 3.8f);

    float GetBField(const pandora::CartesianVector &positionVector) const;

private:
    pandora::StatusCode Initialize();
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    float   m_innerBField;              ///< The bfield in the main tracker, ecal and hcal, units Tesla
 };

} // namespace CMS_content

#endif // #ifndef CMS_BFIELD_CALCULATOR_H
