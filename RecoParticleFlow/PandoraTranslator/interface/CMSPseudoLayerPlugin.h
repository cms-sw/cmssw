/**
 *  @file   CMSPseudoLayerPlugin.h
 * 
 *  @brief  Header file for the lc pseudo layer plugin class.
 * 
 *  $Log: $
 */
#ifndef CMS_PSEUDO_LAYER_PLUGIN_H
#define CMS_PSEUDO_LAYER_PLUGIN_H 1

#include "Pandora/PandoraInputTypes.h"

#include "Plugins/PseudoLayerPlugin.h"

namespace cms_content
{

/**
 *  @brief  CMSPseudoLayerPlugin class
 */
class CMSPseudoLayerPlugin : public pandora::PseudoLayerPlugin
{
public:
    /**
     *  @brief  Default constructor
     */
    CMSPseudoLayerPlugin();

private:
    pandora::StatusCode Initialize();
    unsigned int GetPseudoLayer(const pandora::CartesianVector &positionVector) const;
    unsigned int GetPseudoLayerAtIp() const;
    
    typedef std::vector<float> LayerPositionList;

    /**
     *  @brief  Find the layer number corresponding to a specified position, via reference to a specified layer position list
     * 
     *  @param  position the specified position
     *  @param  layerPositionList the specified layer position list
     *  @param  layer to receive the layer number
     */
    pandora::StatusCode FindMatchingLayer(const float position, const LayerPositionList &layerPositionList, unsigned int &layer) const;

    /**
     *  @brief  Store all revelevant barrel and endcap layer positions upon initialization
     */
    void StoreLayerPositions();

    /**
     *  @brief  Store subdetector layer positions upon initialization
     * 
     *  @param  subDetector the sub detector
     *  @param  layerParametersList the layer parameters list
     */
    void StoreLayerPositions(const pandora::SubDetector &subDetector, LayerPositionList &LayerPositionList);

    /**
     *  @brief  Store positions of barrel and endcap outer edges upon initialization
     */
    void StoreDetectorOuterEdge();
    
    typedef std::vector< std::pair<float, float> > AngleVector;
    
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    LayerPositionList       m_barrelLayerPositions;     ///< List of barrel layer positions
    LayerPositionList       m_endCapLayerPositions;     ///< List of endcap layer positions
    
    float                   m_barrelInnerEdgeR;         ///< Barrel inner edge r coordinate
    float                   m_barrelOuterEdgeR;         ///< Barrel outer edge r coordinate
    float                   m_endCapInnerEdgeZ;         ///< EndCap inner edge z coordinate
    float                   m_endCapOuterEdgeZ;         ///< Endcap outer edge z coordinate
    float                   m_endCapOuterEdgeR;         ///< Endcap outer edge r coordinate
};

//------------------------------------------------------------------------------------------------------------------------------------------

inline unsigned int CMSPseudoLayerPlugin::GetPseudoLayerAtIp() const
{
    const unsigned int pseudoLayerAtIp(this->GetPseudoLayer(pandora::CartesianVector(0.f, 0.f, 0.f)));
    return pseudoLayerAtIp;
}

} // namespace cms_content

#endif // #ifndef CMS_PSEUDO_LAYER_PLUGIN_H
