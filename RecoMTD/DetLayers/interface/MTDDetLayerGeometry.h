#ifndef RecoMTD_DetLayers_MTDDetLayerGeometry_h
#define RecoMTD_DetLayers_MTDDetLayerGeometry_h

/** \class MTDDetLayerGeometry
 *
 *  Provide access to the DetLayers of mip timing detectors.
 *
 *  \author L. Gray - FNAL
 *  
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"
#include <vector>
#include <map>

class DetLayer;

class MTDDetLayerGeometry : public DetLayerGeometry {
public:
  /// Constructor
  MTDDetLayerGeometry();

  friend class MTDDetLayerGeometryESProducer;

  /// Destructor
  ~MTDDetLayerGeometry() override;

  /// return all barrel layers
  const std::vector<const DetLayer*>& allBarrelLayers() const;

  /// return all endcap layers
  const std::vector<const DetLayer*>& allEndcapLayers() const;

  /// return all endcap layers
  const std::vector<const DetLayer*>& allForwardLayers() const;

  /// return all endcap layers
  const std::vector<const DetLayer*>& allBackwardLayers() const;

  /// return the BTL DetLayers (barrel), inside-out
  const std::vector<const DetLayer*>& allBTLLayers() const;

  /// return the ETL DetLayers (endcap), -Z to +Z
  const std::vector<const DetLayer*>& allETLLayers() const;

  /// return all DetLayers (barrel + endcap), -Z to +Z
  const std::vector<const DetLayer*>& allLayers() const;

  /// return the DetLayer which correspond to a certain DetId
  const DetLayer* idToLayer(const DetId& detId) const override;

private:
  /// Add ETL layers
  /// etllayers.first=forward (+Z), etllayers.second=backward (-Z)
  /// both vectors are ASSUMED to be sorted inside-out
  void addETLLayers(const std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> >& etllayers);

  //. Add BTL layers; dtlayers is ASSUMED to be sorted inside-out
  void addBTLLayers(const std::vector<DetLayer*>& btllayers);

  DetId makeDetLayerId(const DetLayer* detLayer) const;

  void sortLayers();

  std::vector<const DetLayer*> etlLayers_fw;
  std::vector<const DetLayer*> etlLayers_bk;
  std::vector<const DetLayer*> etlLayers_all;

  //////////////////////////////
  std::vector<const DetLayer*> btlLayers;
  std::vector<const DetLayer*> allForward;
  std::vector<const DetLayer*> allBackward;
  std::vector<const DetLayer*> allEndcap;
  std::vector<const DetLayer*> allBarrel;
  std::vector<const DetLayer*> allDetLayers;

  std::map<DetId, const DetLayer*> detLayersMap;
};
#endif
