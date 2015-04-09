#ifndef DetLayers_MuonDetLayerGeometry_h
#define DetLayers_MuonDetLayerGeometry_h

/** \class MuonDetLayerGeometry
 *
 *  Provide access to the DetLayers of muon detectors.
 *
 *  \author N. Amapane - CERN
 *  \modified by R. Radogna & C. Calabria
 *  \modified by D. Nash
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"
#include <vector>
#include <map>

class DetLayer;

class MuonDetLayerGeometry : public DetLayerGeometry{
 public:

  /// Constructor
  MuonDetLayerGeometry();

  friend class MuonDetLayerGeometryESProducer;  

  /// Destructor
  virtual ~MuonDetLayerGeometry();

  /// return the DT DetLayers (barrel), inside-out
  const std::vector<DetLayer*>& allDTLayers() const;

  /// return the CSC DetLayers (endcap), -Z to +Z
  const std::vector<DetLayer*>& allCSCLayers() const;

  /// return the forward (+Z) CSC DetLayers, inside-out
  const std::vector<DetLayer*>& forwardCSCLayers() const;

  /// return the backward (-Z) CSC DetLayers, inside-out
  const std::vector<DetLayer*>& backwardCSCLayers() const;

/////////////////////////////// GEMs

  /// return the GEM DetLayers (endcap), -Z to +Z
  const std::vector<DetLayer*>& allGEMLayers() const;

  /// return the forward (+Z) GEM DetLayers, inside-out
  const std::vector<DetLayer*>& forwardGEMLayers() const;

  /// return the backward (-Z) GEM DetLayers, inside-out
  const std::vector<DetLayer*>& backwardGEMLayers() const;

//////////////////////////////


/////////////////////////////// ME0s

  /// return the ME0 DetLayers (endcap), -Z to +Z
  const std::vector<DetLayer*>& allME0Layers() const;

  /// return the forward (+Z) ME0 DetLayers, inside-out
  const std::vector<DetLayer*>& forwardME0Layers() const;

  /// return the backward (-Z) ME0 DetLayers, inside-out
  const std::vector<DetLayer*>& backwardME0Layers() const;

//////////////////////////////

  /// return all RPC DetLayers, order: backward, barrel, forward
  const std::vector<DetLayer*>& allRPCLayers() const;

  /// return the barrel RPC DetLayers, inside-out
  const std::vector<DetLayer*>& barrelRPCLayers() const;

  /// return the endcap RPC DetLayers, -Z to +Z
  const std::vector<DetLayer*>& endcapRPCLayers() const;

  /// return the forward (+Z) RPC DetLayers, inside-out
  const std::vector<DetLayer*>& forwardRPCLayers() const;

  /// return the backward (-Z) RPC DetLayers, inside-out
  const std::vector<DetLayer*>& backwardRPCLayers() const;

  /// return all layers (DT+CSC+RPC), order: backward, barrel, forward
  const std::vector<DetLayer*>& allLayers() const;

  /// return all barrel DetLayers (DT+RPC), inside-out
  const std::vector<DetLayer*>& allBarrelLayers() const;

  /// return all endcap DetLayers (CSC+RPC+GEM+ME0), -Z to +Z
  const std::vector<DetLayer*>& allEndcapLayers() const;

  /// return all forward (+Z) layers (CSC+RPC+GEM+ME0), inside-out
  const std::vector<DetLayer*>& allForwardLayers() const;

  /// return all backward (-Z) layers (CSC+RPC+GEM+ME0), inside-out
  const std::vector<DetLayer*>& allBackwardLayers() const;

/////////////////////////////// GEMs

  /// return all endcap DetLayers (CSC+GEM), -Z to +Z
  const std::vector<DetLayer*>& allEndcapCscGemLayers() const;

  /// return all endcap DetLayers (CSC+GEM), -Z to +Z
  const std::vector<DetLayer*>& allCscGemForwardLayers() const;

  /// return all endcap DetLayers (CSC+GEM), -Z to +Z
  const std::vector<DetLayer*>& allCscGemBackwardLayers() const;

//////////////////////////////


/////////////////////////////// ME0s

  /// return all endcap DetLayers (CSC+ME0), -Z to +Z
  const std::vector<DetLayer*>& allEndcapCscME0Layers() const;

  /// return all endcap DetLayers (CSC+ME0), -Z to +Z
  const std::vector<DetLayer*>& allCscME0ForwardLayers() const;

  /// return all endcap DetLayers (CSC+ME0), -Z to +Z
  const std::vector<DetLayer*>& allCscME0BackwardLayers() const;

//////////////////////////////
  
  /// return the DetLayer which correspond to a certain DetId
  virtual const DetLayer* idToLayer(const DetId& detId) const;

 private:
  /// Add CSC layers 
  /// csclayers.first=forward (+Z), csclayers.second=backward (-Z)
  /// both vectors are ASSUMED to be sorted inside-out
  void addCSCLayers(std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > csclayers);

  //. Add DT layers; dtlayers is ASSUMED to be sorted inside-out
  void addDTLayers(std::vector<DetLayer*> dtlayers);

  /// Add RPC layers
  /// endcapRPCLayers.first=forward (+Z), endcapRPCLayers.second=backward (-Z)
  /// All three vectors are ASSUMED to be sorted inside-out
  void addRPCLayers(std::vector<DetLayer*> barrelRPCLayers, std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > endcapRPCLayers);

/////////////////////////////// GEMs

  /// Add GEM layers 
  /// gemlayers.first=forward (+Z), gemlayers.second=backward (-Z)
  /// both vectors are ASSUMED to be sorted inside-out
  void addGEMLayers(std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > gemlayers);

//////////////////////////////


/////////////////////////////// ME0s

  /// Add ME0 layers 
  /// gemlayers.first=forward (+Z), gemlayers.second=backward (-Z)
  /// both vectors are ASSUMED to be sorted inside-out
  void addME0Layers(std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > gemlayers);

//////////////////////////////

  
  DetId makeDetLayerId(const DetLayer* detLayer) const;
  
  void sortLayers();

  std::vector<DetLayer*> cscLayers_fw;
  std::vector<DetLayer*> cscLayers_bk;
  std::vector<DetLayer*> cscLayers_all;

/////////////////////////////// GEMs

  std::vector<DetLayer*> gemLayers_fw;
  std::vector<DetLayer*> gemLayers_bk;
  std::vector<DetLayer*> gemLayers_all;

//////////////////////////////


/////////////////////////////// ME0s

  std::vector<DetLayer*> me0Layers_fw;
  std::vector<DetLayer*> me0Layers_bk;
  std::vector<DetLayer*> me0Layers_all;

//////////////////////////////

  std::vector<DetLayer*> rpcLayers_all;
  std::vector<DetLayer*> rpcLayers_endcap;
  std::vector<DetLayer*> rpcLayers_fw;
  std::vector<DetLayer*> rpcLayers_bk;
  std::vector<DetLayer*> rpcLayers_barrel;
  std::vector<DetLayer*> dtLayers;
  std::vector<DetLayer*> allForward;
  std::vector<DetLayer*> allBackward;
  std::vector<DetLayer*> allEndcap;
  std::vector<DetLayer*> allBarrel;
  std::vector<DetLayer*> allDetLayers;

/////////////////////////////// GEMs

  std::vector<DetLayer*> allEndcapCscGem;
  std::vector<DetLayer*> allCscGemForward;
  std::vector<DetLayer*> allCscGemBackward;

//////////////////////////////

/////////////////////////////// ME0s

  std::vector<DetLayer*> allEndcapCscME0;
  std::vector<DetLayer*> allCscME0Forward;
  std::vector<DetLayer*> allCscME0Backward;

//////////////////////////////
    
  std::map<DetId,DetLayer*> detLayersMap;
};
#endif

