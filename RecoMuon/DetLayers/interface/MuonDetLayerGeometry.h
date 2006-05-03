#ifndef DetLayers_MuonDetLayerGeometry_h
#define DetLayers_MuonDetLayerGeometry_h

/** \class MuonDetLayerGeometry
 *
 *  Provide access to the DetLayers of muon detectors.
 *
 *  $Date: 2006/05/02 10:23:54 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include <vector>

class DetLayer;

class MuonDetLayerGeometry {
 public:

  /// Constructor
  MuonDetLayerGeometry();

  friend class MuonDetLayerGeometryESProducer;  

  /// Destructor
  virtual ~MuonDetLayerGeometry();

  // FIXME: review which method return a reference!

  /// return the DT DetLayers (barrel)
  const std::vector<DetLayer*>& allDTLayers() const;

  /// return the CSC DetLayers (endcap)
  const std::vector<DetLayer*>& allCSCLayers() const;

  /// return the forward CSC DetLayers
  const std::vector<DetLayer*>& forwardCSCLayers() const;

  /// return the backward CSC DetLayers
  const std::vector<DetLayer*>& backwardCSCLayers() const;

  /// return all RPC DetLayers (barrel and endcap)
  const std::vector<DetLayer*>& allRPCLayers() const;

  /// return the barrel RPC DetLayers
  const std::vector<DetLayer*>& barrelRPCLayers() const;

  /// return the endcap RPC DetLayers
  const std::vector<DetLayer*>& endcapRPCLayers() const;

  /// return the forward RPC DetLayers
  const std::vector<DetLayer*>& forwardRPCLayers() const;

  /// return the backward RPC DetLayers
  const std::vector<DetLayer*>& backwardRPCLayers() const;

  /// return all layers (DT+CSC+RPC)
  const std::vector<DetLayer*> allLayers() const;

  /// return all barrel DetLayers (DT+RPC)
  const std::vector<DetLayer*> allBarrelLayers() const;

  /// return all endcap DetLayers (CSC+RPC)
  const std::vector<DetLayer*> allEndcapLayers() const;

  /// return all forward layers (CSC+RPC)
  const std::vector<DetLayer*> allForwardLayers() const;

  /// return all backward layers (CSC+RPC)
  const std::vector<DetLayer*> allBackwardLayers() const;  

 private:
    
    void addCSCLayers(std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > csclayers);
    void addDTLayers(std::vector<DetLayer*> dtlayers);
    //void addRPCLayers(std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > csclayers);
       
    std::vector<DetLayer*> cscLayers_fw;
    std::vector<DetLayer*> cscLayers_bk;
    std::vector<DetLayer*> cscLayers_all;
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
};
#endif

