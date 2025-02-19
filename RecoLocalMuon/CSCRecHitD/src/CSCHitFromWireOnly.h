#ifndef CSCRecHitD_CSCHitFromWireOnly_h
#define CSCRecHitD_CSCHitFromWireOnly_h

/**
 * \class CSCHitFromWireOnly
 *
 * Search for hits within the wire groups.  For each of these hits, try forming clusters
 * by looking at adjacent wiregroup.  Form a wire Hit out of these clusters by finding 
 * the center-of-mass position of the hit in terms of the wire #.
 * The DetId, wire hit position, and peaking time are stored in a CSCWireHit collection.
 *
 *
 * To keep wire hits only so they can be used in segment building.
 * Only the DetId and wiregroup # are stored in a CSCWireHit collection
 *
 * \author Dominique Fortin - UCR
 * \author Stoyan Stoynev - Northwestern
 *
 */

#include <RecoLocalMuon/CSCRecHitD/src/CSCWireHit.h>

#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>

#include <RecoLocalMuon/CSCRecHitD/src/CSCRecoConditions.h>

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <vector>

class CSCLayer;
class CSCLayerGeometry;
class CSCDetId;


class CSCHitFromWireOnly 
{
 public:

  typedef std::vector<int> ChannelContainer;

  
  explicit CSCHitFromWireOnly( const edm::ParameterSet& ps );
  
  ~CSCHitFromWireOnly();
  
  std::vector<CSCWireHit> runWire( const CSCDetId& id, const CSCLayer* layer, const CSCWireDigiCollection::Range& rwired ); 
  void setConditions( const CSCRecoConditions* reco ) {
    recoConditions_ = reco;
  }
  void makeWireCluster(const CSCWireDigi& digi);
  bool addToCluster(const CSCWireDigi& digi); 
  float findWireHitPosition();
  
  CSCDetId id_;    
  const CSCLayer * layer_;
  const CSCLayerGeometry * layergeom_;
  
 private:
  bool isDeadWG(const CSCDetId& id, int WG); 

  std::vector<CSCWireDigi> wire_cluster;
  std::vector<int> wire_in_cluster;
  std::vector<float> wire_spacing;
  int theTime;
  int theLastChannel;
  std::vector<int> wire_in_clusterAndBX; /// To fill BX + wiregroup in CSCWireHit
  
  int deltaT;
  //int clusterSize;

  /// Hold pointer to current conditions data
  const CSCRecoConditions* recoConditions_;
};

#endif


