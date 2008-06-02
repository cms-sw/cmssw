#ifndef Demo_PFClusterAlgo_PFLayer_h
#define Demo_PFClusterAlgo_PFLayer_h

/**\class PFLayer
   \brief layer definition for PFRecHit and PFCluster

   \author Colin Bernet
   \date   July 2006
*/
class PFLayer {

 public:
  /// constructor
  PFLayer();

  /// destructor
  virtual ~PFLayer() = 0;

  /// layer definition
  enum Layer {PS2 = -12, 
              PS1 = -11,
              ECAL_ENDCAP = -2,
              ECAL_BARREL = -1,
              HCAL_BARREL1 = 1,
              HCAL_BARREL2 = 2,
              HCAL_ENDCAP = 3,
              VFCAL = 11 };
};

#endif
