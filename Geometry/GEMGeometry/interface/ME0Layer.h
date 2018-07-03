#ifndef Geometry_GEMGeometry_ME0Layer_h
#define Geometry_GEMGeometry_ME0Layer_h

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"

class ME0EtaPartition;

class ME0Layer : public GeomDet {
public:
  /// Constructor
  ME0Layer(ME0DetId id, const ReferenceCountingPointer<BoundPlane>& plane);

  /// Destructor
  ~ME0Layer() override;

  /// Return the ME0DetId of this layer
  ME0DetId id() const;

  // Which subdetector
  SubDetector subDetector() const override {return GeomDetEnumerators::ME0;}

  /// equal if the id is the same
  bool operator==(const ME0Layer& ch) const;

  /// Add EtaPartition to the layer which takes ownership
  void add(const ME0EtaPartition* roll);

  /// Return the rolls in the layer
  std::vector<const GeomDet*> components() const override;

  /// Return the sub-component (roll) with a given id in this layer
  const GeomDet* component(DetId id) const override;

  /// Return the eta partition corresponding to the given id 
  const ME0EtaPartition* etaPartition(ME0DetId id) const;

  const ME0EtaPartition* etaPartition(int isl) const;
  
  /// Return the eta partitions
  const std::vector<const ME0EtaPartition*>& etaPartitions() const;

  /// Retunr numbers of eta partitions
  int nEtaPartitions() const;

private:

  ME0DetId detId_;

  // vector of eta partitions for a layer
  std::vector<const ME0EtaPartition*> etaPartitions_;

};
#endif
