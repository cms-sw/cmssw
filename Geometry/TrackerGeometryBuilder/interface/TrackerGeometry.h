#ifndef Geometry_TrackerGeometryBuilder_TrackerGeometry_H
#define Geometry_TrackerGeometryBuilder_TrackerGeometry_H

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"

class GeometricDet;

#include "Geometry/TrackerGeometryBuilder/interface/trackerHierarchy.h"

#include "DataFormats/Common/interface/Trie.h"

// FIXME here just to allow prototyping...
namespace trackerTrie {
  typedef GeomDet const* PDet;
  typedef edm::Trie<PDet> DetTrie;
  typedef edm::TrieNode<PDet> Node;
  typedef Node const * node_pointer; // sigh....
  typedef edm::TrieNodeIter<PDet> node_iterator;
}


/**
 * A specific Tracker Builder which builds a Tracker from a list of DetUnits. 
 * Pattern recognition is used to discover layers, rings etc.
 */
class TrackerGeometry final : public TrackingGeometry {

  explicit TrackerGeometry(GeometricDet const* gd=0);  

  friend class TrackerGeomBuilderFromGeometricDet;

  void addType(GeomDetType const * p);
  void addDetUnit(GeomDetUnit const * p);
  void addDetUnitId(DetId p);
  void addDet(GeomDet const * p);
  void addDetId(DetId p);
  void finalize();

public:
  typedef GeomDetEnumerators::SubDetector SubDetector;

  virtual ~TrackerGeometry() ;


  virtual const DetTypeContainer&  detTypes()         const;
  virtual const DetUnitContainer&  detUnits()         const;
  virtual const DetContainer&      dets()             const;
  virtual const DetIdContainer&    detUnitIds()       const;
  virtual const DetIdContainer&    detIds()           const;
  virtual const TrackerGeomDet*    idToDetUnit(DetId) const;
  virtual const TrackerGeomDet*    idToDet(DetId)     const;

  const GeomDetEnumerators::SubDetector geomDetSubDetector(int subdet) const;
  unsigned int numberOfLayers(int subdet) const;
  bool isThere(GeomDetEnumerators::SubDetector subdet) const;

  unsigned int offsetDU(SubDetector sid) const { return theOffsetDU[sid];}
  unsigned int endsetDU(SubDetector sid) const { return theEndsetDU[sid];}
  // Magic : better be called at the right moment...
  void setOffsetDU(SubDetector sid) { theOffsetDU[sid]=detUnits().size();}
  void setEndsetDU(SubDetector sid) { theEndsetDU[sid]=detUnits().size();}

  GeometricDet const * trackerDet() const {return  theTrackerDet;}

  const DetContainer& detsPXB() const;
  const DetContainer& detsPXF() const;
  const DetContainer& detsTIB() const;
  const DetContainer& detsTID() const;
  const DetContainer& detsTOB() const;
  const DetContainer& detsTEC() const;

private:

  GeometricDet const * theTrackerDet; 

  /// Aligner has access to map
  friend class GeometryAligner;

  DetTypeContainer  theDetTypes;  // owns the DetTypes
  DetUnitContainer  theDetUnits;  // they're all also into 'theDets', so we assume 'theDets' owns them
  unsigned int      theOffsetDU[6]; // offsets in the above
  unsigned int      theEndsetDU[6]; // end offsets in the above
  DetContainer      theDets;      // owns *ONLY* the GeomDet * corresponding to GluedDets.
  DetIdContainer    theDetUnitIds;
  DetIdContainer    theDetIds; 
  mapIdToDetUnit    theMapUnit; // does not own GeomDetUnit *
  mapIdToDet        theMap;     // does not own GeomDet *

  DetContainer      thePXBDets; // not owned: they're also in 'theDets'
  DetContainer      thePXFDets; // not owned: they're also in 'theDets'
  DetContainer      theTIBDets; // not owned: they're also in 'theDets'
  DetContainer      theTIDDets; // not owned: they're also in 'theDets'
  DetContainer      theTOBDets; // not owned: they're also in 'theDets'
  DetContainer      theTECDets; // not owned: they're also in 'theDets'

  GeomDetEnumerators::SubDetector theSubDetTypeMap[6];
  unsigned int theNumberOfLayers[6];

};

#endif
