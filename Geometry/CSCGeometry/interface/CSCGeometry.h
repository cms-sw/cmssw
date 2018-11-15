#ifndef CSCGeometry_CSCGeometry_h
#define CSCGeometry_CSCGeometry_h

/** \class CSCGeometry
 *
 *  The model of the geometry of the endcap muon CSC detectors.
 *
 *  \author Tim Cox
 */

#include <DataFormats/DetId/interface/DetId.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <Geometry/CommonDetUnit/interface/TrackingGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <vector>
#include <map>

class GeomDetType;

class CSCChamber;
class CSCChamberSpecs;
class CSCWireGroupPackage;

class CSCGeometry : public TrackingGeometry {

  typedef std::map<DetId, GeomDet*> CSCDetMap;
  // The buffer for specs need not really be a map. Could do it with a vector!
  typedef std::map<int, const CSCChamberSpecs*, std::less<int> > CSCSpecsContainer;

 public:

  typedef std::vector<const CSCChamber*> ChamberContainer;
  typedef std::vector<const CSCLayer*> LayerContainer;

  friend class CSCGeometryBuilder; //FromDDD;
  friend class GeometryAligner;

  /// Default constructor
  CSCGeometry();

  /// Real constructor
  CSCGeometry( bool debugV, bool gangedstripsME1a_, bool onlywiresME1a_, bool realWireGeometry_, bool useCentreTIOffsets_ );

  /// Destructor
  ~CSCGeometry() override;

  void clear();

  //---- Base class' interface

  // Return a vector of all det types
  const DetTypeContainer&  detTypes() const override;

  // Return a vector of all GeomDetUnit
  const DetContainer& detUnits() const override;

  // Return a vector of all GeomDet (including all GeomDetUnits)
  const DetContainer& dets() const override;
  
  // Return a vector of all GeomDetUnit DetIds
  const DetIdContainer&    detUnitIds() const override;

  // Return a vector of all GeomDet DetIds (including those of GeomDetUnits)
  const DetIdContainer& detIds() const override;

  // Return the pointer to the GeomDetUnit corresponding to a given DetId
  const GeomDet* idToDetUnit(DetId) const override;

  // Return the pointer to the GeomDet corresponding to a given DetId
  const GeomDet* idToDet(DetId) const override;

  //---- Extension of the interface

  /// Return the chamber corresponding to given DetId
  const CSCChamber* chamber(CSCDetId id) const;

  /// Return the layer corresponding to given DetId
  const CSCLayer* layer(CSCDetId id) const;

  /// Return a vector of all chambers
  const ChamberContainer& chambers() const;

  /// Return a vector of all layers
  const LayerContainer& layers() const;



  /**
   * Return the CSCChamberSpecs* for given chamber type
   * if it exists, or 0 if it has not been created.
   */
  const CSCChamberSpecs* findSpecs( int iChamberType );

  /**
   * Build CSCChamberSpecs for given chamber type.
   *
   * @@ a good candidate to be replaced by a factory?
   */
  const CSCChamberSpecs* buildSpecs( int iChamberType,
				 const std::vector<float>& fpar,
				 const std::vector<float>& fupar,
				 const CSCWireGroupPackage& wg );

  void setGangedStripsInME1a(bool gs) { gangedstripsME1a_ = gs; }
  void setOnlyWiresInME1a(bool ow) { onlywiresME1a_ = ow; }
  void setUseRealWireGeometry(bool rwg) { realWireGeometry_ = rwg; }
  void setUseCentreTIOffsets(bool cti) { useCentreTIOffsets_ = cti; }
  void setDebugV(bool dbgv) { debugV_ = dbgv; }

  /**
   * Ganged strips in ME1a
   */
  bool gangedStrips() const { return gangedstripsME1a_; }

  /**
   * Wires only in ME1a
   */
  bool wiresOnly() const { return onlywiresME1a_; }

  /**
   * Wire geometry modelled as real hardware (complex
   * groupings of wires and dead regions) or as a pseudo
   * geometry with just one wire grouping per chamber type
   * (as was done in ORCA versions up to and including ORCA_8_8_1).
   *
   */
  bool realWireGeometry() const { return realWireGeometry_; }

  /**
   * Use the backed-out offsets for theCentreToIntersection in
   * CSCLayerGeometry
   */
  bool centreTIOffsets() const { return useCentreTIOffsets_; }

  /// Dump parameters for overall strip and wire modelling
  void queryModelling() const;

 private:

  /// deallocate managed memory
  void deallocate();

  /// Add a chamber with given DetId.
  void addChamber(CSCChamber* ch);
  
  /// Add a DetUnit
  void addLayer(CSCLayer* l);

  /// Add a DetType
  void addDetType(GeomDetType* type);

  /// Add a DetId
  void addDetId(DetId id);

  /// Add a GeomDet; not to be called by the builder.
  void addDet(GeomDet* det);

  // The chambers are owned by the geometry (which in turn own layers)
  ChamberContainer  theChambers; 

  // Map for efficient lookup by DetId 
  CSCDetMap         theMap;

  // These are used rarely; they could be computed at runtime 
  // to save memory.
  DetTypeContainer  theDetTypes;
  DetContainer      theDets;       // all dets (chambers and layers)
  DetContainer      theDetUnits;   // all layers
  DetIdContainer    theDetIds;
  DetIdContainer    theDetUnitIds;

  // These are reduntant copies, to satisfy the interface.
  LayerContainer    theLayers;

  // Parameters controlling modelling of geometry 

  bool debugV_; // for debug printout etc.

  bool gangedstripsME1a_;
  bool onlywiresME1a_;
  bool realWireGeometry_;
  bool useCentreTIOffsets_;

  // Store pointers to Specs objects as we build them.
  CSCSpecsContainer specsContainer;

};

#endif

