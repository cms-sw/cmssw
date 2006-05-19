#ifndef Alignment_MuonAlignment_AlignableMuon_H
#define Alignment_MuonAlignment_AlignableMuon_H

#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"

// Classes that will be used to construct the muon
class AlignableMuBarrel;
class AlignableMuEndCap;

/// Constructor of the full muon geometry.
/// This object is stored to the EventSetup for further retrieval

class AlignableMuon: public AlignableComposite 
{

public:
  
  /// Constructor from record (builds the full hierarchy)
  AlignableMuon( const edm::Event& iEvent, const edm::EventSetup& iSetup ); 

  /// Destructor
  ~AlignableMuon();
  
public:

  // Some typdefs to simplify notation
  typedef GlobalPoint           _PositionType;
  typedef TkRotation<float>     _RotationType;
  typedef GeometricDet::ConstGeometricDetContainer _DetContainer;

  /// Recursive printout of the muon structure
  void dump( void ) const;
  
  /// Return all components
  virtual std::vector<Alignable*> components() const { return theMuonComponents; }


  std::vector<Alignable*> theDTChambers();

  std::vector<Alignable*> theDTStations();
  
  std::vector<Alignable*> theDTWheels();

  std::vector<Alignable*> theCSCChambers();
   
  std::vector<Alignable*> theCSCStations();
  


private:
  
  /// Get the position (centered at 0 by default)
  PositionType computePosition(); 
  /// Get the global orientation (no rotation by default)
  RotationType computeOrientation();
  /// Get the Surface
  AlignableSurface computeSurface();

   // Sub-structure builders 


  // Pointer to DTGeometry
  edm::ESHandle<DTGeometry> pDT;

  // Pointer to CSCGeometry
  edm::ESHandle<CSCGeometry> pCSC;

   // Build muon barrel
   void buildMuBarrel( edm::ESHandle<DTGeometry> pDD );

   // Build muon end caps
   void buildMUEndCap( edm::ESHandle<CSCGeometry> pDD );



  /// Return all components of a given type
//  std::vector<const GeometricDet*> getAllComponents( const GeometricDet* Det, const GeometricDet::GDEnumType type ) const;  


  // Container of all components
  std::vector<Alignable*> theMuonComponents;

  // Containers of separate components

  std::vector<AlignableDTChamber*>   theDTChambers;
  std::vector<AlignableDTStation*>   theStations;
  std::vector<AlignableDTWheel*>     theDTWheels;
  AlignableMuBarrel*                 theMuBarrel;
  
  std::vector<AlignableCSCChamber*>   theCSCChambers;
  std::vector<AlignableCSCStation*>   theCSCStations;
  std::vector<AlignableCSCStation*>   theMuEndCaps;


};

#endif //AlignableMuon_H




