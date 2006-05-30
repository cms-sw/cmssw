#ifndef Alignment_MuonAlignment_AlignableMuon_H
#define Alignment_MuonAlignment_AlignableMuon_H

#include <vector>

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/Vector/interface/GlobalPoint.h>
#include <Geometry/DTGeometry/interface/DTLayer.h>
#include <Geometry/DTGeometry/interface/DTSuperLayer.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

//#include "Alignment/MuonAlignment/interface/AlignableDTBarrel.h"
//#include "Alignment/MuonAlignment/interface/AlignableDTWheel.h"
//#include "Alignment/MuonAlignment/interface/AlignableDTStation.h"
//#include "Alignment/MuonAlignment/interface/AlignableDTChamber.h"
//#include "Alignment/MuonAlignment/interface/AlignableCSCEndcap.h"
//#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"
//#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"


#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"

// Classes that will be used to construct the muon
class AlignableDTBarrel;
class AlignableDTWheel;
class AlignableDTStation;
class AlignableDTChamber;
class AlignableCSCEndcap;
class AlignableCSCStation;
class AlignableCSCChamber;




/// Constructor of the full muon geometry.
/// This object is stored to the EventSetup for further retrieval

class AlignableMuon: public AlignableComposite 
{

public:
  
  /// Constructor from record (builds the full hierarchy)
  AlignableMuon( const edm::EventSetup&  ); 

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

  // Methods to return specific of components
  std::vector<AlignableDTChamber*> DTChambers();
  std::vector<AlignableDTStation*> DTStations();
  std::vector<AlignableDTWheel*> DTWheels();
  AlignableDTBarrel* DTBarrel();
  std::vector<AlignableCSCChamber*> CSCChambers();
  std::vector<AlignableCSCStation*> CSCStations();
  std::vector<AlignableCSCEndcap*> CSCEndcaps();


private:
  
  /// Get the position (centered at 0 by default)
  PositionType computePosition(); 
  /// Get the global orientation (no rotation by default)
  RotationType computeOrientation();
  /// Get the Surface
  AlignableSurface computeSurface();

  /// Return alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableMuon; }

   // Sub-structure builders 


  // Pointer to DTGeometry
  edm::ESHandle<DTGeometry> pDT;

  // Pointer to CSCGeometry
  edm::ESHandle<CSCGeometry> pCSC;

   // Build muon barrel
   void buildDTBarrel( edm::ESHandle<DTGeometry> pDD );

   // Build muon end caps
   void buildCSCEndcap( edm::ESHandle<CSCGeometry> pDD );



  /// Return all components of a given type
//  std::vector<const GeometricDet*> getAllComponents( const GeometricDet* Det, const GeometricDet::GDEnumType type ) const;  



  // Containers of separate components

  std::vector<AlignableDTChamber*>   theDTChambers;
  std::vector<AlignableDTStation*>   theDTStations;
  std::vector<AlignableDTWheel*>     theDTWheels;
  AlignableDTBarrel*                 theDTBarrel;
  
  std::vector<AlignableCSCChamber*>   theCSCChambers;
  std::vector<AlignableCSCStation*>   theCSCStations;
  std::vector<AlignableCSCEndcap*>    theCSCEndcaps;

  std::vector<Alignable*> theMuonComponents;

};

#endif //AlignableMuon_H




