#ifndef Alignment_MuonAlignment_AlignableMuon_H
#define Alignment_MuonAlignment_AlignableMuon_H

/** \class AlignableMuon
 *  The alignable muon.
 *
 *  $Date: 2008/04/25 21:23:15 $
 *  $Revision: 1.21 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include <DataFormats/GeometryVector/interface/GlobalPoint.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

class CSCGeometry;

// Classes that will be used to construct the muon
class AlignableDTBarrel;
class AlignableDTWheel;
class AlignableDTStation;
class AlignableDTChamber;
class AlignableCSCEndcap;
class AlignableCSCStation;
class AlignableCSCRing;
class AlignableCSCChamber;




/// Constructor of the full muon geometry.

class AlignableMuon: public AlignableComposite 
{

public:

  /// Constructor from geometries
  AlignableMuon( const DTGeometry* , const CSCGeometry* );

  /// Destructor
  ~AlignableMuon();
  
  /// Updater using DTGeometry and CSCGeometry.
  /// The given geometries have to match the current ones.
  void update(const DTGeometry* , const CSCGeometry*);

  /// Return all components
  virtual align::Alignables components() const { return theMuonComponents; }

  /// Alignable tracker has no mother
  virtual Alignable* mother() { return 0; }

  /// Methods to return specific of components
  align::Alignables DTLayers();
  align::Alignables DTSuperLayers();
  align::Alignables DTChambers();
  align::Alignables DTStations();
  align::Alignables DTWheels();
  align::Alignables DTBarrel();
  align::Alignables CSCLayers();
  align::Alignables CSCChambers();
  align::Alignables CSCStations();
  align::Alignables CSCRings();
  align::Alignables CSCEndcaps();

  /// Get DT alignments sorted by DetId
  Alignments* dtAlignments();

  /// Get DT alignment errors sorted by DetId
  AlignmentErrorsExtended* dtAlignmentErrorsExtended();

  /// Get CSC alignments sorted by DetId
  Alignments* cscAlignments();

  /// Get CSC alignment errors sorted by DetId
  AlignmentErrorsExtended* cscAlignmentErrorsExtended();


  /// Return muon alignable object ID provider derived from the muon system geometry
  const AlignableObjectId& objectIdProvider() const { return alignableObjectId_; }

private:
  
  /// Get the position (centered at 0 by default)
  PositionType computePosition(); 

  /// Get the global orientation (no rotation by default)
  RotationType computeOrientation();

  /// Get the Surface
  AlignableSurface computeSurface();

  /// Get alignments sorted by DetId
  Alignments* alignments() const;

  /// Get alignment errors sorted by DetId
  AlignmentErrorsExtended* alignmentErrors() const;



  // Sub-structure builders

  /// Build muon barrel
  void buildDTBarrel(const DTGeometry*, bool update = false);

  /// Build muon end caps
  void buildCSCEndcap(const CSCGeometry*, bool update = false);

  /// Set mothers recursively
  void recursiveSetMothers( Alignable* alignable );

  /// alignable object ID provider
  const AlignableObjectId alignableObjectId_;

  /// Containers of separate components
  std::vector<AlignableDTChamber*>   theDTChambers;
  std::vector<AlignableDTStation*>   theDTStations;
  std::vector<AlignableDTWheel*>     theDTWheels;
  std::vector<AlignableDTBarrel*>    theDTBarrel;
  
  std::vector<AlignableCSCChamber*>  theCSCChambers;
  std::vector<AlignableCSCStation*>  theCSCStations;
  std::vector<AlignableCSCRing*>     theCSCRings;
  std::vector<AlignableCSCEndcap*>   theCSCEndcaps;

  align::Alignables theMuonComponents;

};

#endif //AlignableMuon_H

