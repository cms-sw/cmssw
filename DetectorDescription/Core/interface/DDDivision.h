#ifndef DETECTOR_DESCRIPTION_CORE_DD_DIVISION_H
#define DETECTOR_DESCRIPTION_CORE_DD_DIVISION_H

// The following is based on G4PVDivision of Gean4 as of 4/2004
//
// The elements' positions are calculated by means of a simple
// linear formula.
// 
// G4PVDivision(const G4String& pName,
//                    G4LogicalVolume* pLogical,
//                    G4LogicalVolume* pMother,
//              const EAxis pAxis,
//              const G4int nReplicas,
//              const G4double width,
//              const G4double offset=0)
//
// Division may occur along:
//
// o Cartesian axes (kXAxis,kYAxis,kZAxis)
//
//   The divisions, of specified width have coordinates of
//   form (-width*(nReplicas-1)*0.5+n*width,0,0) where n=0.. nReplicas-1
//   for the case of kXAxis, and are unrotated.
//
// o Radial axis (cylindrical polar) (kRho)
//
//   The divisions are cons/tubs sections, centred on the origin
//   and are unrotated.
//   They have radii of width*n+offset to width*(n+1)+offset
//                      where n=0..nReplicas-1
//
// o Phi axis (cylindrical polar) (kPhi)
//   The divisions are `phi sections' or wedges, and of cons/tubs form
//   They have phi of offset+n*width to offset+(n+1)*width where
//   n=0..nReplicas-1

// GEANT4 History:
// 09.05.01 - P.Arce Initial version
//
// DDD History:
// 13.04.04 - M. Case Initial DDD version.
// ********************************************************************


//! A DDDivision contains the parameterization that Geant4 needs in order to do its divisions.
/** 
    A DDDivision simply holds the division information for Geant4 or other
    client software to recover.  The actual dividing of one solid into a set of
    identical shapes placed in different positions is performed in a
    DDAlgorithm which (in the default DDD/CMS way of running) is called by the 
    parser.  In other words, someone who wants to use this part of the DDD must 
    reproduce the algorithms (or in the case of Geant4, re-use) or use the 
    DDAlgorithm (i.e. load up the appropriate parameters and put run the DDAlgorithm.
*/

#include <iosfwd>
#include <map>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/Singleton.h"
#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"

class DDDivision;
class DDMaterial;
class DDPartSelection;
class DDSolid;

namespace DDI {
  class Division;
}

std::ostream & operator<<( std::ostream &, const DDDivision &);

class DDDivision : public DDBase<DDName, DDI::Division*>
{
 public:      
  
  //! The default constructor provides an uninitialzed reference object.
  DDDivision();

  //! Creates a refernce object referring to the appropriate XML specification.
  DDDivision(const DDName & name);
  
  //! Registers (creates) a reference object representing a Division
  /** ... Constructor with number of divisions and width
   */
  DDDivision(const DDName & name,
             const DDLogicalPart & parent,
	     DDAxes axis,
	     int nReplicas,
	     double width,
	     double offset );

  //! Registers (creates) a reference object representing a Division
  /** ...  Constructor with number of divisions 
   */
  DDDivision(const DDName & name,
	     const DDLogicalPart & parent,
	     DDAxes axis,
	     int nReplicas,
	     double offset );

    //! Registers (creates) a reference object representing a Division
  /** ...  Constructor with width
   */
  DDDivision(const DDName & name,
             const DDLogicalPart & parent,
	     DDAxes axis,
	     double width,
	     double offset );
  
  DDAxes axis() const;
  int nReplicas() const;
  double width() const;
  double offset() const;
  const DDLogicalPart & parent() const;

};

#endif
