#ifndef DDSolid_h
#define DDSolid_h

#include <iosfwd>
#include <vector>

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"

namespace DDI { class Solid; }
namespace DDI { class Reflection; }
namespace DDI { class BooleanSolid; }

class DDSolid;
struct DDSolidFactory;
class DDStreamer;

std::ostream & operator<<( std::ostream &, const DDSolid & );

//! A DDSolid represents the shape of a part.  
/** An object of this class is a reference-object and thus is a lightweight
    class. It can be copied by value without having a large overhead.
    Assignment to the reference-object invalidates the object to which it was 
    referred.  Assignment also affects all other instances of this class which 
    were created using the same value of DDName. In fact, the value of DDName
    identifies a DDSolid uniquely.
    
    For further details concerning the usage of reference-objects refer
    to the documentation of DDLogicalPart.   

*/
class DDSolid : public DDBase<DDName, DDI::Solid*>
{
  friend std::ostream & operator<<( std::ostream &, const DDSolid & );
  friend struct DDSolidFactory;
  friend class DDDToPersFactory;
  friend class DDPersToDDDFactory;
  friend class DDStreamer;
    
public: 
  //! Uninitialilzed solid reference-object; for further details on reference-objects see documentation of DDLogicalPart
  DDSolid( void );
  
  //! Creates a reference-object to a solid named \a name
  /** If the solid was not yet created using one of the solid generating factory 
      functions \c DDbox(), \c DDtub , ... this constructor creates a (default) 
      initialized reference object named \a name. It can be used as placeholder 
      everywhere and becomes a reference to a valid solid as soon as one of the 
      factory functions \c DDBox, ... has been called (using the same value for DDName).
      
      For further details concerning reference-objects refer to the documentation of DDLogicalPart.       
  */
  DDSolid( const DDName & name ); 
  
  //! Give the parameters of the solid
  const std::vector<double> & parameters( void ) const;
  
  //! Returns the volume of the given solid (\b does \b not \b work \b with \b boolean \b soids \b !)        
  double volume( void ) const;
  
  //! The type of the solid
  DDSolidShape shape( void ) const;
  
private:
  DDSolid( const DDName &, DDI::Solid * );    
  DDSolid( const DDName &, DDSolidShape, const std::vector<double> & );
};

//! Interface to a Trapezoid
/**
   The definition (parameters, local frame) of the Trapezoid is 
   the same than in Geant4.
*/
class DDTrap : public DDSolid
{
public:
  DDTrap( const DDSolid & s );
  //! half of the z-Axis
  double halfZ( void ) const;
  //! Polar angle of the line joining the centres of the faces at -/+pDz
  double theta( void ) const;
  //! Azimuthal angle of the line joining the centres of the faces at -/+pDz
  double phi( void ) const;
  //! Half-length along y of the face at -pDz
  double y1( void ) const;
  //! Half-length along x of the side at y=-pDy1 of the face at -pDz
  double x1( void ) const;
  //! Half-length along x of the side at y=+pDy1 of the face at -pDz
  double x2( void ) const;
  //! Angle with respect to the y axis from the centre of the side at y=-pDy1 to the centre at y=+pDy1 of the face at -pDz
  double alpha1( void ) const;
  //! Half-length along y of the face at +pDz
  double y2( void ) const;
  //! Half-length along x of the side at y=-pDy2 of the face at +pDz
  double x3( void ) const;
  //! Half-length along x of the side at y=+pDy2 of the face at +pDz
  double x4( void ) const;
  //! Angle with respect to the y axis from the centre of the side at y=-pDy2 to the centre at y=+pDy2 of the face at +pDz
  double alpha2( void ) const;
  
private:
  DDTrap( void );
};

class DDPseudoTrap : public DDSolid
{
public:
  DDPseudoTrap( const DDSolid & s );
  //! half of the z-Axis
  double halfZ( void ) const;
  //! half length along x on -z
  double x1( void ) const;
  //! half length along x on +z
  double x2( void ) const;
  //! half length along y on -z
  double y1( void ) const;
  //! half length along y on +z
  double y2( void ) const;   
  //! radius of the cut-out (neg.) or rounding (pos.)
  double radius( void ) const;
  //! true, if cut-out or rounding is on the -z side
  bool atMinusZ( void ) const;

private:
  DDPseudoTrap( void );
};
  
/// A truncated tube section    
class DDTruncTubs : public DDSolid
{
public:
  DDTruncTubs( const DDSolid & s );
  //! half of the z-Axis
  double zHalf( void ) const;
  //! inner radius
  double rIn( void ) const;
  //! outer radius
  double rOut( void ) const;
  //! angular start of the tube-section
  double startPhi( void ) const;
  //! angular span of the tube-section
  double deltaPhi( void ) const;
  //! truncation at begin of the tube-section
  double cutAtStart( void ) const;
  //! truncation at end of the tube-section
  double cutAtDelta( void ) const;
  //! true, if truncation is on the inner side of the tube-section
  bool cutInside( void ) const;

private:
  DDTruncTubs( void );
};

//! Interface to a Box
/**
   The definition (parameters, local frame) of the Box is 
   the same than in Geant4.
*/
class DDBox : public DDSolid
{
public:
  DDBox( const DDSolid & s );
  double halfX( void ) const; 
  double halfY( void ) const; 
  double halfZ( void ) const; 

private:
  DDBox( void );
};

/// This is simply a handle on the solid.
class DDShapelessSolid : public DDSolid
{
public:
  DDShapelessSolid( const DDSolid & s );

private:
  DDShapelessSolid( void );
};

class DDReflectionSolid : public DDSolid
{
public:
  DDReflectionSolid( const DDSolid & s );
  DDSolid unreflected( void ) const;

private:
  DDReflectionSolid( void ); 
  DDI::Reflection * reflected_; 
}; 

class DDBooleanSolid : public DDSolid
{
public:
  DDBooleanSolid( const DDSolid & s );
  DDSolid solidA( void ) const;
  DDSolid solidB( void ) const;
  DDTranslation translation( void ) const;
  DDRotation rotation( void ) const;

private:
  DDBooleanSolid( void );
  DDI::BooleanSolid * boolean_;  
};

/// Abstract class for DDPolycone and DDPolyhedra.  Basically a common member function.
class DDPolySolid : public DDSolid
{
public:
  DDPolySolid( const DDSolid & s );

protected:
  /// note defaults please.
  virtual std::vector<double> getVec( const size_t& which, const size_t& offset = 0, const size_t& nVecs = 1 ) const;
  DDPolySolid( void );
};
  
class DDPolycone : public DDPolySolid
{
public:
  DDPolycone( const DDSolid & s );
  double startPhi( void ) const;
  double deltaPhi( void ) const;
  std::vector<double> zVec( void ) const;
  std::vector<double> rVec( void ) const;
  std::vector<double> rMinVec( void ) const;
  std::vector<double> rMaxVec( void ) const;

private:
  DDPolycone( void );
};

class DDPolyhedra : public DDPolySolid
{
public:
  DDPolyhedra( const DDSolid & s );
  int sides( void ) const;
  double startPhi( void ) const;
  double deltaPhi( void ) const;
  std::vector<double> zVec( void ) const;
  std::vector<double> rVec( void ) const;
  std::vector<double> rMinVec( void ) const;
  std::vector<double> rMaxVec( void ) const;

private:
  DDPolyhedra( void );
};

class DDTubs : public DDSolid
{
public:
  DDTubs( const DDSolid & s );
  double zhalf( void ) const;
  double rIn( void ) const;
  double rOut( void ) const;
  double startPhi( void ) const;
  double deltaPhi( void ) const;

private:
  DDTubs( void );
};

class DDCons : public DDSolid
{
public:
  DDCons( const DDSolid & s );
  double zhalf( void ) const;
  double rInMinusZ( void ) const;
  double rOutMinusZ( void ) const;
  double rInPlusZ( void ) const;
  double rOutPlusZ( void ) const;
  double phiFrom( void ) const;
  double deltaPhi( void ) const;

private:
  DDCons( void );
};

class DDTorus : public DDSolid
{
public:
  DDTorus( const DDSolid & s );
  double rMin( void ) const;
  double rMax( void ) const;
  double rTorus( void ) const;
  double startPhi( void ) const;
  double deltaPhi( void ) const;

private:
  DDTorus( void );
};

class DDUnion : public DDBooleanSolid
{
public:
  DDUnion( const DDSolid & s );
  
private:
  DDUnion( void );
};

class DDIntersection : public DDBooleanSolid
{
public:
  DDIntersection( const DDSolid & s );

private:
  DDIntersection( void );
};

class DDSubtraction : public DDBooleanSolid
{
public:
  DDSubtraction( const DDSolid & s );

private:
  DDSubtraction( void );
};

class DDSphere : public DDSolid
{
public:
  DDSphere( const DDSolid & s );
  double innerRadius( void ) const;
  double outerRadius( void ) const;
  double startPhi( void ) const;
  double deltaPhi( void ) const;
  double startTheta( void ) const;
  double deltaTheta( void ) const;

private:
  DDSphere( void );
};

class DDOrb : public DDSolid
{
public:
  DDOrb( const DDSolid & s );
  double radius( void ) const;

private:
  DDOrb( void );
};

class DDEllipticalTube : public DDSolid
{
public:
  DDEllipticalTube( const DDSolid & s );
  double xSemiAxis( void ) const;
  double ySemiAxis( void ) const;
  double zHeight( void ) const;
  
private:
  DDEllipticalTube( void );
};

class DDEllipsoid : public DDSolid
{
public:
  DDEllipsoid( const DDSolid & s );
  double xSemiAxis( void ) const;
  double ySemiAxis( void ) const;
  double zSemiAxis( void ) const;
  double zBottomCut( void ) const;
  double zTopCut( void ) const;

private:
  DDEllipsoid( void );
};

class DDParallelepiped : public DDSolid
{
public:
  DDParallelepiped( const DDSolid & s );
  double xHalf( void ) const;
  double yHalf( void ) const;
  double zHalf( void ) const;
  double alpha( void ) const;
  double theta( void ) const;
  double phi( void ) const;

private:
  DDParallelepiped( void );
};

// Solid generation functions
//
struct DDSolidFactory
{
  //! Creates a box with side length 2*xHalf, 2*yHalf, 2*zHalf
  /** \arg \c name unique name identifying the box
      \arg \c xHalf half length in x 
      \arg \c yHalf half length in y
      \arg \c zHalf helf length in z
      The center of the box (for positioning) is the center of gravity.
  */    
  static DDSolid box( const DDName & name,
		      double xHalf, 
		      double yHalf,
		      double zHalf );

  //! Creates a polycone (refere to \b Geant3 or \b Geant4 documentation)
  /** The center of the polycone (for positioning) is the center of coordinates
      of the polycone definition (x=y=z=0)
  */    
  static DDSolid polycone( const DDName & name, double startPhi, double deltaPhi,
			   const std::vector<double> & z,
			   const std::vector<double> & rmin,
			   const std::vector<double> & rmax );

  //! Creates a polycone (refere to \b Geant4 documentation)
  /** The center of the polycone (for positioning) is the center of coordinates
      of the polycone definition (x=y=z=0)
  */    
  static DDSolid polycone( const DDName & name, double startPhi, double deltaPhi,
			   const std::vector<double> & z,
			   const std::vector<double> & r );

  //! Creates a polyhedra (refere to \b Geant3 or \b Geant4 documentation)
  /** The center of the polyhedra (for positioning) is the center of coordinates
      of the polyhedra definition (x=y=z=0)
  */    
  static DDSolid polyhedra( const DDName & name,
			    int sides,
			    double startPhi, double deltaPhi,
			    const std::vector<double> & z,
			    const std::vector<double> & rmin,
			    const std::vector<double> & rmax );
		     
  //! Creates a polyhedra (refere to \b Geant4 documentation)
  /** The center of the polyhedra (for positioning) is the center of coordinates
      of the polyhedra definition (x=y=z=0)
  */    
  static DDSolid polyhedra( const DDName & name,
			    int sides,
			    double startPhi, double deltaPhi,
			    const std::vector<double> & z,
			    const std::vector<double> & r );

  static DDSolid unionSolid( const DDName & name,
			     const DDSolid & a,
			     const DDSolid & b,
			     const DDTranslation & t,
			     const DDRotation & r );

  static DDSolid intersection( const DDName & name,
			       const DDSolid & a,
			       const DDSolid & b,
			       const DDTranslation & t,
			       const DDRotation & r );

  static DDSolid subtraction( const DDName & name,
			      const DDSolid & a,
			      const DDSolid & b,
			      const DDTranslation & t,
			      const DDRotation & r );

  static DDSolid trap( const DDName & name,
		       double pDz,
		       double pTheta, double pPhi,
		       double pDy1, double pDx1, double pDx2,
		       double pAlp1,
		       double pDy2, double pDx3, double pDx4,
		       double pAlp2 );		     		     

  static DDSolid pseudoTrap( const DDName & name,
			     double pDx1, 	/**< Half-length along x at the surface positioned at -dz */
			     double pDx2, 	/**< Half-length along x at the surface positioned at +dz */
			     double pDy1, 	/**< Half-length along y at the surface positioned at -dz */
			     double pDy2, 	/**< Half-length along y at the surface positioned at +dz */
			     double pDz, 	/**< Half of the height of the pseudo trapezoid along z */
			     double radius, 	/**< radius of the cut-out (negative sign) or rounding (pos. sign) */
			     bool atMinusZ ); 	/**< if true, the cut-out or rounding is applied at -dz, else at +dz */

  static DDSolid truncTubs( const DDName & name,
			    double zHalf, 	/**< half-length of the z-axis */
			    double rIn, 	/**< inner radius of the tube-section */
			    double rOut, 	/**< outer radius of the tube-section */
			    double startPhi, 	/**< starting angle of the tube-section */
			    double deltaPhi, 	/**< spanning angle of the tube-section */
			    double cutAtStart, 	/**< tructation */
			    double cutAtDelta, 	/**< */
			    bool cutInside );

  static DDSolid tubs( const DDName & name,
		       double zhalf,
		       double rIn, double rOut,	      	      
		       double startPhi, 
		       double deltaPhi );

  static DDSolid cons( const DDName & name,
		       double zhalf,
		       double rInMinusZ,	      	      
		       double rOutMinusZ,
		       double rInPlusZ,
		       double rOutPlusZ,
		       double phiFrom,
		       double deltaPhi );

  static DDSolid torus( const DDName & name,
		        double rMin,
		        double rMax,
		        double rTorus,
		        double startPhi,
		        double deltaPhi );
  
  static DDSolid sphere( const DDName & name,
			 double innerRadius,
			 double outerRadius,	      	      
			 double startPhi,
			 double deltaPhi,
			 double startTheta,
			 double deltaTheta );

  static DDSolid orb( const DDName & name,
		      double radius );
  
  static DDSolid ellipticalTube( const DDName & name,
				 double xSemiAxis,
				 double ySemiAxis,
				 double zHeight );

  static DDSolid ellipsoid( const DDName & name,
			    double  xSemiAxis,
			    double  ySemiAxis,
			    double  zSemiAxis,
			    double  zBottomCut = 0.,
			    double  zTopCut = 0. );

  static DDSolid parallelepiped( const DDName & name,
				 double xHalf, double yHalf, double zHalf,
				 double alpha, double theta, double phi );


  static DDSolid shapeless( const DDName & name );

  static DDSolid reflection( const DDName & name,
			     const DDSolid & s );
};		     		     				    		     		    
		      		      
#endif
