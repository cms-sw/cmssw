#ifndef DDSolid_h
#define DDSolid_h

#include <stdexcept>
#include <iosfwd>
#include <vector>
#include <utility>

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"

namespace DDI { class Solid; }
namespace DDI { class Reflection; }
namespace DDI { class BooleanSolid; }


//class DDSolidImpl;
class DDSolid;

class DDSolidFactory;
class DDStreamer;
//class DDBooleanImpl;

std::ostream & operator<<( std::ostream &, const DDSolid &);


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
class DDSolid : public DDBase<DDName,DDI::Solid*>
{
  friend std::ostream & operator<<(std::ostream &, const DDSolid &);
  friend class DDSolidFactory;
  friend class DDDToPersFactory;
  friend class DDPersToDDDFactory;
  friend class DDStreamer;
    
public: 
  //! Uninitialilzed solid reference-object; for further details on reference-objects see documentation of DDLogicalPart
  DDSolid();
  
  //! Creates a reference-object to a solid named \a name
  /** If the solid was not yet created using one of the solid generating factory 
      functions \c DDbox(), \c DDtub , ... this constructor creates a (default) 
      initialized reference object named \a name. It can be used as placeholder 
      everywhere and becomes a reference to a valid solid as soon as one of the 
      factory functions \c DDBox, ... has been called (using the same value for DDName).
      
      For further details concerning reference-objects refer to the documentation of DDLogicalPart.       
  */
  DDSolid(const DDName & name); 
  
  //! \b Don't \b use (only meant to be used by DDbox(), DDtub(), ...)
  /** \todo make it private and add friends DDbox(), ... 
  DDSolid(DDSolidImpl* p)  
      : DDRegistered<DDSolidImpl>(p) 
      { }
  */
  
  //! Give the parameters of the solid
  const std::vector<double> & parameters() const;// { return rep().parameters(); }
  
  //! Returns the volume of the given solid (\b does \b not \b work \b with \b boolean \b soids \b !)        
  double volume() const; // { return rep().volume(); }
  
  //! The type of the solid
  DDSolidShape shape() const; // { return rep().shape(); }
  
  //! True, if the solid is a boolean solid. Then it consists of 2 boolean compounds
  //bool boolean() const; // { return rep().boolean(); }
  
  //! Returns the compounds of a boolean solid
  /** If DDSolid::boolean() == true, this method provides access to the
      boolean compounds of the solid. The type of the boolean solid 
      (intersection, union or subtraction solid) is the return value and
      the compounds are assigned to the arguments \a A and \a B. Further
      the relative position of solid \a B towards \a A is assigned to the arguments
      DDRotation& and DDTranslation&.
  */   
  //DDSolidShape boolean( DDSolid & A, DDSolid & B, DDRotation &, DDTranslation &);

/*   static void clear(); */
  
private:
  DDSolid(const DDName &, DDI::Solid *);    
  DDSolid(const DDName &, DDSolidShape, const std::vector<double> &);
};

  struct Composites {
     Composites (const DDSolid & A, const DDSolid & B,
                const DDTranslation & T, const DDRotation & R)
	       : a_(A), b_(B), t_(T), r_(R) { }
     const DDSolid a_, b_;
     const DDTranslation t_;
     const DDRotation r_;	         
  }; 

//! Interface to a Trapezoid
/**
  The definition (parameters, local frame) of the Trapezoid is 
  the same than in Geant4.
*/
class DDTrap : public DDSolid
{
public:
  DDTrap(const DDSolid & s);
  //! half of the z-Axis
  double halfZ() const;
  //! Polar angle of the line joining the centres of the faces at -/+pDz
  double theta() const;
  //! Azimuthal angle of the line joining the centres of the faces at -/+pDz
  double phi() const;
  //! Half-length along y of the face at -pDz
  double y1() const;
  //! Half-length along x of the side at y=-pDy1 of the face at -pDz
  double x1() const;
  //! Half-length along x of the side at y=+pDy1 of the face at -pDz
  double x2() const;
  //! Angle with respect to the y axis from the centre of the side at y=-pDy1 to the centre at y=+pDy1 of the face at -pDz
  double alpha1() const;
  //! Half-length along y of the face at +pDz
  double y2() const;
  //! Half-length along x of the side at y=-pDy2 of the face at +pDz
  double x3() const;
  //! Half-length along x of the side at y=+pDy2 of the face at +pDz
  double x4() const;
  //! Angle with respect to the y axis from the centre of the side at y=-pDy2 to the centre at y=+pDy2 of the face at +pDz
  double alpha2() const;
};

class DDPseudoTrap : public DDSolid
{
public:
   DDPseudoTrap(const DDSolid & s);
   //! half of the z-Axis
   double halfZ() const;
   //! half length along x on -z
   double x1() const;
   //! half length along x on +z
   double x2() const;
   //! half length along y on -z
   double y1() const;
   //! half length along y on +z
   double y2() const;   
   //! radius of the cut-out (neg.) or rounding (pos.)
   double radius() const;
   //! true, if cut-out or rounding is on the -z side
   bool atMinusZ() const;
};
  
/// A truncated tube section    
class DDTruncTubs : public DDSolid
{
public:
  DDTruncTubs(const DDSolid & s);
  //! half of the z-Axis
  double zHalf() const;
  //! inner radius
  double rIn() const;
  //! outer radius
  double rOut() const;
  //! angular start of the tube-section
  double startPhi() const;
  //! angular span of the tube-section
  double deltaPhi() const;
  //! truncation at begin of the tube-section
  double cutAtStart() const;
  //! truncation at end of the tube-section
  double cutAtDelta() const;
  //! true, if truncation is on the inner side of the tube-section
  bool cutInside() const;
};
class DDBox : public DDSolid
{
public:
  DDBox(const DDSolid & s);
  double halfX() const; 
  double halfY() const; 
  double halfZ() const; 
private:
  DDBox();
};

/// This is simply a handle on the solid.
class DDShapelessSolid : public DDSolid

{
 public:
  DDShapelessSolid(const DDSolid & s);

 private:
  DDShapelessSolid();
};

class DDReflectionSolid : public DDSolid
{
public:
  DDReflectionSolid(const DDSolid & s);
  DDSolid unreflected() const;
private:
  DDReflectionSolid(); 
  DDI::Reflection * reflected_; 
}; 


class DDBooleanSolid : public DDSolid
{
public:
  DDBooleanSolid(const DDSolid & s);
  DDSolid solidA() const;
  DDSolid solidB() const;
  DDTranslation translation() const;
  DDRotation rotation() const;
private:
  DDBooleanSolid();
  DDI::BooleanSolid * boolean_;  
};

/// Abstract class for DDPolycone and DDPolyhedra.  Basically a common member function.
class DDPolySolid : public DDSolid
{
 public:
  DDPolySolid(const DDSolid & s);

 protected:
  /// note defaults please.
  virtual std::vector<double> getVec(const size_t& which, const size_t& offset = 0, const size_t& nVecs = 1) const;
  DDPolySolid();

};
  
class DDPolycone : public DDPolySolid
{
 public:
  DDPolycone(const DDSolid & s);
  double startPhi() const;
  double deltaPhi() const;
  std::vector<double> zVec() const;
  std::vector<double> rVec() const;
  std::vector<double> rMinVec() const;
  std::vector<double> rMaxVec() const;

 private:
  DDPolycone();
};

class DDPolyhedra : public DDPolySolid
{
 public:
  DDPolyhedra(const DDSolid & s);
  int sides() const;
  double startPhi() const;
  double deltaPhi() const;
  std::vector<double> zVec() const;
  std::vector<double> rVec() const;
  std::vector<double> rMinVec() const;
  std::vector<double> rMaxVec() const;

 private:
  DDPolyhedra();
};

class DDTubs : public DDSolid
{
 public:
  DDTubs(const DDSolid & s);
  double zhalf() const;
  double rIn() const;
  double rOut() const;
  double startPhi() const;
  double deltaPhi() const;

 private:
  DDTubs();
};

class DDCons : public DDSolid
{
 public:
  DDCons(const DDSolid & s);
  double zhalf() const;
  double rInMinusZ() const;
  double rOutMinusZ() const;
  double rInPlusZ() const;
  double rOutPlusZ() const;
  double phiFrom() const;
  double deltaPhi() const;

 private:
  DDCons();
};

class DDTorus : public DDSolid
{
 public:
  DDTorus (const DDSolid & s);
  double rMin() const;
  double rMax() const;
  double rTorus() const;
  double startPhi() const;
  double deltaPhi() const;

 private:
  DDTorus
();
};

class DDUnion : public DDBooleanSolid
{
 public:
  DDUnion(const DDSolid & s);
  
 private:
  DDUnion();
};

class DDIntersection : public DDBooleanSolid
{
 public:
  DDIntersection(const DDSolid & s);

 private:
  DDIntersection();
};

class DDSubtraction : public DDBooleanSolid
{
 public:
  DDSubtraction(const DDSolid & s);

 private:
  DDSubtraction();
};


class DDSphere : public DDSolid
{
 public:
  DDSphere(const DDSolid & s);
  double innerRadius() const;
  double outerRadius() const;
  double startPhi() const;
  double deltaPhi() const;
  double startTheta() const;
  double deltaTheta() const;

 private:
  DDSphere();
};

class DDOrb : public DDSolid
{
 public:
  DDOrb(const DDSolid & s);
  double radius() const;

 private:
  DDOrb();
};

class DDEllipticalTube : public DDSolid
{
 public:
  DDEllipticalTube(const DDSolid & s);
  double xSemiAxis() const;
  double ySemiAxis() const;
  double zHeight() const;
  
 private:
  DDEllipticalTube();
};

class DDEllipsoid : public DDSolid
{
 public:
  DDEllipsoid(const DDSolid & s);
  double xSemiAxis() const;
  double ySemiAxis() const;
  double zSemiAxis() const;
  double zBottomCut() const;
  double zTopCut() const;

 private:
  DDEllipsoid();
};

class DDParallelepiped : public DDSolid
{
 public:
  DDParallelepiped(const DDSolid & s);
  double xHalf() const;
  double yHalf() const;
  double zHalf() const;
  double alpha() const;
  double theta() const;
  double phi() const;

 private:
  DDParallelepiped();
};

// Solid generation function
//! Creates a box with side length 2*xHalf, 2*yHalf, 2*zHalf
/** \arg \c name unique name identifying the box
    \arg \c xHalf half length in x 
    \arg \c yHalf half length in y
    \arg \c zHalf helf length in z
    The center of the box (for positioning) is the center of gravity.
*/    
struct DDSolidFactory {


static DDSolid box(const DDName & name,
                     double xHalf, 
	             double yHalf,
	             double zHalf);

//! Creates a polycone (refere to \b Geant3 or \b Geant4 documentation)
/** The center of the polycone (for positioning) is the center of coordinates
    of the polycone definition (x=y=z=0)
*/    
static DDSolid   polycone(const DDName & name, double startPhi, double deltaPhi,
                     const std::vector<double> & z,
		     const std::vector<double> & rmin,
		     const std::vector<double> & rmax);

//! Creates a polycone (refere to \b Geant4 documentation)
/** The center of the polycone (for positioning) is the center of coordinates
    of the polycone definition (x=y=z=0)
*/    
static DDSolid   polycone(const DDName & name, double startPhi, double deltaPhi,
                     const std::vector<double> & z,
		     const std::vector<double> & r);

//! Creates a polyhedra (refere to \b Geant3 or \b Geant4 documentation)						     //! Creates a polycone (refere to \b Geant3 or \b Geant4 documentation)
/** The center of the polyhedra (for positioning) is the center of coordinates
    of the polyhedra definition (x=y=z=0)
*/    
static DDSolid polyhedra(const DDName & name,
                     int sides,
		     double startPhi, double deltaPhi,
                     const std::vector<double> & z,
		     const std::vector<double> & rmin,
		     const std::vector<double> & rmax);
		     
//! Creates a polyhedra (refere to \b Geant4 documentation)						     //! Creates a polycone (refere to \b Geant3 or \b Geant4 documentation)
/** The center of the polyhedra (for positioning) is the center of coordinates
    of the polyhedra definition (x=y=z=0)
*/    
static DDSolid polyhedra(const DDName & name,
                     int sides,
                     double startPhi, double deltaPhi,
		     const std::vector<double> & z,
		     const std::vector<double> & r);

static DDSolid unionSolid(const DDName & name,
                     const DDSolid & a,
		     const DDSolid & b,
		     const DDTranslation & t,
		     const DDRotation & r);

static DDSolid intersection(const DDName & name,
                     const DDSolid & a,
		     const DDSolid & b,
		     const DDTranslation & t,
		     const DDRotation & r);

static DDSolid subtraction(const DDName & name,
                     const DDSolid & a,
		     const DDSolid & b,
		     const DDTranslation & t,
		     const DDRotation & r);

static DDSolid trap(const DDName & name,
                     double pDz,
	             double pTheta, double pPhi,
	             double pDy1, double pDx1, double pDx2,
	             double pAlp1,
	             double pDy2, double pDx3, double pDx4,
	             double pAlp2);		     		     

static DDSolid pseudoTrap(const DDName & name,
                          double pDx1, /**< Half-length along x at the surface positioned at -dz */
			     double pDx2, /**<  Half-length along x at the surface positioned at +dz */
			     double pDy1, /**<  Half-length along y at the surface positioned at -dz */
			     double pDy2, /**<  Half-length along y at the surface positioned at +dz */
			     double pDz, /**< Half of the height of the pseudo trapezoid along z */
			     double radius, /**< radius of the cut-out (negative sign) or rounding (pos. sign) */
			     bool atMinusZ /**< if true, the cut-out or rounding is applied at -dz, else at +dz */
			     );

static DDSolid truncTubs(const DDName & name,
                                  double zHalf, /**< half-length of the z-axis */
				  double rIn, /**< inner radius of the tube-section */
				  double rOut, /**< outer radius of the tube-section */
				  double startPhi, /**< starting angle of the tube-section */
				  double deltaPhi, /**< spanning angle of the tube-section */
				  double cutAtStart, /**< tructation */
				  double cutAtDelta, /**< */
				  bool cutInside);

static DDSolid tubs(const DDName & name,
                     double zhalf,
		     double rIn, double rOut,	      	      
		     double startPhi, 
		     double deltaPhi);

static DDSolid cons(const DDName & name,
                     double zhalf,
	   	     double rInMinusZ,	      	      
		     double rOutMinusZ,
		     double rInPlusZ,
		     double rOutPlusZ,
		     double phiFrom,
		     double deltaPhi);

static DDSolid torus(const DDName & name,
		     double rMin,
		     double rMax,
		     double rTorus,
		     double startPhi,
		     double deltaPhi);
  
static DDSolid sphere(const DDName & name,
		     double innerRadius,
	   	     double outerRadius,	      	      
		     double startPhi,
		     double deltaPhi,
		     double startTheta,
		     double deltaTheta);

  static DDSolid orb(const DDName & name,
		     double radius);
  
  static DDSolid ellipticalTube(const DDName & name,
				double xSemiAxis,
				double ySemiAxis,
				double zHeight);

static DDSolid ellipsoid(const DDName & name,
                         double  xSemiAxis,
                         double  ySemiAxis,
                         double  zSemiAxis,
                         double  zBottomCut=0,
                         double  zTopCut=0
			 );

static DDSolid parallelepiped(const DDName & name,
                              double xHalf, double yHalf, double zHalf,
                              double alpha, double theta, double phi);


static DDSolid shapeless(const DDName & name);

static DDSolid reflection(const DDName & name,
                          const DDSolid & s);

};		     		     				    		     		    
		      		      
/* //! Creates a cone or a cone-section (refere to \b Geant4 documentation) */
/* /\** The center of the cone or cone-section (for positioning) is the center of */
/*     coordinates of its definition (x=y=z=0). */
/* *\/     */
/* DDSolid *      DDcons(const DDName & name, */
/*                      double zhalf, */
/* 	   	     double rInMinusZ, */
/* 		     double rOutMinusZ, */
/* 		     double rInPlusZ, */
/* 		     double rOutPlusZ, */
/* 		     double startPhi, */
/* 		     double deltaPhi); */

/* //! Creates a torus or a toroidal-section (refere to \b Geant4 documentation) */
/* /\** The center of the torus or toroidal-section (for positioning) is the center of */
/*     coordinates of its definition (x=y=z=0). */
/* *\/     */
/* DDSolid *      DDtorus(const DDName & name, */
/* 		       double rMin, */
/* 		       double rMax, */
/* 		       double rTorus, */
/* 		       double startPhi, */
/* 		       double deltaPhi); */

/* //! Creates a tube or a tube-section (refere to \b Geant4 documentation) */
/* /\** The center of the tube or tube-section (for positioning) is the center of */
/*     coordinates of its definition (x=y=z=0). */
/* *\/     */
/* DDSolid *      DDtubs(const DDName & name, */
/*                      double zhalf, */
/* 		     double rIn, double rOut,	      	       */
/* 		     double startPhi,  */
/* 		     double deltaPhi); */
		     

/* //! Creates a trapezoid (refere to \b Geant3 or \b Geant4 documentation) */
/* /\** The center of the tube or tube-section (for positioning) is the center of */
/*     coordinates of its definition (x=y=z=0). */
/* *\/     */
/* DDSolid *      DDtrap(const DDName & name, */
/*                      double pDz, */
/* 	             double pTheta, double pPhi, */
/* 	             double pDy1, double pDx1, double pDx2, */
/* 	             double pAlp1, */
/* 	             double pDy2, double pDx3, double pDx4, */
/* 	             double pAlp2); */
		     
/* //! Creates a shapeless solid */
/* /\** When used in a mother-volume this type of solid forms an assembly */
/*     in the \b Geant4 sense */
/* *\/     */
/* DDSolid * DDshapeless(const DDName & name); */

/* /\* */
/* DDSolid      DDunion(const DDName & name, */
/*                      const DDSolid & a = DDSolid(),  */
/*                      const DDSolid & b = DDSolid() ); */
/* *\/ */

/* //! Creates a boolean solid - the union of 2 compound solids */
/* /\** \arg \c name unique name */
/*     \arg \c a first compound of the union */
/*     \arg \c b second comound of the union */
/*     \arg \c r relative rotation of \c b towards \c a */
/*     \arg \c t relative translation of \c b towards \c a */
/* *\/    								 				 		 		  */
/* DDSolid *     DDunion(const DDName & name, */
/*                      const DDSolid & a,  */
/*                      const DDSolid & b, */
/* 		     const DDRotation &r =DDRotation(), */
/* 		     const DDTranslation &t =DDTranslation()); */

/* //! Creates a boolean solid - the union of 2 compound solids */
/* /\** \arg \c name unique name */
/*     \arg \c a first compound of the subtraction */
/*     \arg \c b second comound of the subtraction */
/*     \arg \c r relative rotation of \c b towards \c a */
/*     \arg \c t relative translation of \c b towards \c a */
/*     \c b is subtracted from \c a . */
/* *\/    								 				 		 		  */
/* DDSolid *     DDsubtraction(const DDName & name, */
/*                            const DDSolid & a,  */
/*                            const DDSolid & b, */
/* 		           const DDRotation &r =DDRotation(), */
/* 		           const DDTranslation &t =DDTranslation());  */
		     
/* //! Creates a boolean solid - the intersection of 2 compound solids */
/* /\** \arg \c name unique name */
/*     \arg \c a first compound of the intersection */
/*     \arg \c b second comound of the intersection */
/*     \arg \c r relative rotation of \c b towards \c a */
/*     \arg \c t relative translation of \c b towards \c a */
/*     The resulting solid is the common volume of \c a and \c b . */
/* *\/    								 				 		 		  */
/* DDSolid *     DDintersection(const DDName & name, */
/*                             const DDSolid & a,  */
/*                             const DDSolid & b, */
/* 		            const DDRotation &r =DDRotation(), */
/* 		            const DDTranslation &t =DDTranslation());  */
		     

/* //! Creates a cone or a cone-section (refere to \b Geant4 documentation) */
/* /\** The center of the cone or cone-section (for positioning) is the center of */
/*     coordinates of its definition (x=y=z=0). */
/* *\/     */
/* DDSolid *      DDcons(const DDName & name, */
/*                      double zhalf, */
/* 	   	     double rInMinusZ, */
/* 		     double rOutMinusZ, */
/* 		     double rInPlusZ, */
/* 		     double rOutPlusZ, */
/* 		     double startPhi, */
/* 		     double deltaPhi); */
			
#endif
