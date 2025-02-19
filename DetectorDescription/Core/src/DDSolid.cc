#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/src/Solid.h"

#include "DetectorDescription/Core/src/Box.h"
#include "DetectorDescription/Core/src/Polycone.h"
#include "DetectorDescription/Core/src/Polyhedra.h"
#include "DetectorDescription/Core/src/Boolean.h"
#include "DetectorDescription/Core/src/Reflection.h"
#include "DetectorDescription/Core/src/Shapeless.h"
#include "DetectorDescription/Core/src/Torus.h"
#include "DetectorDescription/Core/src/Trap.h"
#include "DetectorDescription/Core/src/Tubs.h"
#include "DetectorDescription/Core/src/Cons.h"
#include "DetectorDescription/Core/src/PseudoTrap.h"
#include "DetectorDescription/Core/src/TruncTubs.h"
#include "DetectorDescription/Core/src/Sphere.h"
#include "DetectorDescription/Core/src/Orb.h"
#include "DetectorDescription/Core/src/EllipticalTube.h"
#include "DetectorDescription/Core/src/Ellipsoid.h"
#include "DetectorDescription/Core/src/Parallelepiped.h"
#include <algorithm>

// Message logger.
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using DDI::Solid;

//DDBase<DDName,Solid*>::StoreT::pointer_type 
//  DDBase<DDName,Solid*>::StoreT::instance_ = 0;


std::ostream & 
operator<<(std::ostream & os, const DDSolid & solid)
{ 
  DDBase<DDName,DDI::Solid*>::def_type defined(solid.isDefined());
  if (defined.first) {
    os << *(defined.first) << " ";
    if (defined.second) {
      os << "  " << DDSolidShapesName::name(solid.shape()) << ": ";
      solid.rep().stream(os); 
    }
    else {
      os << "* solid not defined * ";  
    }
  }  
  else {
    os << "* solid not declared * ";  
  }  
  return os;
}


// =================================================================================

DDSolid::DDSolid() : DDBase<DDName,Solid*>() { }


DDSolid::DDSolid(const DDName & n) : DDBase<DDName,Solid*>()
{
  prep_ = StoreT::instance().create(n);
}

DDSolid::DDSolid(const DDName & n, Solid * s) : DDBase<DDName,Solid*>()
{
  prep_ = StoreT::instance().create(n,s);
}


DDSolid::DDSolid(const DDName & n, DDSolidShape s, const std::vector<double> & p)
{
  DDI::Solid * solid(0);
  std::vector<double> dummy;
  switch(s) {
       case ddbox:
        solid = new DDI::Box(0,0,0);
        break;
       case ddtubs:
        solid = new DDI::Tubs(0,0,0,0,0);
        break;
       case ddcons:
        solid = new DDI::Cons(0,0,0,0,0,0,0);
        break;
       case ddpseudotrap:
        solid = new DDI::PseudoTrap(0,0,0,0,0,0,0);
        break;
       case ddshapeless:
        solid = new DDI::Shapeless();
        break;
       case ddtrap:
        solid = new DDI::Trap(0,0,0,0,0,0,0,0,0,0,0);
        break;
       case ddpolyhedra_rz:
        solid = new DDI::Polyhedra(0,0,0,dummy,dummy);
	break;
       case ddpolyhedra_rrz:
        solid = new DDI::Polyhedra(0,0,0,dummy,dummy,dummy);
	break;
       case ddpolycone_rz:
        solid = new DDI::Polycone(0,0,dummy,dummy);
	break;
       case ddpolycone_rrz:
        solid = new DDI::Polycone(0,0,dummy,dummy,dummy);
	break;			
       case ddtrunctubs:
	 solid = new DDI::TruncTubs(0,0,0,0,0,0,0,0);
	 break;
       case ddtorus:
	 solid = new DDI::Torus(0,0,0,0,0);
	 break;
       case ddsphere:
	 solid = new DDI::Sphere(0,0,0,0,0,0);
	 break;
       case ddorb:
	 solid = new DDI::Orb(0);
	 break;
       case ddellipticaltube:
	 solid = new DDI::EllipticalTube(0,0,0);
	 break;
       case ddellipsoid:
	 solid = new DDI::Ellipsoid(0,0,0,0,0);
	 break;
       case ddparallelepiped:
	 solid = new DDI::Parallelepiped(0,0,0,0,0,0);
	 break;
       default:
        throw cms::Exception("DDException") << "DDSolid::DDSolid(DDName,DDSolidShape,std::vector<double>: wrong shape";   
  }
  solid->setParameters(p);
  prep_ = StoreT::instance().create(n,solid);
}


double DDSolid::volume() const
{
  return rep().volume();
}

// void DDSolid::clear()
// {
//  StoreT::instance().clear();
// }


DDSolidShape DDSolid::shape() const
{
  return rep().shape();
}


const std::vector<double> & DDSolid::parameters() const
{ 
  return rep().parameters(); 
}


// =================================================================================

DDTrap::DDTrap(const DDSolid & s) : DDSolid(s)
{
  if (s.shape() != ddtrap) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is no DDTrap.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double DDTrap::halfZ() const { return rep().parameters()[0]; }

double DDTrap::theta() const { return rep().parameters()[1]; }

double DDTrap::phi() const { return rep().parameters()[2]; }

double DDTrap::y1() const { return rep().parameters()[3]; }

double DDTrap::x1() const { return rep().parameters()[4]; }

double DDTrap::x2() const { return rep().parameters()[5]; }

double DDTrap::alpha1() const { return rep().parameters()[6]; }

double DDTrap::y2() const { return rep().parameters()[7]; }

double DDTrap::x3() const { return rep().parameters()[8]; }

double DDTrap::x4() const { return rep().parameters()[9]; }

double DDTrap::alpha2() const { return rep().parameters()[10]; }

// =================================================================================

DDTruncTubs::DDTruncTubs(const DDSolid & s) : DDSolid(s)
{
  if (s.shape() != ddtrunctubs) {
    edm::LogError ("DDSolid") << "s.shape()=" << s.shape() << "  " << s << std::endl;
    std::string ex = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is no DDTruncTubs\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double DDTruncTubs::zHalf() const { return rep().parameters()[0];}

double DDTruncTubs::rIn() const { return rep().parameters()[1];}

double DDTruncTubs::rOut() const { return rep().parameters()[2];}

double DDTruncTubs::startPhi() const { return rep().parameters()[3];}

double DDTruncTubs::deltaPhi() const { return rep().parameters()[4];}

double DDTruncTubs::cutAtStart() const { return rep().parameters()[5];}

double DDTruncTubs::cutAtDelta() const { return rep().parameters()[6];}

bool DDTruncTubs::cutInside() const { return bool(rep().parameters()[7]);}

// =================================================================================

DDPseudoTrap::DDPseudoTrap(const DDSolid & s) : DDSolid(s)
{
  if (s.shape() != ddpseudotrap) {
    std::string ex = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is no DDPseudoTrap\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double DDPseudoTrap::halfZ() const { return rep().parameters()[4]; }

double DDPseudoTrap::x1() const { return rep().parameters()[0]; }

double DDPseudoTrap::x2() const { return rep().parameters()[1]; }

double DDPseudoTrap::y1() const { return rep().parameters()[2]; }

double DDPseudoTrap::y2() const { return rep().parameters()[3]; }

double DDPseudoTrap::radius() const { return rep().parameters()[5]; }

bool DDPseudoTrap::atMinusZ() const { return rep().parameters()[6]; }

// =================================================================================

DDBox::DDBox(const DDSolid & s) 
 : DDSolid(s) 
{ 
 if (s.shape() != ddbox) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDBox.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
 }
}

double DDBox::halfX() const
{ return rep().parameters()[0]; }
double DDBox::halfY() const
{ return rep().parameters()[1]; }
double DDBox::halfZ() const
{ return rep().parameters()[2]; }


// =================================================================================

DDReflectionSolid::DDReflectionSolid(const DDSolid & s)
 : DDSolid(s), reflected_(0)
{ 
  //FIXME: exception handling!
  reflected_ = dynamic_cast<DDI::Reflection*>(&s.rep());
}


DDSolid DDReflectionSolid::unreflected() const
{ return reflected_->solid();}


// =================================================================================

DDShapelessSolid::DDShapelessSolid (const DDSolid & s) : DDSolid(s)
{
  if (s.shape() != ddshapeless) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDShapelessSolid.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}


// =================================================================================

DDUnion::DDUnion(const DDSolid & s) 
  : DDBooleanSolid(s)
{
  if (s.shape() != ddunion) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDUnion.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}


// =================================================================================

DDIntersection::DDIntersection(const DDSolid & s) 
  : DDBooleanSolid(s)
{
  if (s.shape() != ddunion) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDIntersection.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}


// =================================================================================

DDSubtraction::DDSubtraction(const DDSolid & s) 
  : DDBooleanSolid(s)
{
  if (s.shape() != ddsubtraction) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is no DDSubtraction.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}


// =================================================================================

DDPolySolid::DDPolySolid(const DDSolid & s)
  : DDSolid(s)
{ }

std::vector<double> DDPolySolid::getVec (const size_t& which, 
					 const size_t& offset,
					 const size_t& numVecs) const {

  // which:  first second or third std::vector 
  // offset: number of non-std::vector components before std::vectors start
  std::string locErr;
//   size_t szVec = 0;
  std::vector<double> tvec; // = new std::vector<double>;
  if ( (rep().parameters().size() - offset) % numVecs != 0 ) { // / 2 != (rep().parameters().size() - 2) \ 2) {
    locErr = std::string("Could not find equal sized components of std::vectors in a PolySolid description.");
    edm::LogError ("DDSolid") << "rep().parameters().size()=" << rep().parameters().size() << "  numVecs=" << numVecs
	 << "  offset=" << offset << std::endl;
  }
//   else {
//     szVec = (rep().parameters().size() - offset)/ numVecs;
//   }
  for (size_t i = offset + which; i < rep().parameters().size(); i = i + numVecs) {
    tvec.push_back(rep().parameters()[i]);
  }		   
  return tvec;
}

// =================================================================================

DDPolycone::DDPolycone(const DDSolid & s)
  : DDPolySolid(s)
{
  if (s.shape() != ddpolycone_rz && s.shape() != ddpolycone_rrz) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDPolycone.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double DDPolycone::startPhi() const { return rep().parameters()[0]; }

double DDPolycone::deltaPhi() const { return rep().parameters()[1]; }

std::vector<double> DDPolycone::rVec() const {
  std::vector<double> tvec;
  if (shape() == ddpolycone_rz)
    tvec = getVec(1, 2, 2);
  return tvec; 
}

std::vector<double> DDPolycone::zVec() const {
  if (shape() == ddpolycone_rz)
    return getVec(0, 2, 2);
  else // (shape() == ddpolycone_rrz)
    return getVec(0, 2, 3);
}

std::vector<double> DDPolycone::rMinVec() const {
  std::vector<double> tvec;
  if (shape() == ddpolycone_rrz)
    tvec = getVec(1, 2, 3);
  return tvec; 
}

std::vector<double> DDPolycone::rMaxVec() const {
  std::vector<double> tvec;
  if (shape() == ddpolycone_rrz)
    tvec = getVec(2, 2, 3);
  return tvec; 
}


// =================================================================================

DDPolyhedra::DDPolyhedra(const DDSolid & s)
  : DDPolySolid(s)
{
  if (s.shape() != ddpolyhedra_rz && s.shape() != ddpolyhedra_rrz) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDPolyhedra.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

int DDPolyhedra::sides() const { return int(rep().parameters()[0]); }

double DDPolyhedra::startPhi() const { return rep().parameters()[1]; }

double DDPolyhedra::deltaPhi() const { return rep().parameters()[2]; }

std::vector<double> DDPolyhedra::rVec() const {
  std::vector<double> tvec;
  if (shape() == ddpolyhedra_rz)
    tvec = getVec(1, 3, 2);
  return tvec;
}

std::vector<double> DDPolyhedra::zVec() const {
  if (shape() == ddpolyhedra_rz)
    return getVec(0, 3, 2);
  else // (shape() == ddpolycone_rrz)
    return getVec(0, 3, 3);
}

std::vector<double> DDPolyhedra::rMinVec() const {
  std::vector<double> tvec;
  if (shape() == ddpolyhedra_rrz)
    tvec = getVec(1, 3, 3);
  return tvec;
}

std::vector<double> DDPolyhedra::rMaxVec() const {
  std::vector<double> tvec;
  if (shape() == ddpolyhedra_rrz)
    tvec = getVec(2, 3, 3);
  return tvec;
}

// =================================================================================

DDCons::DDCons(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != ddcons) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDCons.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double DDCons::zhalf() const { return rep().parameters()[0]; }

double DDCons::rInMinusZ() const { return rep().parameters()[1]; }

double DDCons::rOutMinusZ () const { return rep().parameters()[2]; }

double DDCons::rInPlusZ() const { return rep().parameters()[3]; }

double DDCons::rOutPlusZ() const { return rep().parameters()[4]; }

double DDCons::phiFrom() const { return rep().parameters()[5]; }

double DDCons::deltaPhi() const { return rep().parameters()[6]; }

// =================================================================================

DDTorus::DDTorus(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != ddtorus) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDTorus.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double DDTorus::rMin() const { return rep().parameters()[0]; }

double DDTorus::rMax() const { return rep().parameters()[1]; }

double DDTorus::rTorus () const { return rep().parameters()[2]; }

double DDTorus::startPhi() const { return rep().parameters()[3]; }

double DDTorus::deltaPhi() const { return rep().parameters()[4]; }


// =================================================================================

DDTubs::DDTubs(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != ddtubs) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDTubs.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double DDTubs::zhalf() const { return rep().parameters()[0]; }

double DDTubs::rIn() const { return rep().parameters()[1]; }

double DDTubs::rOut() const { return rep().parameters()[2]; }

double DDTubs::startPhi() const { return rep().parameters()[3]; }

double DDTubs::deltaPhi() const { return rep().parameters()[4]; }


// =================================================================================


DDSolid DDSolidFactory::box(const DDName & name, 
                              double xHalf, 
			      double yHalf, 
			      double zHalf)
{
  return DDSolid(name, new DDI::Box(xHalf, yHalf, zHalf ));
}


DDBooleanSolid::DDBooleanSolid(const DDSolid &s)
 : DDSolid(s), boolean_(0)
{
  boolean_ = dynamic_cast<DDI::BooleanSolid*>(&s.rep());
}


DDRotation DDBooleanSolid::rotation() const
{
  return boolean_->r();
}

DDTranslation DDBooleanSolid::translation() const
{
  return boolean_->t();
}

DDSolid DDBooleanSolid::solidA() const
{
  return boolean_->a();
}

DDSolid DDBooleanSolid::solidB() const
{
  return boolean_->b();
}

// =================================================================================

DDSphere::DDSphere(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != ddsphere) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDSphere (or sphere section).\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double DDSphere::innerRadius() const { return rep().parameters()[0]; }

double DDSphere::outerRadius() const { return rep().parameters()[1]; }

double DDSphere::startPhi () const { return rep().parameters()[2]; }

double DDSphere::deltaPhi() const { return rep().parameters()[3]; }

double DDSphere::startTheta() const { return rep().parameters()[4]; }

double DDSphere::deltaTheta() const { return rep().parameters()[5]; }

// =================================================================================

DDOrb::DDOrb(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != ddorb) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDOrb.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double DDOrb::radius() const { return rep().parameters()[0]; }

// =================================================================================

DDEllipticalTube::DDEllipticalTube(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != ddellipticaltube) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDEllipticalTube.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double DDEllipticalTube::xSemiAxis() const { return rep().parameters()[0]; }

double DDEllipticalTube::ySemiAxis() const { return rep().parameters()[1]; }

double DDEllipticalTube::zHeight() const { return rep().parameters()[2]; }

// =================================================================================

DDEllipsoid::DDEllipsoid(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != ddellipsoid) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDEllipsoid (or truncated ellipsoid).\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double DDEllipsoid::xSemiAxis() const { return rep().parameters()[0]; }

double DDEllipsoid::ySemiAxis() const { return rep().parameters()[1]; }

double DDEllipsoid::zSemiAxis() const { return rep().parameters()[2]; }

double DDEllipsoid::zBottomCut() const { return rep().parameters()[3]; }

double DDEllipsoid::zTopCut() const { return rep().parameters()[4]; }

// =================================================================================

DDParallelepiped::DDParallelepiped(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != ddparallelepiped) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDParallelepiped.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double DDParallelepiped::xHalf() const { return rep().parameters()[0]; }

double DDParallelepiped::yHalf() const { return rep().parameters()[1]; }

double DDParallelepiped::zHalf () const { return rep().parameters()[2]; }

double DDParallelepiped::alpha() const { return rep().parameters()[3]; }

double DDParallelepiped::theta() const { return rep().parameters()[4]; }

double DDParallelepiped::phi() const { return rep().parameters()[5]; }


// =================================================================================
// =========================SolidFactory============================================

DDSolid DDSolidFactory::polycone(const DDName & name, double startPhi, double deltaPhi,
                  const std::vector<double> & z,
		  const std::vector<double> & rmin,
		  const std::vector<double> & rmax) 
{
  return DDSolid(name, new DDI::Polycone(startPhi, deltaPhi, z, rmin, rmax));
}


DDSolid DDSolidFactory::polycone(const DDName & name, double startPhi, double deltaPhi,
                  const std::vector<double> & z,
		  const std::vector<double> & r)
{
  return DDSolid(name, new DDI::Polycone(startPhi, deltaPhi, z, r));
}   		     		  


DDSolid DDSolidFactory::polyhedra(const DDName & name,
                     int sides,
                     double startPhi,
                     double deltaPhi,
                     const std::vector<double> & z,
		     const std::vector<double> & rmin,
		     const std::vector<double> & rmax)
{		
  return DDSolid(name, new DDI::Polyhedra(sides,  startPhi, deltaPhi, z, rmin,rmax));     
}


DDSolid  DDSolidFactory::polyhedra(const DDName & name,
                     int sides,
                     double startPhi,
                     double deltaPhi,
		     const std::vector<double> & z,
		     const std::vector<double> & r)
{
  return DDSolid(name, new DDI::Polyhedra(sides,  startPhi, deltaPhi, z, r));
}


DDSolid DDSolidFactory::unionSolid(const DDName & name,
                    const DDSolid & a, const DDSolid & b,
		    const DDTranslation & t,
		    const DDRotation & r)
{
  return DDSolid(name, new DDI::Union(a,b,t,r));
}


DDSolid DDSolidFactory::subtraction(const DDName & name,
                    const DDSolid & a, const DDSolid & b,
		    const DDTranslation & t,
		    const DDRotation & r)
{
  return DDSolid(name, new DDI::Subtraction(a,b,t,r));
}


DDSolid DDSolidFactory::intersection(const DDName & name,
                    const DDSolid & a, const DDSolid & b,
		    const DDTranslation & t,
		    const DDRotation & r)
{
  return DDSolid(name, new DDI::Intersection(a,b,t,r));
}


DDSolid DDSolidFactory::trap(const DDName & name,
                     double pDz,
	             double pTheta, double pPhi,
	             double pDy1, double pDx1, double pDx2,
	             double pAlp1,
	             double pDy2, double pDx3, double pDx4,
	             double pAlp2)
{
  return DDSolid(name, new DDI::Trap(pDz, pTheta, pPhi,
                                     pDy1, pDx1, pDx2, pAlp1,
				     pDy2, pDx3, pDx4, pAlp2));
}	    		     


DDSolid DDSolidFactory::pseudoTrap(const DDName & name,
                          double pDx1, /**< Half-length along x at the surface positioned at -dz */
			     double pDx2, /**<  Half-length along x at the surface positioned at +dz */
			     double pDy1, /**<  Half-length along y at the surface positioned at -dz */
			     double pDy2, /**<  Half-length along y at the surface positioned at +dz */
			     double pDz, /**< Half of the height of the pseudo trapezoid along z */
			     double radius, /**< radius of the cut-out (negative sign) or rounding (pos. sign) */
			     bool atMinusZ /**< if true, the cut-out or rounding is applied at -dz, else at +dz */
			     )
{
   return DDSolid(name, new DDI::PseudoTrap(pDx1, pDx2, pDy1, pDy2, pDz, radius, atMinusZ));
}

DDSolid DDSolidFactory::truncTubs(const DDName & name,
                                  double zHalf, /**< half-length of the z-axis */
				  double rIn, /**< inner radius of the tube-section */
				  double rOut, /**< outer radius of the tube-section */
				  double startPhi, /**< starting angle of the tube-section */
				  double deltaPhi, /**< spanning angle of the tube-section */
				  double cutAtStart, /**< tructation at startPhi side */
				  double cutAtDelta, /**< truncation at deltaPhi side */
				  bool cutInside /**< */)
{
  return DDSolid(name, new DDI::TruncTubs(zHalf,rIn,rOut,startPhi,deltaPhi,cutAtStart,cutAtDelta,cutInside));
}				  

DDSolid DDSolidFactory::cons(const DDName & name,
                     double zhalf,
	   	     double rInMinusZ,	      	      
		     double rOutMinusZ,
		     double rInPlusZ,
		     double rOutPlusZ,
		     double phiFrom,
		     double deltaPhi)
{
  return DDSolid(name, new DDI::Cons(zhalf,
                                     rInMinusZ, rOutMinusZ,
				     rInPlusZ, rOutPlusZ,
				     phiFrom, deltaPhi));
}		     

DDSolid DDSolidFactory::torus(const DDName & name,
			      double rMin,
			      double rMax,
			      double rTorus,
			      double startPhi,
			      double deltaPhi)
{
  return DDSolid(name, new DDI::Torus(rMin, rMax, rTorus, startPhi, deltaPhi));
}		     

DDSolid DDSolidFactory::tubs(const DDName & name,
                             double zhalf,
		             double rIn, double rOut,	      	      
		             double phiFrom, double deltaPhi)
{		     
  return DDSolid(name, new DDI::Tubs(zhalf,rIn,rOut,phiFrom,deltaPhi));
}


DDSolid DDSolidFactory::sphere(const DDName & name,
                     double innerRadius,
	   	     double outerRadius,	      	      
		     double startPhi,
		     double deltaPhi,
		     double startTheta,
		     double deltaTheta)
{
  return DDSolid(name, new DDI::Sphere(innerRadius, outerRadius, 
				       startPhi, deltaPhi,
				       startTheta, deltaTheta));
}		     

DDSolid DDSolidFactory::orb(const DDName & name, double radius)
{
  return DDSolid(name, new DDI::Orb(radius));
}		     

DDSolid DDSolidFactory::ellipticalTube(const DDName & name,
				       double xSemiAxis, double ySemiAxis, double zHeight)
{
  return DDSolid(name, new DDI::EllipticalTube(xSemiAxis, ySemiAxis, zHeight));
}		     

DDSolid DDSolidFactory::ellipsoid(const DDName & name,
				  double  xSemiAxis,
				  double  ySemiAxis,
				  double  zSemiAxis,
				  double  zBottomCut,
				  double  zTopCut
				  )
  
{
  return DDSolid(name, new DDI::Ellipsoid( xSemiAxis,
					   ySemiAxis,
					   zSemiAxis,
					   zBottomCut,
					   zTopCut
					  ));
}		     

DDSolid DDSolidFactory::parallelepiped(const DDName & name,
				       double xHalf, double yHalf, double zHalf,
				       double alpha, double theta, double phi)
{
  return DDSolid(name, new DDI::Parallelepiped(xHalf, yHalf, zHalf,
					       alpha, theta, phi));
}		     

DDSolid DDSolidFactory::shapeless(const DDName & name)
{
  return DDSolid(name, new DDI::Shapeless());
}  


DDSolid DDSolidFactory::reflection(const DDName & name,
                                   const DDSolid & s)
{
  return DDSolid(name, new DDI::Reflection(s));
}				   

/*
DDSolid       DDcons(const DDName & name,
                     double zhalf,
	   	     double rInMinusZ,	      	      
		     double rOutMinusZ,
		     double rInPlusZ,
		     double rOutPlusZ,
		     double phiFrom,
		     double deltaPhi)
{		     
  return new DDConsImpl(name, zhalf, rInMinusZ, rOutMinusZ, rInPlusZ, rOutPlusZ,phiFrom, deltaPhi);
}

		     		     

DDSolid       DDtubs(const DDName & name,
                     double zhalf,
		     double rIn, double rOut,	      	      
		     double phiFrom, double deltaPhi)
{		     
  return new DDTubsImpl(name, zhalf,rIn,rOut,phiFrom,deltaPhi);
}
		     

DDSolid       DDtrap(const DDName & name,
                     double pDz,
	             double pTheta, double pPhi,
	             double pDy1, double pDx1, double pDx2,
	             double pAlp1,
	             double pDy2, double pDx3, double pDx4,
	             double pAlp2)
{		     
 return new DDTrapImpl(name, pDz, pTheta, pPhi, pDy1, pDx1, pDx2, pAlp1, pDy2, pDx3, pDx4, pAlp2); 
}
		     
                     
DDSolid  DDshapeless(const DDName & name)
{		     
  return new DDShapelessImpl(name);
}


DDSolid      DDunion(const DDName & name,
                     const DDSolid & a, 
                     const DDSolid & b, 
	   	     const DDRotation & r,
		     const DDTranslation & t)

{		     
  return new DDUnionImpl( name, a, b,r,t);
}
		     
		     
DDSolid      DDsubtraction(const DDName & name,
                           const DDSolid & a, 
                           const DDSolid & b, 
	   	           const DDRotation & r,
		           const DDTranslation & t)

{		     
  return new DDSubtractionImpl( name, a, b,r,t); 
}


DDSolid      DDintersection(const DDName & name,
                            const DDSolid & a, 
                            const DDSolid & b, 
	   	            const DDRotation & r,
		            const DDTranslation & t)

{		     
  return new DDIntersectionImpl( name, a, b,r,t); 
}


DDSolid   DDreflectionSolid(const DDName & name,
                            const DDSolid & unreflected)
{			    						
  return new DDReflectionSolidImpl( name, unreflected );
}

*/   
