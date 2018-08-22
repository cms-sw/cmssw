#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"

#include <ostream>
#include <string>
#include <array>

#include "DetectorDescription/Core/src/Boolean.h"
#include "DetectorDescription/Core/src/Box.h"
#include "DetectorDescription/Core/src/Cons.h"
#include "DetectorDescription/Core/src/EllipticalTube.h"
#include "DetectorDescription/Core/src/ExtrudedPolygon.h"
#include "DetectorDescription/Core/src/Polycone.h"
#include "DetectorDescription/Core/src/Polyhedra.h"
#include "DetectorDescription/Core/src/PseudoTrap.h"
#include "DetectorDescription/Core/src/Shapeless.h"
#include "DetectorDescription/Core/src/Solid.h"
#include "DetectorDescription/Core/src/Sphere.h"
#include "DetectorDescription/Core/src/Torus.h"
#include "DetectorDescription/Core/src/Trap.h"
#include "DetectorDescription/Core/src/TruncTubs.h"
#include "DetectorDescription/Core/src/Tubs.h"
#include "DetectorDescription/Core/src/CutTubs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using DDI::Solid;

std::ostream&
operator<<(std::ostream& os, const DDSolidShape s)
{
  return os << "DDSolidShape index:" << static_cast<int>(s) << ", name: " << DDSolidShapesName::name(s);
}

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

DDSolid::DDSolid()
  : DDBase< DDName, std::unique_ptr< Solid >>() { }

DDSolid::DDSolid( const DDName & name )
  : DDBase< DDName, std::unique_ptr< Solid >>()
{
  create( name );
}

DDSolid::DDSolid( const DDName & name, std::unique_ptr< Solid > solid )
  : DDBase< DDName, std::unique_ptr< Solid >>()
{
  create( name, std::move( solid ));
}

DDSolid::DDSolid( const DDName & name, DDSolidShape shape, const std::vector<double> & pars )
{
  std::unique_ptr<DDI::Solid> solid( nullptr );
  std::vector<double> dummy;
  switch( shape ) {
  case DDSolidShape::ddbox:
    solid = std::move( std::make_unique< DDI::Box >( 0, 0, 0 ));
    break;
  case DDSolidShape::ddtubs:
    solid = std::move( std::make_unique< DDI::Tubs >( 0, 0, 0, 0, 0 ));
    break;
  case DDSolidShape::ddcons:
    solid = std::move( std::make_unique< DDI::Cons >( 0, 0, 0, 0, 0, 0, 0 ));
    break;
  case DDSolidShape::ddpseudotrap:
    solid = std::move( std::make_unique< DDI::PseudoTrap >( 0, 0, 0, 0, 0, 0, false ));
    break;
  case DDSolidShape::ddshapeless:
    solid = std::move( std::make_unique< DDI::Shapeless >( ));
    break;
  case DDSolidShape::ddtrap:
    solid = std::move( std::make_unique< DDI::Trap >( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ));
    break;
  case DDSolidShape::ddpolyhedra_rz:
    solid = std::move( std::make_unique< DDI::Polyhedra >( 0, 0, 0, dummy, dummy ));
    break;
  case DDSolidShape::ddpolyhedra_rrz:
    solid = std::move( std::make_unique< DDI::Polyhedra >( 0, 0, 0, dummy, dummy, dummy ));
    break;
  case DDSolidShape::ddpolycone_rz:
    solid = std::move( std::make_unique< DDI::Polycone >( 0, 0, dummy, dummy ));
    break;
  case DDSolidShape::ddpolycone_rrz:
    solid = std::move( std::make_unique< DDI::Polycone >( 0, 0, dummy, dummy, dummy ));
    break;			
  case DDSolidShape::ddtrunctubs:
    solid = std::move( std::make_unique< DDI::TruncTubs >( 0, 0, 0, 0, 0, 0, 0, false ));
    break;
  case DDSolidShape::ddtorus:
    solid = std::move( std::make_unique< DDI::Torus >( 0, 0, 0, 0, 0 ));
    break;
  case DDSolidShape::ddsphere:
    solid = std::move( std::make_unique< DDI::Sphere >( 0, 0, 0, 0, 0, 0 ));
    break;
  case DDSolidShape::ddellipticaltube:
    solid = std::move( std::make_unique< DDI::EllipticalTube >( 0, 0, 0 ));
    break;
  case DDSolidShape::ddcuttubs:
    solid = std::move( std::make_unique< DDI::CutTubs >( 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., -1. ));
    break;
  case DDSolidShape::ddextrudedpolygon:
    solid = std::move( std::make_unique< DDI::ExtrudedPolygon >( dummy, dummy, dummy, dummy, dummy, dummy ));
    break;
  default:
    throw cms::Exception( "DDException" )
		<< "DDSolid::DDSolid( DDName, DDSolidShape, std::vector<double> ): wrong shape.";   
  }
  solid->setParameters( pars );
		create( name, std::move( solid ));
}

double
DDSolid::volume() const
{
  return rep().volume();
}

DDSolidShape
DDSolid::shape() const
{
  return rep().shape();
}

const std::vector<double> &
DDSolid::parameters() const
{ 
  return rep().parameters(); 
}

DDTrap::DDTrap(const DDSolid & s) : DDSolid(s)
{
  if (s.shape() != DDSolidShape::ddtrap) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is no DDTrap.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double
DDTrap::halfZ() const { return rep().parameters()[0]; }

double
DDTrap::theta() const { return rep().parameters()[1]; }

double
DDTrap::phi() const { return rep().parameters()[2]; }

double
DDTrap::y1() const { return rep().parameters()[3]; }

double
DDTrap::x1() const { return rep().parameters()[4]; }

double
DDTrap::x2() const { return rep().parameters()[5]; }

double
DDTrap::alpha1() const { return rep().parameters()[6]; }

double
DDTrap::y2() const { return rep().parameters()[7]; }

double
DDTrap::x3() const { return rep().parameters()[8]; }

double
DDTrap::x4() const { return rep().parameters()[9]; }

double
DDTrap::alpha2() const { return rep().parameters()[10]; }

DDTruncTubs::DDTruncTubs(const DDSolid & s) : DDSolid(s)
{
  if (s.shape() != DDSolidShape::ddtrunctubs) {
    edm::LogError ("DDSolid") << "s.shape()=" << s.shape() << "  " << s.name() << std::endl;
    std::string ex = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is no DDTruncTubs\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double
DDTruncTubs::zHalf() const { return rep().parameters()[0];}

double
DDTruncTubs::rIn() const { return rep().parameters()[1];}

double
DDTruncTubs::rOut() const { return rep().parameters()[2];}

double
DDTruncTubs::startPhi() const { return rep().parameters()[3];}

double
DDTruncTubs::deltaPhi() const { return rep().parameters()[4];}

double
DDTruncTubs::cutAtStart() const { return rep().parameters()[5];}

double
DDTruncTubs::cutAtDelta() const { return rep().parameters()[6];}

bool
DDTruncTubs::cutInside() const { return bool(rep().parameters()[7]);}

DDPseudoTrap::DDPseudoTrap(const DDSolid & s) : DDSolid(s)
{
  if (s.shape() != DDSolidShape::ddpseudotrap) {
    std::string ex = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is no DDPseudoTrap\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double
DDPseudoTrap::halfZ() const { return rep().parameters()[4]; }

double
DDPseudoTrap::x1() const { return rep().parameters()[0]; }

double
DDPseudoTrap::x2() const { return rep().parameters()[1]; }

double
DDPseudoTrap::y1() const { return rep().parameters()[2]; }

double
DDPseudoTrap::y2() const { return rep().parameters()[3]; }

double
DDPseudoTrap::radius() const { return rep().parameters()[5]; }

bool
DDPseudoTrap::atMinusZ() const { return rep().parameters()[6]; }

DDBox::DDBox(const DDSolid & s) 
 : DDSolid(s) 
{ 
 if (s.shape() != DDSolidShape::ddbox) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDBox.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
 }
}

double
DDBox::halfX() const
{ return rep().parameters()[0]; }

double
DDBox::halfY() const
{ return rep().parameters()[1]; }

double
DDBox::halfZ() const
{ return rep().parameters()[2]; }

DDShapelessSolid::DDShapelessSolid (const DDSolid & s) : DDSolid(s)
{
  if (s.shape() != DDSolidShape::ddshapeless) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDShapelessSolid.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

DDUnion::DDUnion(const DDSolid & s) 
  : DDBooleanSolid(s)
{
  if (s.shape() != DDSolidShape::ddunion) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDUnion.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

DDIntersection::DDIntersection(const DDSolid & s) 
  : DDBooleanSolid(s)
{
  if (s.shape() != DDSolidShape::ddunion) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDIntersection.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

DDSubtraction::DDSubtraction(const DDSolid & s) 
  : DDBooleanSolid(s)
{
  if (s.shape() != DDSolidShape::ddsubtraction) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is no DDSubtraction.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

DDPolySolid::DDPolySolid(const DDSolid & s)
  : DDSolid(s)
{ }

std::vector<double>
DDPolySolid::getVec (const size_t& which, 
		     const size_t& offset,
		     const size_t& numVecs) const
{
  // which:  first second or third std::vector 
  // offset: number of non-std::vector components before std::vectors start
  if(( rep().parameters().size() - offset) % numVecs != 0 ) {
    edm::LogError ("DDSolid") << "Could not find equal sized components of std::vectors in a PolySolid description."
			      << "rep().parameters().size()=" << rep().parameters().size() << "  numVecs=" << numVecs
			      << "  offset=" << offset << std::endl;
  }
  std::vector<double> tvec;
  for( size_t i = offset + which; i < rep().parameters().size(); i = i + numVecs ) {
    tvec.emplace_back( rep().parameters()[i]);
  }               
  return tvec;
}

DDPolycone::DDPolycone(const DDSolid & s)
  : DDPolySolid(s)
{
  if( s.shape() != DDSolidShape::ddpolycone_rz && s.shape() != DDSolidShape::ddpolycone_rrz ) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDPolycone.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double
DDPolycone::startPhi() const { return rep().parameters()[0]; }

double
DDPolycone::deltaPhi() const { return rep().parameters()[1]; }

std::vector<double> DDPolycone::rVec() const {
  std::vector<double> tvec;
  if (shape() == DDSolidShape::ddpolycone_rz)
    tvec = getVec(1, 2, 2);
  return tvec; 
}

std::vector<double>
DDPolycone::zVec() const {
  if (shape() == DDSolidShape::ddpolycone_rz)
    return getVec(0, 2, 2);
  else // (shape() == DDSolidShape::ddpolycone_rrz)
    return getVec(0, 2, 3);
}

std::vector<double>
DDPolycone::rMinVec() const {
  std::vector<double> tvec;
  if (shape() == DDSolidShape::ddpolycone_rrz)
    tvec = getVec(1, 2, 3);
  return tvec; 
}

std::vector<double>
DDPolycone::rMaxVec() const {
  std::vector<double> tvec;
  if (shape() == DDSolidShape::ddpolycone_rrz)
    tvec = getVec(2, 2, 3);
  return tvec; 
}

DDPolyhedra::DDPolyhedra(const DDSolid & s)
  : DDPolySolid(s)
{
  if (s.shape() != DDSolidShape::ddpolyhedra_rz && s.shape() != DDSolidShape::ddpolyhedra_rrz) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDPolyhedra.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

int
DDPolyhedra::sides() const { return int(rep().parameters()[0]); }

double
DDPolyhedra::startPhi() const { return rep().parameters()[1]; }

double
DDPolyhedra::deltaPhi() const { return rep().parameters()[2]; }

std::vector<double>
DDPolyhedra::rVec() const {
  std::vector<double> tvec;
  if (shape() == DDSolidShape::ddpolyhedra_rz)
    tvec = getVec(1, 3, 2);
  return tvec;
}

std::vector<double>
DDPolyhedra::zVec() const {
  if (shape() == DDSolidShape::ddpolyhedra_rz)
    return getVec(0, 3, 2);
  else // (shape() == ddpolycone_rrz)
    return getVec(0, 3, 3);
}

std::vector<double>
DDPolyhedra::rMinVec() const {
  std::vector<double> tvec;
  if (shape() == DDSolidShape::ddpolyhedra_rrz)
    tvec = getVec(1, 3, 3);
  return tvec;
}

std::vector<double>
DDPolyhedra::rMaxVec() const {
  std::vector<double> tvec;
  if (shape() == DDSolidShape::ddpolyhedra_rrz)
    tvec = getVec(2, 3, 3);
  return tvec;
}

DDExtrudedPolygon::DDExtrudedPolygon(const DDSolid & s)
  : DDPolySolid(s)
{
  if( s.shape() != DDSolidShape::ddextrudedpolygon ) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDExtrudedPolygon.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

auto
DDExtrudedPolygon::xyPointsSize( void ) const -> std::size_t
{
  // Compute the size of the X and Y coordinate vectors
  // which define the vertices of the outlined polygon
  // defined in clock-wise order

  return ( rep().parameters().size() - 4 * zSectionsSize()) * 0.5;
}

auto
DDExtrudedPolygon::zSectionsSize( void ) const -> std::size_t
{
  // The first parameters element stores a size of the four equal size vectors
  // which form the ExtrudedPolygon shape Z sections:
  // 
  //    * first: Z coordinate of the XY polygone plane,
  //    * second and third: x and y offset of the polygone in the XY plane,
  //    * fourth: the polygone scale in each XY plane
  //
  // The Z sections defined by the z position in an increasing order.

  return rep().parameters()[0];
}

std::vector<double>
DDExtrudedPolygon::xVec( void ) const
{
  return std::vector<double>( rep().parameters().begin() + 1,
			      rep().parameters().begin() + 1 + xyPointsSize());
}

std::vector<double>
DDExtrudedPolygon::yVec( void ) const
{
  return std::vector<double>( rep().parameters().begin() + 1 + xyPointsSize(),
			      rep().parameters().begin() + 1 + 2 * xyPointsSize());
}

std::vector<double>
DDExtrudedPolygon::zVec( void ) const
{
  return std::vector<double>( rep().parameters().end() - 4 * zSectionsSize(),
			      rep().parameters().end() - 3 * zSectionsSize());
}

std::vector<double>
DDExtrudedPolygon::zxVec( void ) const
{
  return std::vector<double>( rep().parameters().end() - 3 * zSectionsSize(),
			      rep().parameters().end() - 2 * zSectionsSize());
}

std::vector<double>
DDExtrudedPolygon::zyVec( void ) const
{
  return std::vector<double>( rep().parameters().end() - 2 * zSectionsSize(),
			      rep().parameters().end() - zSectionsSize());
}

std::vector<double>
DDExtrudedPolygon::zscaleVec( void ) const
{
  return std::vector<double>( rep().parameters().end() - zSectionsSize(),
			      rep().parameters().end());
}

DDCons::DDCons(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != DDSolidShape::ddcons) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDCons.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double
DDCons::zhalf() const { return rep().parameters()[0]; }

double
DDCons::rInMinusZ() const { return rep().parameters()[1]; }

double
DDCons::rOutMinusZ () const { return rep().parameters()[2]; }

double
DDCons::rInPlusZ() const { return rep().parameters()[3]; }

double
DDCons::rOutPlusZ() const { return rep().parameters()[4]; }

double
DDCons::phiFrom() const { return rep().parameters()[5]; }

double
DDCons::deltaPhi() const { return rep().parameters()[6]; }

DDTorus::DDTorus(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != DDSolidShape::ddtorus) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDTorus.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double
DDTorus::rMin() const { return rep().parameters()[0]; }

double
DDTorus::rMax() const { return rep().parameters()[1]; }

double
DDTorus::rTorus () const { return rep().parameters()[2]; }

double
DDTorus::startPhi() const { return rep().parameters()[3]; }

double
DDTorus::deltaPhi() const { return rep().parameters()[4]; }

DDTubs::DDTubs(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != DDSolidShape::ddtubs) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDTubs.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double
DDTubs::zhalf() const { return rep().parameters()[0]; }

double
DDTubs::rIn() const { return rep().parameters()[1]; }

double
DDTubs::rOut() const { return rep().parameters()[2]; }

double
DDTubs::startPhi() const { return rep().parameters()[3]; }

double
DDTubs::deltaPhi() const { return rep().parameters()[4]; }

DDBooleanSolid::DDBooleanSolid( const DDSolid &s )
  : DDSolid( s ),
    boolean_( static_cast< DDI::BooleanSolid& >( s.rep()))
{}

DDRotation
DDBooleanSolid::rotation() const
{
  return boolean_.r();
}

DDTranslation
DDBooleanSolid::translation() const
{
  return boolean_.t();
}

DDSolid
DDBooleanSolid::solidA() const
{
  return boolean_.a();
}

DDSolid
DDBooleanSolid::solidB() const
{
  return boolean_.b();
}

DDSphere::DDSphere( const DDSolid& s ) 
  : DDSolid( s ) {
  if( s.shape() != DDSolidShape::ddsphere ) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDSphere (or sphere section).\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double
DDSphere::innerRadius() const { return rep().parameters()[0]; }

double
DDSphere::outerRadius() const { return rep().parameters()[1]; }

double
DDSphere::startPhi () const { return rep().parameters()[2]; }

double
DDSphere::deltaPhi() const { return rep().parameters()[3]; }

double
DDSphere::startTheta() const { return rep().parameters()[4]; }

double
DDSphere::deltaTheta() const { return rep().parameters()[5]; }

DDEllipticalTube::DDEllipticalTube(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != DDSolidShape::ddellipticaltube) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDEllipticalTube.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double
DDEllipticalTube::xSemiAxis() const { return rep().parameters()[0]; }

double
DDEllipticalTube::ySemiAxis() const { return rep().parameters()[1]; }

double
DDEllipticalTube::zHeight() const { return rep().parameters()[2]; }

DDCutTubs::DDCutTubs(const DDSolid& s) 
  : DDSolid(s) {
  if (s.shape() != DDSolidShape::ddcuttubs) {
    std::string ex  = "Solid [" + s.name().ns() + ":" + s.name().name() + "] is not a DDCutTubs.\n";
    ex = ex + "Use a different solid interface!";
    throw cms::Exception("DDException") << ex;
  }
}

double
DDCutTubs::zhalf() const { return rep().parameters()[0]; }

double
DDCutTubs::rIn() const { return rep().parameters()[1]; }

double
DDCutTubs::rOut() const { return rep().parameters()[2]; }

double
DDCutTubs::startPhi() const { return rep().parameters()[3]; }

double
DDCutTubs::deltaPhi() const { return rep().parameters()[4]; }

std::array<double, 3>
DDCutTubs::lowNorm( void ) const {
  return std::array<double, 3>{{rep().parameters()[5],rep().parameters()[6],rep().parameters()[7]}};
}

std::array<double, 3>
DDCutTubs::highNorm( void ) const {
  return std::array<double, 3>{{rep().parameters()[8],rep().parameters()[9],rep().parameters()[10]}};
}
    
// =================================================================================
// =========================SolidFactory============================================

DDSolid
DDSolidFactory::box( const DDName & name, 
		     double xHalf, 
		     double yHalf, 
		     double zHalf )
{
  return DDSolid( name, std::make_unique< DDI::Box >( xHalf, yHalf, zHalf ));
}

DDSolid
DDSolidFactory::polycone( const DDName & name, double startPhi, double deltaPhi,
			  const std::vector<double> & z,
			  const std::vector<double> & rmin,
			  const std::vector<double> & rmax ) 
{
  return DDSolid( name, std::make_unique< DDI::Polycone >( startPhi, deltaPhi, z, rmin, rmax ));
}

DDSolid
DDSolidFactory::polycone( const DDName & name, double startPhi, double deltaPhi,
			  const std::vector<double> & z,
			  const std::vector<double> & r )
{
  return DDSolid(name, std::make_unique< DDI::Polycone >( startPhi, deltaPhi, z, r ));
}   		     		  

DDSolid
DDSolidFactory::polyhedra( const DDName & name,
			   int sides,
			   double startPhi,
			   double deltaPhi,
			   const std::vector<double> & z,
			   const std::vector<double> & rmin,
			   const std::vector<double> & rmax )
{
  return DDSolid( name, std::make_unique< DDI::Polyhedra >( sides, startPhi, deltaPhi, z, rmin, rmax ));     
}

DDSolid
DDSolidFactory::polyhedra( const DDName & name,
			   int sides,
			   double startPhi,
			   double deltaPhi,
			   const std::vector<double> & z,
			   const std::vector<double> & r )
{
  return DDSolid( name, std::make_unique< DDI::Polyhedra >( sides, startPhi, deltaPhi, z, r ));
}

DDSolid
DDSolidFactory::extrudedpolygon( const DDName & name,
				 const std::vector<double> & x,
				 const std::vector<double> & y,
				 const std::vector<double> & z,
				 const std::vector<double> & zx,
				 const std::vector<double> & zy,
				 const std::vector<double> & zscale )
{
  return DDSolid( name, std::make_unique< DDI::ExtrudedPolygon >( x, y, z, zx, zy, zscale ));
}

DDSolid
DDSolidFactory::unionSolid( const DDName & name,
			    const DDSolid & a, const DDSolid & b,
			    const DDTranslation & t,
			    const DDRotation & r )
{
  return DDSolid( name, std::make_unique< DDI::Union >( a, b, t, r ));
}

DDSolid
DDSolidFactory::subtraction( const DDName & name,
			     const DDSolid & a, const DDSolid & b,
			     const DDTranslation & t,
			     const DDRotation & r )
{
  return DDSolid( name, std::make_unique< DDI::Subtraction >( a, b, t, r ));
}

DDSolid
DDSolidFactory::intersection( const DDName & name,
			      const DDSolid & a, const DDSolid & b,
			      const DDTranslation & t,
			      const DDRotation & r )
{
  return DDSolid( name, std::make_unique< DDI::Intersection >( a, b, t, r ));
}

DDSolid
DDSolidFactory::trap( const DDName & name,
		      double pDz,
		      double pTheta, double pPhi,
		      double pDy1, double pDx1, double pDx2,
		      double pAlp1,
		      double pDy2, double pDx3, double pDx4,
		      double pAlp2 )
{
  return DDSolid( name, std::make_unique< DDI::Trap >( pDz, pTheta, pPhi,
						       pDy1, pDx1, pDx2, pAlp1,
						       pDy2, pDx3, pDx4, pAlp2 ));
}

DDSolid
DDSolidFactory::pseudoTrap( const DDName & name,
			    double pDx1, /**< Half-length along x at the surface positioned at -dz */
			    double pDx2, /**<  Half-length along x at the surface positioned at +dz */
			    double pDy1, /**<  Half-length along y at the surface positioned at -dz */
			    double pDy2, /**<  Half-length along y at the surface positioned at +dz */
			    double pDz, /**< Half of the height of the pseudo trapezoid along z */
			    double radius, /**< radius of the cut-out (negative sign) or rounding (pos. sign) */
			    bool atMinusZ /**< if true, the cut-out or rounding is applied at -dz, else at +dz */
			    )
{
  return DDSolid( name, std::make_unique< DDI::PseudoTrap >( pDx1, pDx2, pDy1, pDy2, pDz,
							     radius, atMinusZ ));
}

DDSolid
DDSolidFactory::truncTubs( const DDName & name,
			   double zHalf, /**< half-length of the z-axis */
			   double rIn, /**< inner radius of the tube-section */
			   double rOut, /**< outer radius of the tube-section */
			   double startPhi, /**< starting angle of the tube-section */
			   double deltaPhi, /**< spanning angle of the tube-section */
			   double cutAtStart, /**< tructation at startPhi side */
			   double cutAtDelta, /**< truncation at deltaPhi side */
			   bool cutInside )
{
  return DDSolid( name, std::make_unique< DDI::TruncTubs >( zHalf, rIn, rOut,
							    startPhi, deltaPhi,
							    cutAtStart, cutAtDelta,
							    cutInside ));
}				  

DDSolid
DDSolidFactory::cons( const DDName & name,
		      double zhalf,
		      double rInMinusZ,	      	      
		      double rOutMinusZ,
		      double rInPlusZ,
		      double rOutPlusZ,
		      double phiFrom,
		      double deltaPhi )
{
  return DDSolid( name, std::make_unique< DDI::Cons >( zhalf,
						       rInMinusZ, rOutMinusZ,
						       rInPlusZ, rOutPlusZ,
						       phiFrom, deltaPhi ));
}		     

DDSolid
DDSolidFactory::torus( const DDName & name,
		       double rMin,
		       double rMax,
		       double rTorus,
		       double startPhi,
		       double deltaPhi )
{
  return DDSolid( name, std::make_unique< DDI::Torus >( rMin, rMax, rTorus, startPhi, deltaPhi ));
}

DDSolid
DDSolidFactory::tubs( const DDName & name,
		      double zhalf,
		      double rIn, double rOut,	      	      
		      double phiFrom, double deltaPhi )
{
  return DDSolid( name, std::make_unique< DDI::Tubs >( zhalf, rIn, rOut, phiFrom, deltaPhi ));
}

DDSolid
DDSolidFactory::cuttubs( const DDName & name,
			 double zhalf,
			 double rIn, double rOut,	      	      
			 double phiFrom, double deltaPhi,
			 double lx, double ly, double lz,
			 double tx, double ty, double tz )
{
  return DDSolid( name, std::make_unique< DDI::CutTubs >( zhalf, rIn, rOut,
							  phiFrom, deltaPhi,
							  lx, ly, lz,
							  tx, ty, tz ));
}

DDSolid
DDSolidFactory::sphere( const DDName & name,
			double innerRadius,
			double outerRadius,	      	      
			double startPhi,
			double deltaPhi,
			double startTheta,
			double deltaTheta )
{
  return DDSolid( name, std::make_unique< DDI::Sphere >( innerRadius, outerRadius, 
							 startPhi, deltaPhi,
							 startTheta, deltaTheta ));
}

DDSolid
DDSolidFactory::ellipticalTube( const DDName & name,
				double xSemiAxis, double ySemiAxis, double zHeight )
{
  return DDSolid( name, std::make_unique< DDI::EllipticalTube >( xSemiAxis, ySemiAxis, zHeight ));
}

DDSolid
DDSolidFactory::shapeless( const DDName & name )
{
  return DDSolid( name, std::make_unique< DDI::Shapeless >());
}
