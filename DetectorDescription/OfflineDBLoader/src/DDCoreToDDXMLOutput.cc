#include <DetectorDescription/Core/interface/DDMaterial.h>
#include <DetectorDescription/Core/interface/DDPartSelection.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>
#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include <DetectorDescription/OfflineDBLoader/interface/DDCoreToDDXMLOutput.h>
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDValuePair.h"
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Rotation3D.h"

#include <cstddef>
#include <iomanip>
#include <vector>

using namespace dd::operators;

void 
DDCoreToDDXMLOutput::solid( const DDSolid& solid, std::ostream& xos ) 
{
   switch( solid.shape()) 
   {
      case DDSolidShape::ddunion:
      case DDSolidShape::ddsubtraction:
      case DDSolidShape::ddintersection: 
      {      
         DDBooleanSolid rs( solid );
         if( solid.shape() == DDSolidShape::ddunion ) 
         {
            xos << "<UnionSolid ";
         } 
         else if( solid.shape() == DDSolidShape::ddsubtraction ) 
         {
            xos << "<SubtractionSolid ";
         } 
         else if( solid.shape() == DDSolidShape::ddintersection ) 
         {
            xos << "<IntersectionSolid ";
         }
         xos << "name=\""  << rs.toString() << "\">"  << std::endl;
         // if translation is == identity there are no parameters.
         // if there is no rotation the name will be ":"
         xos << "<rSolid name=\""  << rs.solidA().toString() << "\"/>" << std::endl;
         xos << "<rSolid name=\""  << rs.solidB().toString() << "\"/>" << std::endl;
         xos << "<Translation x=\"" << rs.translation().X() << "*mm\"";
         xos << " y=\"" << rs.translation().Y() << "*mm\"";
         xos << " z=\"" << rs.translation().Z() << "*mm\"";
         xos << "/>" << std::endl; 
         std::string rotName = rs.rotation().toString();
         if( rotName == ":" ) 
         {
            rotName = "gen:ID";
         }
         xos << "<rRotation name=\""  << rs.rotation().toString() << "\"/>" << std::endl;
         if( solid.shape() == DDSolidShape::ddunion ) 
         {
            xos << "</UnionSolid>" << std::endl;
         } 
         else if( solid.shape() == DDSolidShape::ddsubtraction ) 
         {
            xos << "</SubtractionSolid>" << std::endl;
         } 
         else if( solid.shape() == DDSolidShape::ddintersection ) 
         {
            xos << "</IntersectionSolid>" << std::endl;
         }
         break;
      }
      case DDSolidShape::ddbox: 
      {
         //    <Box name="box1" dx="10*cm" dy="10*cm" dz="10*cm"/>
         DDBox rs(solid);
         xos << "<Box name=\""  << rs.toString()  << "\"" //<< rs.toString() << "\"" //
	     << " dx=\"" << rs.halfX() << "*mm\""
	     << " dy=\"" << rs.halfY() << "*mm\""
	     << " dz=\"" << rs.halfZ() << "*mm\"/>"
	     << std::endl;
         break;
      }
      case DDSolidShape::ddtubs: 
      {
         //      <Tubs name="TrackerSupportTubeNomex"         rMin="[SupportTubeR1]+[Tol]" 
         //            rMax="[SupportTubeR2]-[Tol]"           dz="[SupportTubeL]" 
         //            startPhi="0*deg"                       deltaPhi="360*deg"/>
         DDTubs rs(solid);
         xos << "<Tubs name=\""  << rs.toString() << "\""
	     << " rMin=\"" << rs.rIn() << "*mm\""
	     << " rMax=\"" << rs.rOut() << "*mm\""
	     << " dz=\"" << rs.zhalf() << "*mm\""
	     << " startPhi=\"" << CONVERT_TO( rs.startPhi(), deg ) << "*deg\""
	     << " deltaPhi=\"" << CONVERT_TO( rs.deltaPhi(), deg ) << "*deg\"/>"
	     << std::endl;
         break;
      }
      case DDSolidShape::ddtrap: 
      {
	//    <Trapezoid name="UpL_CSC_for_TotemT1_Plane_2_5_7" dz="[PCB_Epoxy_Thick_3P]/2."
	//      alp1="-[Up_Signal_Side_alpL_3P]" alp2="-[Up_Signal_Side_alpL_3P]"  
	//     bl1="[Max_Base_Signal_SideL_3P]/2." tl1="[Up_Min_Base_Signal_SideL_3P]/2." h1="[Up_Height_Signal_SideL_3P]/2."
	//     h2="[Up_Height_Signal_SideL_3P]/2." bl2="[Max_Base_Signal_SideL_3P]/2." tl2="[Up_Min_Base_Signal_SideL_3P]/2."/>
	DDTrap rs(solid);
	xos << "<Trapezoid name=\""  << rs.toString() << "\""
	    << " dz=\"" << rs.halfZ() << "*mm\""
	    << " theta=\"" << CONVERT_TO( rs.theta(), deg ) << "*deg\""
	    << " phi=\"" << CONVERT_TO( rs.phi(), deg ) << "*deg\""
	    << " h1=\"" << rs.y1() << "*mm\""
	    << " bl1=\"" << rs.x1() << "*mm\""
	    << " tl1=\"" << rs.x2() << "*mm\""
	    << " alp1=\"" << CONVERT_TO( rs.alpha1(), deg ) << "*deg\""
	    << " h2=\"" << rs.y2() << "*mm\""
	    << " bl2=\"" << rs.x3() << "*mm\""
	    << " tl2=\"" << rs.x4() << "*mm\""
	    << " alp2=\"" << CONVERT_TO( rs.alpha2(), deg ) << "*deg\"/>"
	    << std::endl;
	break;
      }
      case DDSolidShape::ddcons: 
      {
	DDCons rs(solid);
	xos << "<Cone name=\""  << rs.toString() << "\""
	    << " dz=\"" << rs.zhalf() << "*mm\""
	    << " rMin1=\"" << rs.rInMinusZ() << "*mm\""
	    << " rMax1=\"" << rs.rOutMinusZ() << "*mm\""
	    << " rMin2=\"" << rs.rInPlusZ() << "*mm\""
	    << " rMax2=\"" << rs.rOutPlusZ() << "*mm\""
	    << " startPhi=\"" << CONVERT_TO( rs.phiFrom(), deg ) << "*deg\""
	    << " deltaPhi=\"" << CONVERT_TO( rs.deltaPhi(), deg ) << "*deg\"/>"
	    << std::endl;
	break;
      }
      case DDSolidShape::ddpolycone_rz: 
      {
	DDPolycone rs(solid);
	xos << "<Polycone name=\""  << rs.toString() << "\""
	    << " startPhi=\"" << CONVERT_TO( rs.startPhi(), deg ) << "*deg\""
	    << " deltaPhi=\"" << CONVERT_TO( rs.deltaPhi(), deg ) << "*deg\">"
	    << std::endl;
	const std::vector<double> & zV(rs.zVec());
	const std::vector<double> & rV(rs.rVec());
	for ( size_t i = 0; i < zV.size(); ++i ) {
	  xos << "<RZPoint r=\"" << rV[i] << "*mm\""
	      << " z=\"" << zV[i] << "*mm\"/>"
	      << std::endl;
	}
	xos << "</Polycone>" << std::endl;
	break;
      }
      case DDSolidShape::ddpolyhedra_rz: 
      {
	DDPolyhedra rs(solid);
	xos << "<Polyhedra name=\""  << rs.toString() << "\""
	    << " numSide=\"" << rs.sides() << "\""
	    << " startPhi=\"" << CONVERT_TO( rs.startPhi(), deg ) << "*deg\""
	    << " deltaPhi=\"" << CONVERT_TO( rs.deltaPhi(), deg ) << "*deg\">"
	    << std::endl;
	const std::vector<double> & zV(rs.zVec());
	const std::vector<double> & rV(rs.rVec());
	for ( size_t i = 0; i < zV.size(); ++i ) {
	  xos << "<RZPoint r=\"" << rV[i] << "*mm\""
	      << " z=\"" << zV[i] << "*mm\"/>"
	      << std::endl;
	}
	xos << "</Polyhedra>" << std::endl;
	break;
      }
      case DDSolidShape::ddpolycone_rrz:
      {
	//   <Polycone name="OCMS" startPhi="0*deg" deltaPhi="360*deg" >
	//    <ZSection z="-[CMSZ1]"  rMin="[Rmin]"  rMax="[CMSR2]" />
	//    <ZSection z="-[HallZ]"  rMin="[Rmin]"  rMax="[CMSR2]" /> 
	//    <ZSection z="-[HallZ]"  rMin="[Rmin]"  rMax="[HallR]" />
	//    <ZSection z="[HallZ]"   rMin="[Rmin]"  rMax="[HallR]" />
	//    <ZSection z="[HallZ]"   rMin="[Rmin]"  rMax="[CMSR2]" />
	//    <ZSection z="[CMSZ1]"   rMin="[Rmin]"  rMax="[CMSR2]" />
	DDPolycone rs(solid);
	xos << "<Polycone name=\""  << rs.toString() << "\""
	    << " startPhi=\"" << CONVERT_TO( rs.startPhi(), deg ) << "*deg\""
	    << " deltaPhi=\"" << CONVERT_TO( rs.deltaPhi(), deg ) << "*deg\">"
	    << std::endl;
	const std::vector<double> & zV(rs.zVec());
	const std::vector<double> & rMinV(rs.rMinVec());
	const std::vector<double> & rMaxV(rs.rMaxVec());
	for ( size_t i = 0; i < zV.size(); ++i ) {
	  xos << "<ZSection z=\"" << zV[i] << "*mm\""
	      << " rMin=\"" << rMinV[i] << "*mm\""
	      << " rMax=\"" << rMaxV[i] << "*mm\"/>"
	      << std::endl;
	}
	xos << "</Polycone>" << std::endl;
	break;
      }
      case DDSolidShape::ddpolyhedra_rrz:
      {
	DDPolyhedra rs(solid);
	xos << "<Polyhedra name=\""  << rs.toString() << "\""
	    << " numSide=\"" << rs.sides() << "\""
	    << " startPhi=\"" << CONVERT_TO( rs.startPhi(), deg ) << "*deg\""
	    << " deltaPhi=\"" << CONVERT_TO( rs.deltaPhi(), deg ) << "*deg\">"
	    << std::endl;
	const std::vector<double> & zV(rs.zVec());
	const std::vector<double> & rMinV(rs.rMinVec());
	const std::vector<double> & rMaxV(rs.rMaxVec());
	for ( size_t i = 0; i < zV.size(); ++i ) {
	  xos << "<ZSection z=\"" << zV[i] << "*mm\""
	      << " rMin=\"" << rMinV[i] << "*mm\""
	      << " rMax=\"" << rMaxV[i] << "*mm\"/>"
	      << std::endl;
	}
	xos << "</Polyhedra>" << std::endl;
	break;
      }
      case DDSolidShape::ddpseudotrap:
      {
	// <PseudoTrap name="YE3_b" dx1="0.395967*m" dx2="1.86356*m" dy1="0.130*m" dy2="0.130*m" dz="2.73857*m" radius="-1.5300*m" atMinusZ="true"/> 
	DDPseudoTrap rs(solid);
	xos << "<PseudoTrap name=\""  << rs.toString() << "\""
	    << " dx1=\"" << rs.x1() << "*mm\""
	    << " dx2=\"" << rs.x2() << "*mm\""
	    << " dy1=\"" << rs.y1() << "*mm\""
	    << " dy2=\"" << rs.y2() << "*mm\""
	    << " dz=\"" << rs.halfZ() << "*mm\""
	    << " radius=\"" << rs.radius() << "*mm\""
	    << " atMinusZ=\"" << ( rs.atMinusZ() ? "true" : "false" ) << "\"/>"
	    << std::endl;
	break;
      }
      case DDSolidShape::ddtrunctubs:
      {
	// <TruncTubs name="trunctubs1" zHalf="50*cm" rMin="20*cm" rMax="40*cm"
	//                              startPhi="0*deg" deltaPhi="90*deg"
	//                              cutAtStart="25*cm" cutAtDelta="35*cm"/>
	DDTruncTubs rs(solid);
	xos << "<TruncTubs name=\""  << rs.toString() << "\""
	    << " zHalf=\"" << rs.zHalf() << "*mm\""
	    << " rMin=\"" << rs.rIn() << "*mm\""
	    << " rMax=\"" << rs.rOut() << "*mm\""
	    << " startPhi=\"" << CONVERT_TO( rs.startPhi(), deg ) << "*deg\""
	    << " deltaPhi=\"" << CONVERT_TO( rs.deltaPhi(), deg ) << "*deg\""
	    << " cutAtStart=\"" << rs.cutAtStart() << "*mm\""
	    << " cutAtDelta=\"" << rs.cutAtDelta() << "*mm\""
	    << " cutInside=\"" << ( rs.cutInside() ? "true" : "false" ) << "\"/>"
	    << std::endl;
	break;
      }
      case DDSolidShape::ddshapeless:
      {
	DDShapelessSolid rs(solid);
	xos << "<ShapelessSolid name=\""  << rs.toString() << "\"/>"
	    << std::endl;
	break;
      }
      case DDSolidShape::ddtorus:
      {
	// <Torus name="torus" innerRadius="7.5*cm" outerRadius="10*cm"
	//                     torusRadius="30*cm" startPhi="0*deg" deltaPhi="360*deg"/>
	DDTorus rs(solid);
	xos << "<Torus name=\""  << rs.toString() << "\""
	    << " innerRadius=\"" << rs.rMin() << "*mm\""
	    << " outerRadius=\"" << rs.rMax() << "*mm\""
	    << " torusRadius=\"" << rs.rTorus() << "*mm\""
	    << " startPhi=\"" << CONVERT_TO( rs.startPhi(), deg ) << "*deg\""
	    << " deltaPhi=\"" << CONVERT_TO( rs.deltaPhi(), deg ) << "*deg\"/>"
	    << std::endl;
	break;
      }
      case DDSolidShape::ddellipticaltube:
      {
	// <EllipticalTube name="CMSWall"  xSemiAxis="[cavernData:CMSWallEDX]"
	//                                 ySemiAxis="[cavernData:CMSWallEDY]"
	//                                 zHeight="[cms:HallZ]"/>
	DDEllipticalTube rs(solid);
	xos << "<EllipticalTube name=\"" << rs.toString()  << "\""
	    << " xSemiAxis=\"" << rs.xSemiAxis() << "*mm\""
	    << " ySemiAxis=\"" << rs.ySemiAxis() << "*mm\""
	    << " zHeight=\"" << rs.zHeight() << "*mm\"/>"
	    << std::endl;
	break;
      }
      case DDSolidShape::ddcuttubs: 
      {
	//      <Tubs name="TrackerSupportTubeNomex"         rMin="[SupportTubeR1]+[Tol]" 
	//            rMax="[SupportTubeR2]-[Tol]"           dz="[SupportTubeL]" 
	//            startPhi="0*deg"                       deltaPhi="360*deg"/>
	DDCutTubs rs(solid);
	const std::array<double, 3> &pLowNorm(rs.lowNorm());
	const std::array<double, 3> &pHighNorm(rs.highNorm());

	xos << "<CutTubs name=\""  << rs.toString() << "\""
	    << " dz=\"" << rs.zhalf() << "*mm\""
	    << " rMin=\"" << rs.rIn() << "*mm\""
	    << " rMax=\"" << rs.rOut() << "*mm\""
	    << " startPhi=\"" << CONVERT_TO( rs.startPhi(), deg ) << "*deg\""
	    << " deltaPhi=\"" << CONVERT_TO( rs.deltaPhi(), deg ) << "*deg\""
	    << " lx=\"" << pLowNorm[0] << "\""
	    << " ly=\"" << pLowNorm[1] << "\""
	    << " lz=\"" << pLowNorm[2] << "\""
	    << " tx=\"" << pHighNorm[0] << "\""
	    << " ty=\"" << pHighNorm[1] << "\""
	    << " tz=\"" << pHighNorm[2] << "\"/>"
	    << std::endl;
	break;
      }
      case DDSolidShape::ddextrudedpolygon:
      {
	 DDExtrudedPolygon rs(solid);
	 std::vector<double> x = rs.xVec();
	 std::vector<double> y = rs.yVec();
	 std::vector<double> z = rs.zVec();
	 std::vector<double> zx = rs.zxVec();
	 std::vector<double> zy = rs.zyVec();
	 std::vector<double> zs = rs.zscaleVec();
	 
         xos << "<ExtrudedPolygon name=\""  << rs.toString() << "\"";
	 for( unsigned int i : x )
	   xos << " <XYPoint x=\"" << x[i] << "*mm\" y=\"" << y[i] << "*mm\"/>\n";
	 for( unsigned int k : z )
	   xos << " <ZXYSection z=\"" << z[k] << "*mm\" x=\"" << zx[k] << "*mm\" y=\"" << zy[k] << "*mm scale=" <<  zs[k] << "*mm\"/>\n";
	 xos << "</ExtrudedPolygon>\n";
         break;
      }
         //       return new PSolid( pstrs(solid.toString()), solid.parameters()
         // 			 , solid.shape(), pstrs(""), pstrs(""), pstrs("") );
      case DDSolidShape::dd_not_init:
      default:
	throw cms::Exception("DDException") << "DDCoreToDDXMLOutput::solid(...) " << solid.name() << " either not inited or no such solid.";
         break;
   }
}

void DDCoreToDDXMLOutput::material( const DDMaterial& material, std::ostream& xos ) 
{
   int noc = material.noOfConstituents();
   if( noc == 0 ) 
   {
     xos << "<ElementaryMaterial name=\""  << material.toString() << "\""
	 << " density=\"" 
	 << std::scientific << std::setprecision(5)
	 << CONVERT_TO( material.density(), mg_per_cm3 ) << "*mg/cm3\""
	 << " atomicWeight=\"" 
	 << std::fixed  
	 << CONVERT_TO( material.a(), g_per_mole ) << "*g/mole\""
	 << std::setprecision(0) << std::fixed << " atomicNumber=\"" << material.z() << "\"/>"
	 << std::endl;
   } 
   else 
   {
     xos << "<CompositeMaterial name=\""  << material.toString() << "\""
	 << " density=\"" 
	 << std::scientific << std::setprecision(5)
	 << CONVERT_TO( material.density(), mg_per_cm3 ) << "*mg/cm3\""
	 << " method=\"mixture by weight\">" << std::endl;
      
     int j=0;
     for (; j<noc; ++j) 
     {
       xos << "<MaterialFraction fraction=\"" 
	   << std::fixed << std::setprecision(9)
	   << material.constituent(j).second << "\">" << std::endl;
       xos << "<rMaterial name=\""  << material.constituent(j).first.name() << "\"/>" << std::endl;
       xos << "</MaterialFraction>" << std::endl;
     }
     xos << "</CompositeMaterial>" << std::endl;
   }
}

void
DDCoreToDDXMLOutput::rotation( const DDRotation& rotation, std::ostream& xos,
			       const std::string& rotn ) 
{
   double tol = 1.0e-3; // Geant4 compatible
   DD3Vector x,y,z; 
   rotation.rotation()->GetComponents(x,y,z); 
   double check = (x.Cross(y)).Dot(z); // in case of a LEFT-handed orthogonal system 
                                       // this must be -1
   bool reflection((1.-check)>tol);
   std::string rotName=rotation.toString();
   if( rotName == ":" ) 
   {
      if( !rotn.empty() ) 
      {
         rotName = rotn;
         std::cout << "about to try to make a new DDRotation... should fail!" << std::endl;
         DDRotation rot( DDName(rotn), const_cast<DDRotationMatrix*>(rotation.rotation()));
         std:: cout << "new rotation: " << rot << std::endl;
      } 
      else 
      {
         std::cout << "WARNING: MAKING AN UNNAMED ROTATION" << std::endl;
      }
   }
   if( !reflection ) 
   {
      xos << "<Rotation ";
   } 
   else 
   {
      xos << "<ReflectionRotation ";
   }
   xos << "name=\"" << rotName << "\""
       << " phiX=\"" << CONVERT_TO( x.phi(), deg ) << "*deg\""
       << " thetaX=\"" << CONVERT_TO( x.theta(), deg ) << "*deg\""
       << " phiY=\"" << CONVERT_TO( y.phi(), deg ) << "*deg\""
       << " thetaY=\"" << CONVERT_TO( y.theta(), deg ) << "*deg\""
       << " phiZ=\"" << CONVERT_TO( z.phi(), deg ) << "*deg\""
       << " thetaZ=\"" << CONVERT_TO( z.theta(), deg ) << "*deg\"/>"
       << std::endl;
}

void DDCoreToDDXMLOutput::logicalPart( const DDLogicalPart& lp, std::ostream& xos ) 
{   
   xos << "<LogicalPart name=\""  << lp.toString() << "\">" << std::endl;
   xos << "<rSolid name=\""  << lp.solid().toString() << "\"/>" << std::endl;
   xos << "<rMaterial name=\""  << lp.material().toString() << "\"/>" << std::endl;
   xos << "</LogicalPart>" << std::endl;
}

void DDCoreToDDXMLOutput::position( const DDLogicalPart& parent,
				    const DDLogicalPart& child,
				    DDPosData* edgeToChild, 
				    int& rotNameSeed,
				    std::ostream& xos ) 
{
  std::string rotName = edgeToChild->ddrot().toString();
  DDRotationMatrix myIDENT;
   
  xos << "<PosPart copyNumber=\"" << edgeToChild->copyno() << "\">" << std::endl;
  xos << "<rParent name=\"" << parent.toString() << "\"/>" << std::endl;
  xos << "<rChild name=\"" << child.toString() << "\"/>" << std::endl;
  if( *(edgeToChild->ddrot().rotation()) != myIDENT ) 
  {
    if( rotName == ":" ) 
    {
      rotation(edgeToChild->ddrot(), xos);
    }
    else
    {
      xos << "<rRotation name=\"" << rotName << "\"/>" << std::endl;
    }
  } // else let default Rotation matrix be created?
  xos << "<Translation x=\"" << edgeToChild->translation().x() <<"*mm\""
      << " y=\"" << edgeToChild->translation().y() <<"*mm\""
      << " z=\"" << edgeToChild->translation().z() <<"*mm\"/>" << std::endl;
  xos << "</PosPart>" << std::endl;
}

void 
DDCoreToDDXMLOutput::specpar( const DDSpecifics& sp, std::ostream& xos ) 
{   
   xos << "<SpecPar name=\"" << sp.toString() << "\" eval=\"false\">" << std::endl;
   
   // ========...  all the selection strings out as strings by using the DDPartSelection's std::ostream function...
   for( const auto& psit : sp.selection()) 
   {
      xos << "<PartSelector path=\"" << psit << "\"/>" << std::endl;
   }
   
   // =========  ... and iterate over all DDValues...
   for( const auto& vit : sp.specifics()) 
   {
      const DDValue & v = vit.second;
      size_t s=v.size();
      size_t i=0;
      // ============  ... all actual values with the same name
      const std::vector<std::string>& strvec = v.strings();
      if( v.isEvaluated()) 
      {
         for(; i<s; ++i) 
         {
            xos << "<Parameter name=\"" << v.name() << "\""
            << " value=\"" << v[i] << "\""
            << " eval=\"true\"/>" << std::endl;
         }
      } 
      else 
      {
         for(; i<s; ++i ) 
         {
            xos << "<Parameter name=\"" << v.name() << "\""
            << " value=\"" << strvec[i] << "\""
            << " eval=\"false\"/>" << std::endl;
         }
      }
      
   }
   xos << "</SpecPar>" << std::endl;
}

void DDCoreToDDXMLOutput::specpar( const std::pair<DDsvalues_type, std::set<const DDPartSelection*> >& pssv, std::ostream& xos ) 
{
   static const std::string madeName("specparname");
   static int numspecpars(0);
   std::ostringstream ostr;
   ostr << numspecpars++;
   std::string spname = madeName + ostr.str(); 
   xos << "<SpecPar name=\"" << spname << "\" eval=\"false\">" << std::endl;
   for( const auto& psit : pssv.second ) {
      xos << "<PartSelector path=\"" << *psit << "\"/>" << std::endl;
   }
   
   // =========  ... and iterate over all DDValues...
   for( const auto& vit : pssv.first ) 
   {
      const DDValue & v = vit.second;
      size_t s=v.size();
      size_t i=0;
      // ============  ... all actual values with the same name
      const std::vector<std::string>& strvec = v.strings();
      if( v.isEvaluated() ) 
      {
         for(; i<s; ++i) 
         {
            xos << "<Parameter name=\"" << v.name() << "\""
            << " value=\"" << v[i] << "\""
            << " eval=\"true\"/>" << std::endl;
         }
      } 
      else 
      {
         for(; i<s; ++i ) 
         {
            xos << "<Parameter name=\"" << v.name() << "\""
            << " value=\"" << strvec[i] << "\""
            << " eval=\"false\"/>" << std::endl;
         }
      }
   }
   
   xos << "</SpecPar>" << std::endl;
}
