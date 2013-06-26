// #include "Utilities/Configuration/interface/Architecture.h"

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2013/04/15 16:02:44 $
 *  $Revision: 1.15 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/src/volumeHandle.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometryVector/interface/CoordinateSets.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "MagneticField/Layers/interface/MagVerbosity.h"

#include <string>
#include <iterator>
#include <iomanip>
#include <iostream>
#include <boost/lexical_cast.hpp>

using namespace SurfaceOrientation;
using namespace std;


MagGeoBuilderFromDDD::volumeHandle::~volumeHandle(){
  delete refPlane;
}

MagGeoBuilderFromDDD::volumeHandle::volumeHandle(const DDExpandedView &fv, bool expand2Pi)
  : name(fv.logicalPart().name().name()),
    copyno(fv.copyno()),
    magVolume(0),
    masterSector(1),
    theRN(0.),
    theRMin(0.),
    theRMax(0.),
    refPlane(0),
    solid(fv.logicalPart().solid()),
    center_(GlobalPoint(fv.translation().x()/cm,
			fv.translation().y()/cm,
			fv.translation().z()/cm)),
    expand(expand2Pi),
    isIronFlag(false)
{
  // ASSUMPTION: volume names ends with "_NUM" where NUM is the volume number
  string volName = name;
  volName.erase(0,volName.rfind('_')+1);    
  volumeno =boost::lexical_cast<unsigned short>(volName);

  for (int i=0; i<6; ++i) {
    isAssigned[i] = false;
  }

  
  if (MagGeoBuilderFromDDD::debug) {  
    cout.precision(7);
  }
  

  referencePlane(fv);

  if (solid.shape() == ddbox) {
    buildBox(fv);
  } else if (solid.shape() == ddtrap) {
    buildTrap(fv);
  } else if (solid.shape() == ddcons) {
    buildCons(fv);
  } else if (solid.shape() == ddtubs) {   
    buildTubs(fv);
  } else if (solid.shape() == ddpseudotrap) {   
    buildPseudoTrap(fv);
  } else if (solid.shape() == ddtrunctubs) {   
    buildTruncTubs(fv);
  } else {
    cout << "volumeHandle ctor: Unexpected solid: " << (int) solid.shape() << endl;
  }


  // NOTE: Table name and master sector are no longer taken from xml!
//   DDsvalues_type sv(fv.mergedSpecifics());
    
//   { // Extract the name of associated field file.
//     std::vector<std::string> temp;
//     std::string pname = "table";
//     DDValue val(pname);
//     DDsvalues_type sv(fv.mergedSpecifics());
//     if (DDfetch(&sv,val)) {
//       temp = val.strings();
//       if (temp.size() != 1) {
// 	cout << "*** WARNING: volume has > 1 SpecPar " << pname << endl;
//       }
//       magFile = temp[0];

//       string find="[copyNo]";
//       std::size_t j;
//       for ( ; (j = magFile.find(find)) != string::npos ; ) {
// 	stringstream conv;
// 	conv << setfill('0') << setw(2) << copyno;
// 	string repl;
// 	conv >> repl;
// 	magFile.replace(j, find.length(), repl);
//       }
      
//     } else {
//       cout << "*** WARNING: volume does not have a SpecPar " << pname << endl;
//       cout << " DDsvalues_type:  " << fv.mergedSpecifics() << endl;
//     }
//   }

//   { // Extract the number of the master sector.
//     std::vector<double> temp;
//     const std::string pname = "masterSector";
//     DDValue val(pname);
//     if (DDfetch(&sv,val)) {
//       temp = val.doubles();
//       if (temp.size() != 1) {
//  	cout << "*** WARNING: volume has > 1 SpecPar " << pname << endl;
//       }
//       masterSector = int(temp[0]+.5);
//     } else {
//       if (MagGeoBuilderFromDDD::debug) { 
// 	cout << "Volume does not have a SpecPar " << pname 
// 	     << " using: " << copyno << endl;
// 	cout << " DDsvalues_type:  " << fv.mergedSpecifics() << endl;
//       }
//       masterSector = copyno;
//     }  
//   }
  
  // Get material for this volume
  if (fv.logicalPart().material().name().name() == "Iron") isIronFlag=true;  


  if (MagGeoBuilderFromDDD::debug) {  
    cout << " RMin =  " << theRMin <<endl;
    cout << " RMax =  " << theRMax <<endl;
      
    if (theRMin < 0 || theRN < theRMin || theRMax < theRN) 
      cout << "*** WARNING: wrong RMin/RN/RMax , shape: " << (int) shape() << endl;

    cout << "Summary: " << name << " " << copyno
	 << " Shape= " << (int) shape()
	 << " trasl " << center()
	 << " R " << center().perp()
	 << " phi " << center().phi()
	 << " magFile " << magFile
	 << " Material= " << fv.logicalPart().material().name()
	 << " isIron= " << isIronFlag
	 << " masterSector= " << masterSector << std::endl;

    cout << " Orientation of surfaces:";
    std::string sideName[3] =  {"positiveSide", "negativeSide", "onSurface"};
    for (int i=0; i<6; ++i) {    
      cout << "  " << i << ":" << sideName[surfaces[i]->side(center_,0.3)];
    }
    cout << endl;
  }
}


const Surface::GlobalPoint & MagGeoBuilderFromDDD::volumeHandle::center() const {
  return center_;
}

void MagGeoBuilderFromDDD::volumeHandle::referencePlane(const DDExpandedView &fv){
  // The refPlane is the "main plane" for the solid. It corresponds to the 
  // x,y plane in the DDD local frame, and defines a frame where the local
  // coordinates are the same as in DDD. 
  // In the geometry version 85l_030919, this plane is normal to the 
  // beam line for all volumes but pseudotraps, so that global R is along Y,
  // global phi is along -X and global Z along Z:
  //
  //   Global(for vol at pi/2)    Local 
  //   +R (+Y)                    +Y
  //   +phi(-X)                   -X
  //   +Z                         +Z
  //
  // For pseudotraps the refPlane is parallel to beam line and global R is
  // along Z, global phi is along +-X and and global Z along Y:
  // 
  //   Global(for vol at pi/2)    Local 
  //   +R (+Y)                    +Z
  //   +phi(-X)                   +X
  //   +Z                         +Y
  //
  // Note that the frame is centered in the DDD volume center, which is
  // inside the volume for DDD boxes and (pesudo)trapezoids, on the beam line
  // for tubs, cons and trunctubs. 

  // In geometry version 1103l, trapezoids have X and Z in the opposite direction
  // than the above.  Boxes are either oriented as described above or in some case 
  // have opposite direction for Y and X.

  // The global position
  Surface::PositionType & posResult = center_;

  // The reference plane rotation
  DD3Vector x, y, z;
  fv.rotation().GetComponents(x,y,z);
  if (MagGeoBuilderFromDDD::debug) {
    if (x.Cross(y).Dot(z) < 0.5) {
      cout << "*** WARNING: Rotation is not RH "<< endl;
    }
  }
  
  // The global rotation
  Surface::RotationType
    rotResult(float(x.X()),float(x.Y()),float(x.Z()),
	      float(y.X()),float(y.Y()),float(y.Z()),
	      float(z.X()),float(z.Y()),float(z.Z()));

  refPlane = new GloballyPositioned<float>(posResult, rotResult);

  // Check correct orientation
  if (MagGeoBuilderFromDDD::debug) {

    cout << "Refplane pos  " << refPlane->position() << endl;

    // See comments above for the conventions for orientation.
    LocalVector globalZdir(0.,0.,1.); // Local direction of the axis along global Z 
    if (solid.shape() == ddpseudotrap) {
      globalZdir = LocalVector(0.,1.,0.);    
    }
    if (refPlane->toGlobal(globalZdir).z()<0.) {
      globalZdir=-globalZdir;
    }

    float chk = refPlane->toGlobal(globalZdir).dot(GlobalVector(0,0,1));
    if (chk < .999) cout << "*** WARNING RefPlane check failed!***"
			 << chk << endl; 
  }
}



void MagGeoBuilderFromDDD::volumeHandle::buildPhiZSurf(double startPhi,
						       double deltaPhi,
						       double zhalf,
						       double rCentr) {
  // This is 100% equal for cons and tubs!!!

  GlobalVector planeXAxis = refPlane->toGlobal(LocalVector( 1, 0, 0));
  GlobalVector planeYAxis = refPlane->toGlobal(LocalVector( 0, 1, 0));
  GlobalVector planeZAxis = refPlane->toGlobal(LocalVector( 0, 0, 1));

  // Local Y axis of the faces at +-phi.
  GlobalVector y_phiplus = refPlane->toGlobal(LocalVector(cos(startPhi+deltaPhi),
							  sin(startPhi+deltaPhi),0.));
  GlobalVector y_phiminus = refPlane->toGlobal(LocalVector(cos(startPhi),
							   sin(startPhi),0.));

  Surface::RotationType rot_Z(planeXAxis,planeYAxis);
  Surface::RotationType rot_phiplus(planeZAxis, y_phiplus); 
  Surface::RotationType rot_phiminus(planeZAxis, y_phiminus);

  GlobalPoint pos_zplus(center_.x(),center_.y(),center_.z()+zhalf);
  GlobalPoint pos_zminus(center_.x(),center_.y(),center_.z()-zhalf);
  // BEWARE: in this case, the origin for phiplus,phiminus surfaces is 
  // at radius R and not on a plane passing by center_ orthogonal to the radius.
  GlobalPoint pos_phiplus(refPlane->toGlobal(LocalPoint(rCentr*cos(startPhi+deltaPhi),rCentr*sin(startPhi+deltaPhi),0.)));
  GlobalPoint pos_phiminus(refPlane->toGlobal(LocalPoint(rCentr*cos(startPhi),
 							 rCentr*sin(startPhi),
 							 0.)));
  surfaces[zplus]    = new Plane(pos_zplus, rot_Z);
  surfaces[zminus]   = new Plane(pos_zminus, rot_Z);
  surfaces[phiplus]  = new Plane(pos_phiplus, rot_phiplus);
  surfaces[phiminus] = new Plane(pos_phiminus, rot_phiminus);  
  
  if (MagGeoBuilderFromDDD::debug) {
    cout << "Actual Center at: " << center_ << " R " << center_.perp()
	 << " phi " << center_.phi() << endl;
    cout << "RN            " << theRN << endl;

    cout << "pos_zplus    " << pos_zplus << " "
	 << pos_zplus.perp() << " " << pos_zplus.phi() << endl
	 << "pos_zminus   " << pos_zminus << " "
	 << pos_zminus.perp() << " " << pos_zminus.phi() << endl
	 << "pos_phiplus  " << pos_phiplus << " "
	 << pos_phiplus.perp() << " " << pos_phiplus.phi() <<endl
	 << "pos_phiminus " << pos_phiminus << " "
	 << pos_phiminus.perp() << " " << pos_phiminus.phi() <<endl;

    cout << "y_phiplus " << y_phiplus << endl;
    cout << "y_phiminus " << y_phiminus << endl;

    cout << "rot_Z " << surfaces[zplus]->toGlobal(LocalVector(0.,0.,1.)) << endl
	 << "rot_phi+ " << surfaces[phiplus]->toGlobal(LocalVector(0.,0.,1.))
	 << " phi " << surfaces[phiplus]->toGlobal(LocalVector(0.,0.,1.)).phi()
	 << endl
	 << "rot_phi- " << surfaces[phiminus]->toGlobal(LocalVector(0.,0.,1.))
	 << " phi " << surfaces[phiminus]->toGlobal(LocalVector(0.,0.,1.)).phi()
	 << endl;
  }
  
//   // Check ordering.
  if (MagGeoBuilderFromDDD::debug) {
    if (pos_zplus.z() < pos_zminus.z()) {
      cout << "*** WARNING: pos_zplus < pos_zminus " << endl;
    }
    if (Geom::Phi<float>(pos_phiplus.phi()-pos_phiminus.phi()) < 0. ) {
      cout << "*** WARNING: pos_phiplus < pos_phiminus " << endl;
    }
  }
}



bool MagGeoBuilderFromDDD::volumeHandle::sameSurface(const Surface & s1, Sides which_side, float tolerance)
{
  //Check for null comparison
  if (&s1==(surfaces[which_side]).get()){
    if (MagGeoBuilderFromDDD::debug) cout << "      sameSurface: OK (same ptr)" << endl;
    return true;
  }

  const float maxtilt  = 0.999;

  const Surface & s2 = *(surfaces[which_side]);
  // Try with a plane.
  const Plane * p1 = dynamic_cast<const Plane*>(&s1);
  if (p1!=0) {
    const Plane * p2 = dynamic_cast<const Plane*>(&s2);
    if (p2==0) {
      if (MagGeoBuilderFromDDD::debug) cout << "      sameSurface: different types" << endl;
      return false;
    }
    
    if ( (fabs(p1->normalVector().dot(p2->normalVector())) > maxtilt)
	 && (fabs((p1->toLocal(p2->position())).z()) < tolerance) ) {
      if (MagGeoBuilderFromDDD::debug) cout << "      sameSurface: OK "
		      << fabs(p1->normalVector().dot(p2->normalVector()))
		      << " " << fabs((p1->toLocal(p2->position())).z()) << endl;
      return true;
    } else{
      if (MagGeoBuilderFromDDD::debug) cout << "      sameSurface: not the same: "
		      << p1->normalVector() << p1->position() << endl
		      << "                                 "
		      << p2->normalVector() << p2->position() << endl
		      << fabs(p1->normalVector().dot(p2->normalVector()))
		      << " " << (p1->toLocal(p2->position())).z()<< endl;
      return false;
    }
  }

  // Try with a cylinder.  
  const Cylinder * cy1 = dynamic_cast<const Cylinder*>(&s1);
  if (cy1!=0) {
    const Cylinder * cy2 = dynamic_cast<const Cylinder*>(&s2);
    if (cy2==0) {
      if (MagGeoBuilderFromDDD::debug) cout << "      sameSurface: different types" << endl;
      return false;
    }
    // Assume axis is the same!
    if (fabs(cy1->radius() - cy2->radius()) < tolerance) {
      return true;
    } else {
      return false;
    }
  }

  // Try with a cone.  
  const Cone * co1 = dynamic_cast<const Cone*>(&s1);
  if (co1!=0) {
    const Cone * co2 = dynamic_cast<const Cone*>(&s2);
    if (co2==0) {
      if (MagGeoBuilderFromDDD::debug) cout << "      sameSurface: different types" << endl;
      return false;
    }
    // FIXME
    if (fabs(co1->openingAngle()-co2->openingAngle()) < maxtilt 
	&& (co1->vertex()-co2->vertex()).mag() < tolerance) {
      return true;
    } else {
      return false;
    }
  }

  if (MagGeoBuilderFromDDD::debug) cout << "      sameSurface: unknown surfaces..." << endl;
  return false;
}



bool MagGeoBuilderFromDDD::volumeHandle::setSurface(const Surface & s1, Sides which_side) 
{
 //Check for null assignment
  if (&s1==(surfaces[which_side]).get()){
    isAssigned[which_side] = true;
    return true;
  }

  if (!sameSurface(s1,which_side)){
    cout << "***ERROR: setSurface: trying to assign a surface that does not match destination surface. Skipping." << endl;    
    const Surface & s2 = *(surfaces[which_side]);
    //FIXME: Just planes for the time being!!!
    const Plane * p1 = dynamic_cast<const Plane*>(&s1);
    const Plane * p2 = dynamic_cast<const Plane*>(&s2);
    if (p1!=0 && p2 !=0) 
      cout << p1->normalVector() << p1->position() << endl
	   << p2->normalVector() << p2->position() << endl;
    return false;
  }
  

  if (isAssigned[which_side]) {
    if (&s1!=(surfaces[which_side]).get()){
      cout << "*** WARNING volumeHandle::setSurface: trying to reassign a surface to a different surface instance" << endl;
      return false;
    }
  } else {
    surfaces[which_side] = &s1;
    isAssigned[which_side] = true;
    if (MagGeoBuilderFromDDD::debug) cout << "     Volume " << name << " # " << copyno << " Assigned: " << (int) which_side << endl;
    return true;
  }

  return false; // let the compiler be happy
}



const Surface & 
MagGeoBuilderFromDDD::volumeHandle::surface(Sides which_side) const {
  return *(surfaces[which_side]);
}



const Surface & 
MagGeoBuilderFromDDD::volumeHandle::surface(int which_side) const {
  assert(which_side >=0 && which_side <6);
  return *(surfaces[which_side]);
}


std::vector<VolumeSide>
MagGeoBuilderFromDDD::volumeHandle::sides() const{
  std::vector<VolumeSide> result;
  for (int i=0; i<6; ++i){
    // If this is just a master volume out of wich a 2pi volume
    // should be built (e.g. central cylinder), skip the phi boundaries.
    if (expand && (i==phiplus || i==phiminus)) continue;

    // FIXME: Skip null inner degenerate cylindrical surface
    if (solid.shape() == ddtubs && i == SurfaceOrientation::inner && theRMin < 0.001) continue;

    ReferenceCountingPointer<Surface> s = const_cast<Surface*> (surfaces[i].get());
    result.push_back(VolumeSide(s, GlobalFace(i),
				surfaces[i]->side(center_,0.3)));
  }
  return result;
}

void MagGeoBuilderFromDDD::volumeHandle::printUniqueNames(handles::const_iterator begin, handles::const_iterator end ) {
    std::vector<std::string> names;
    for (handles::const_iterator i = begin; 
	 i != end; ++i){
      names.push_back((*i)->name);
    }
     
    sort(names.begin(),names.end());
    std::vector<std::string>::iterator i = unique(names.begin(),names.end());
    int nvols = int(i - names.begin());
    cout << nvols << " ";
    copy(names.begin(), i, ostream_iterator<std::string>(cout, " "));
     
    cout << endl;
}


#include "MagneticField/GeomBuilder/src/buildBox.icc"
#include "MagneticField/GeomBuilder/src/buildTrap.icc"
#include "MagneticField/GeomBuilder/src/buildTubs.icc"
#include "MagneticField/GeomBuilder/src/buildCons.icc"
#include "MagneticField/GeomBuilder/src/buildPseudoTrap.icc"
#include "MagneticField/GeomBuilder/src/buildTruncTubs.icc"
