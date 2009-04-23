///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerPhiAlgo.cc
// Description: Position n copies at prescribed phi values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerPhiAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTrackerPhiAlgo::DDTrackerPhiAlgo() : startcn(1), incrcn(1) {
  LogDebug("TrackerGeom") << "DDTrackerPhiAlgo info: Creating an instance";
}

DDTrackerPhiAlgo::~DDTrackerPhiAlgo() {}

void DDTrackerPhiAlgo::initialize(const DDNumericArguments & nArgs,
				  const DDVectorArguments & vArgs,
				  const DDMapArguments & ,
				  const DDStringArguments & sArgs,
				  const DDStringVectorArguments & )  {

  if ( nArgs.find("StartCopyNo") != nArgs.end() ) {
    startcn = size_t(nArgs["StartCopyNo"]);
  }
  if ( nArgs.find("IncrCopyNo") != nArgs.end() ) {
    incrcn = int(nArgs["IncrCopyNo"]);
  }

  radius     = nArgs["Radius"];
  tilt       = nArgs["Tilt"];
  phi        = vArgs["Phi"];
  zpos       = vArgs["ZPos"];

  if ( nArgs.find("NumCopies") != nArgs.end() ) {
    numcopies = size_t(nArgs["NumCopies"]);
    if ( numcopies != phi.size() ) {
      edm::LogError("TrackerGeom") << "DDTrackerPhiAlgo error: Parameter NumCopies "
				   << "does not agree with the size of the Phi "
				   << "vector.  It was adjusted to be the size "
				   << "of the Phi vector and may lead to crashes "
				   << "or errors.";
    } 
  }
  numcopies = phi.size() - 1; // -1 for loop in execute.  seems almost redundant...
    

  LogDebug("TrackerGeom") << "DDTrackerPhiAlgo debug: Parameters for position"
			  << "ing:: " << " Radius " << radius << " Tilt " 
			  << tilt/deg << " Copies " << phi.size() << " at";
  for (int i=0; i<(int)(phi.size()); i++)
    LogDebug("TrackerGeom") << "\t[" << i << "] phi = " << phi[i]/deg 
			    << " z = " << zpos[i];

  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  LogDebug("TrackerGeom") <<  "DDTrackerPhiAlgo debug: Parent " << parentName
			  <<"\tChild " << childName << " NameSpace " 
			  << idNameSpace;
}

void DDTrackerPhiAlgo::execute() {

  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  double theta  = 90.*deg;
  int i = 0;
  for (size_t ci=startcn; ci<numcopies+1; ci = ci+incrcn) {
    double phix = phi[i] + tilt;
    double phiy = phix + 90.*deg;
    double phideg = phi[i]/deg;

    std::string rotstr = DDSplit(childName).first + dbl_to_string(phideg);
    DDRotation rotation = DDRotation(DDName(rotstr, idNameSpace));
    if (!rotation) {
      LogDebug("TrackerGeom") << "DDTrackerPhiAlgo test: Creating a new "
			      << "rotation: " << rotstr << "\t" << "90., "
			      << phix/deg << ", 90.," << phiy/deg << ", 0, 0";
      rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
		       0., 0.);
    }
	
    double xpos = radius*cos(phi[i]);
    double ypos = radius*sin(phi[i]);
    DDTranslation tran(xpos, ypos, zpos[i]);
  
    DDpos (child, mother, ci, tran, rotation);
    LogDebug("TrackerGeom") << "DDTrackerPhiAlgo test: " << child << " number "
			    << i+1 << " positioned in " << mother << " at "
			    << tran  << " with " << rotation;
    ++i;
  }
}
