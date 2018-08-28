///////////////////////////////////////////////////////////////////////////////
// File: DDTECCoolAlgo.cc
// Description: Placing cooling pieces in the petal material of a TEC petal
// * in each call all objects are placed at the same radial position.
// * Inserts are placed into the parent object
// * for all i: CoolInsert[i] goes to PhiPosition[i]
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTECCoolAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTECCoolAlgo::DDTECCoolAlgo(): phiPosition(0),coolInsert(0) {
  LogDebug("TECGeom") << "DDTECCoolAlgo info: Creating an instance";
}

DDTECCoolAlgo::~DDTECCoolAlgo() {}

void DDTECCoolAlgo::initialize(const DDNumericArguments & nArgs,
			       const DDVectorArguments & vArgs,
			       const DDMapArguments & ,
			       const DDStringArguments & sArgs,
			       const DDStringVectorArguments & vsArgs) {

  startCopyNo    = int(nArgs["StartCopyNo"]);

  DDName parentName = parent().name(); 
  rPosition       = nArgs["RPosition"];
  LogDebug("TECGeom") << "DDTECCoolAlgo debug: Parent " << parentName 
		      <<" NameSpace " << DDCurrentNamespace() << " at radial Position " 
		      << rPosition ;
  phiPosition    = vArgs["PhiPosition"]; 
  coolInsert     = vsArgs["CoolInsert"];
  if (phiPosition.size() == coolInsert.size()) {
    for (int i=0; i<(int)(phiPosition.size()); i++) 
      LogDebug("TECGeom") << "DDTECCoolAlgo debug: Insert[" << i << "]: "
			  << coolInsert.at(i) << " at Phi " 
			  << phiPosition.at(i)/CLHEP::deg;
  } else {
    LogDebug("TECGeom") << "ERROR: Number of inserts does not match the numer of PhiPositions!";
  }
  LogDebug("TECGeom") << " Done creating instance of DDTECCoolAlgo ";
}

void DDTECCoolAlgo::execute(DDCompactView& cpv) {
  LogDebug("TECGeom") << "==>> Constructing DDTECCoolAlgo...";
  int copyNo  = startCopyNo;
  // loop over the inserts to be placed
  for (int i = 0; i < (int)(coolInsert.size()); i++) {
    // get objects
    DDName child  = DDName(DDSplit(coolInsert.at(i)).first, 
			   DDSplit(coolInsert.at(i)).second);
    DDName mother = parent().name();
    // get positions
    double xpos = rPosition*cos(phiPosition.at(i));
    double ypos = -rPosition*sin(phiPosition.at(i));
    // place inserts
    DDTranslation tran(xpos, ypos, 0.0);
    DDRotation rotation;
   cpv.position(child, mother, copyNo, tran, rotation);
    LogDebug("TECGeom") << "DDTECCoolAlgo test " << child << "["  
			<< copyNo << "] positioned in " << mother 
			<< " at " << tran  << " with " << rotation 
			<< " phi " << phiPosition.at(i)/CLHEP::deg << " r " 
			<< rPosition;
    copyNo++;
  }
  LogDebug("TECGeom") << "<<== End of DDTECCoolAlgo construction ...";
}
