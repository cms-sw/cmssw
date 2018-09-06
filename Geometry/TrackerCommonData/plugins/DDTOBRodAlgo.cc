///////////////////////////////////////////////////////////////////////////////
// File: DDTOBRodAlgo.cc
// Description: Positioning constituents of a TOB rod
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTOBRodAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTOBRodAlgo::DDTOBRodAlgo():
  sideRod(0), sideRodX(0), sideRodY(0), sideRodZ(0), endRod1Y(0), endRod1Z(0),
  clampX(0), clampZ(0), sideCoolX(0), sideCoolY(0), sideCoolZ(0),
  endCoolY(0), endCoolZ(0),
  optFibreX(0),optFibreZ(0),
  sideClampX(0), sideClamp1DZ(0), sideClamp2DZ(0), moduleRot(0), moduleY(0),
  moduleZ(0), connect(0), connectY(0), connectZ(0),
  aohCopies(0), aohX(0), aohY(0), aohZ(0) {
  LogDebug("TOBGeom") << "DDTOBRodAlgo info: Creating an instance";
}

DDTOBRodAlgo::~DDTOBRodAlgo() {}

void DDTOBRodAlgo::initialize(const DDNumericArguments & nArgs,
			      const DDVectorArguments & vArgs,
			      const DDMapArguments & ,
			      const DDStringArguments & sArgs,
			      const DDStringVectorArguments & vsArgs) {

  central      = sArgs["CentralName"];
  shift        = nArgs["Shift"];
  idNameSpace  = DDCurrentNamespace::ns();
  DDName parentName = parent().name();
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug: Parent " << parentName 
		      << " Central " << central << " NameSpace "
		      << idNameSpace << "\tShift " << shift;

  sideRod      = vsArgs["SideRodName"];     
  sideRodX     = vArgs["SideRodX"];    
  sideRodY     = vArgs["SideRodY"];
  sideRodZ     = vArgs["SideRodZ"];
  for (int i=0; i<(int)(sideRod.size()); i++) {
    LogDebug("TOBGeom") << "DDTOBRodAlgo debug: " << sideRod[i] 
			<< " to be positioned " << sideRodX.size() 
			<<" times at y = " << sideRodY[i] << " z = " 
			<< sideRodZ[i] << " and x";
    for (double j : sideRodX)
      LogDebug("TOBGeom") << "\tsideRodX[" << i << "] = " << j;
  }

  endRod1      = sArgs["EndRod1Name"];     
  endRod1Y     = vArgs["EndRod1Y"];    
  endRod1Z     = vArgs["EndRod1Z"];    
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug: " << endRod1 << " to be "
		      << "positioned " << endRod1Y.size() << " times at";
  for (int i=0; i<(int)(endRod1Y.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\ty = " << endRod1Y[i] 
			<< "\tz = " << endRod1Z[i];

  endRod2      = sArgs["EndRod2Name"];     
  endRod2Y     = nArgs["EndRod2Y"];    
  endRod2Z     = nArgs["EndRod2Z"];    
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug: " << endRod2 << " to be "
		      << "positioned at y = " << endRod2Y << " z = " 
		      << endRod2Z;

  cable        = sArgs["CableName"];       
  cableZ       = nArgs["CableZ"];      
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug: " << cable << " to be "
		      << "positioned at z = " << cableZ;

  clamp        = sArgs["ClampName"];       
  clampX       = vArgs["ClampX"];      
  clampZ       = vArgs["ClampZ"];      
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug: " << clamp << " to be "
		      << "positioned " << clampX.size() << " times at";
  for (int i=0; i<(int)(clampX.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tx = " << clampX[i] << "\tz = "
			<< clampZ[i];

  sideCool     = sArgs["SideCoolName"];    
  sideCoolX    = vArgs["SideCoolX"];   
  sideCoolY    = vArgs["SideCoolY"];   
  sideCoolZ    = vArgs["SideCoolZ"];   
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug: " << sideCool << " to be "
		      << "positioned " << sideCoolX.size() << " times at";
  for (int i=0; i<(int)(sideCoolX.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tx = " << sideCoolX[i]
			<< "\ty = " << sideCoolY[i]
			<< "\tz = " << sideCoolZ[i];

  endCool      = sArgs["EndCoolName"];     
  endCoolY     = nArgs["EndCoolY"];    
  endCoolZ     = nArgs["EndCoolZ"];    
  endCoolRot   = sArgs["EndCoolRot"];   
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug: " <<endCool <<" to be "
		      << "positioned with " << endCoolRot << " rotation at"
		      << " y = " << endCoolY
		      << " z = " << endCoolZ;

  optFibre     = sArgs["OptFibreName"];    
  optFibreX    = vArgs["optFibreX"];   
  optFibreZ    = vArgs["optFibreZ"];   
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug: " << optFibre << " to be "
			  << "positioned " << optFibreX.size() << " times at";
  for (int i=0; i<(int)(optFibreX.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tx = " << optFibreX[i] 
			<< "\tz = " << optFibreZ[i];

  sideClamp1   = sArgs["SideClamp1Name"];  
  sideClampX   = vArgs["SideClampX"];  
  sideClamp1DZ = vArgs["SideClamp1DZ"];
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug: " << sideClamp1 << " to be "
		      << "positioned " << sideClampX.size() << " times at";
  for (int i=0; i<(int)(sideClampX.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tx = " << sideClampX[i] 
			<< "\tdz = " << sideClamp1DZ[i];

  sideClamp2   = sArgs["SideClamp2Name"];  
  sideClamp2DZ = vArgs["SideClamp2DZ"];
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug: " << sideClamp2 << " to be "
		      << "positioned " << sideClampX.size() << " times at";
  for (int i=0; i<(int)(sideClampX.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tx = " << sideClampX[i]
			<< "\tdz = " << sideClamp2DZ[i];

  moduleRot    = vsArgs["ModuleRot"];   
  module       = sArgs["ModuleName"]; 
  moduleY      = vArgs["ModuleY"];     
  moduleZ      = vArgs["ModuleZ"];
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug:\t" << module <<" positioned "
		      << moduleRot.size() << " times";
  for (int i=0; i<(int)(moduleRot.size()); i++) 
    LogDebug("TOBGeom") << "\tRotation " << moduleRot[i] << "\ty = " 
			<< moduleY[i] << "\tz = " << moduleZ[i];

  connect      = vsArgs["ICCName"];
  connectY     = vArgs["ICCY"];    
  connectZ     = vArgs["ICCZ"];    
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug:\t" << connect.size() 
		      << " ICC positioned with no rotation";
  for (int i=0; i<(int)(connect.size()); i++) 
    LogDebug("TOBGeom") << "\t" << connect[i] << "\ty = " << connectY[i] 
			<< "\tz = " << connectZ[i];

  aohName   = sArgs["AOHName"]; 
  aohCopies = vArgs["AOHCopies"];
  aohX      = vArgs["AOHx"];     
  aohY      = vArgs["AOHy"];     
  aohZ      = vArgs["AOHz"];     
  LogDebug("TOBGeom") << "DDTOBRodAlgo debug:\t" << aohName <<" AOH will be positioned on ICC's";
  for (int i=0; i<(int)(aohCopies.size()); i++) 
    LogDebug("TOBGeom")  << " copies " << aohCopies[i]
			 << "\tx = " << aohX[i]
			 << "\ty = " << aohY[i] 
			 << "\tz = " << aohZ[i];
  
}

void DDTOBRodAlgo::execute(DDCompactView& cpv) {
  
  LogDebug("TOBGeom") << "==>> Constructing DDTOBRodAlgo...";
  DDName rodName = parent().name();
  DDName centName(DDSplit(central).first, DDSplit(central).second);

  // Side Rods
  for (int i=0; i<(int)(sideRod.size()); i++) {
    for (int j=0; j<(int)(sideRodX.size()); j++) {
      DDTranslation r(sideRodX[j], sideRodY[i], sideRodZ[i]);
      DDName child(DDSplit(sideRod[i]).first, DDSplit(sideRod[i]).second);
      cpv.position(child, rodName, j+1, r, DDRotation());
      LogDebug("TOBGeom") << "DDTOBRodAlgo test: "  << child << " number " 
			  << j+1 << " positioned in " << rodName << " at "
			  << r << " with no rotation";
    }
  }

  // Clamps
  for (int i=0; i<(int)(clampX.size()); i++) {
    DDTranslation r(clampX[i], 0, shift+clampZ[i]);
    DDName child(DDSplit(clamp).first, DDSplit(clamp).second);
    cpv.position(child, rodName, i+1, r, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child << " number " 
			<< i+1 << " positioned in " << rodName << " at "
			<< r << " with no rotation";
  }

  // Side Cooling tubes
  for (int i=0; i<(int)(sideCoolX.size()); i++) {
    DDTranslation r(sideCoolX[i], sideCoolY[i], shift+sideCoolZ[i]);
    DDName child(DDSplit(sideCool).first, DDSplit(sideCool).second);
    cpv.position(child, rodName, i+1, r, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child << " number " 
			<< i+1 << " positioned in " << rodName << " at "
			<< r << " with no rotation";
  }

  // Optical Fibres
  for (int i=0; i<(int)(optFibreX.size()); i++) {
    DDTranslation r(optFibreX[i], 0, shift+optFibreZ[i]);
    DDName child(DDSplit(optFibre).first, DDSplit(optFibre).second);
    cpv.position(child, rodName, i+1, r, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child << " number " 
			<< i+1 << " positioned in " << rodName << " at " 
			<< r << " with no rotation";
  }

  // Side Clamps
  for (int i=0; i<(int)(sideClamp1DZ.size()); i++) {
    int j = i/2;
    DDTranslation r(sideClampX[i],moduleY[j],shift+moduleZ[j]+sideClamp1DZ[i]);
    DDName child(DDSplit(sideClamp1).first, DDSplit(sideClamp1).second);
    cpv.position(child, rodName, i+1, r, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child << " number " 
			<< i+1 << " positioned in " << rodName << " at "
			<< r << " with no rotation";
  }
  for (int i=0; i<(int)(sideClamp2DZ.size()); i++) {
    int j = i/2;
    DDTranslation r(sideClampX[i],moduleY[j],shift+moduleZ[j]+sideClamp2DZ[i]);
    DDName child(DDSplit(sideClamp2).first, DDSplit(sideClamp2).second);
    cpv.position(child, rodName, i+1, r, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child << " number " 
			<< i+1 << " positioned in " << rodName << " at "
			<< r << " with no rotation";
  }

  // End Rods
  for (int i=0; i<(int)(endRod1Y.size()); i++) {
    DDTranslation r(0, endRod1Y[i], shift+endRod1Z[i]);
    DDName child(DDSplit(endRod1).first, DDSplit(endRod1).second);
    cpv.position(child, centName, i+1, r, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child << " number "
			<< i+1 << " positioned in " << centName << " at "
			<< r << " with no rotation";
  }
  DDTranslation r1(0, endRod2Y, shift+endRod2Z);
  DDName child1(DDSplit(endRod2).first, DDSplit(endRod2).second);
  cpv.position(child1, centName, 1, r1, DDRotation());
  LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child1 << " number 1 "
		      << "positioned in " << centName << " at " << r1 
		      << " with no rotation";

  // End cooling tubes
  DDTranslation r2(0, endCoolY, shift+endCoolZ);
  std::string rotstr = DDSplit(endCoolRot).first;
  std::string rotns  = DDSplit(endCoolRot).second;
  DDRotation rot2(DDName(rotstr,rotns));
  DDName child2(DDSplit(endCool).first, DDSplit(endCool).second);
  cpv.position(child2, centName, 1, r2, rot2);
  LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child2 << " number 1 "
		      << "positioned in " << centName << " at " << r2 
		      << " with " << rot2;

  //Mother cable
  DDTranslation r3(0, 0, shift+cableZ);
  DDName child3(DDSplit(cable).first, DDSplit(cable).second);
  cpv.position(child3, centName, 1, r3, DDRotation());
  LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child3 << " number 1 "
		      << "positioned in " << centName << " at " << r3
		      << " with no rotation";

  //Modules
  for (int i=0; i<(int)(moduleRot.size()); i++) {
    DDTranslation r(0, moduleY[i], shift+moduleZ[i]);
    rotstr = DDSplit(moduleRot[i]).first;
    DDRotation rot;
    if (rotstr != "NULL") {
      rotns  = DDSplit(moduleRot[i]).second;
      rot = DDRotation(DDName(rotstr, rotns));
    }
    DDName child(DDSplit(module).first, DDSplit(module).second);
    cpv.position(child, centName, i+1, r, rot);
    LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child << " number " 
			<< i+1 << " positioned in " << centName << " at "
			<< r << " with " << rot;
  }

  //Connectors (ICC, CCUM, ...)
  for (int i=0; i<(int)(connect.size()); i++) {
    DDTranslation r(0, connectY[i], shift+connectZ[i]);
    DDName child(DDSplit(connect[i]).first, DDSplit(connect[i]).second);
    cpv.position(child, centName, i+1, r, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child << " number " 
			<< i+1 << " positioned in " << centName << " at "
			<< r << " with no rotation";
  }

  //AOH (only on ICCs)
  int copyNumber = 0;
  for (int i=0; i<(int)(aohCopies.size()); i++) {
    if(aohCopies[i] != 0) {
      // first copy with (+aohX,+aohZ) translation
      copyNumber++;
      DDTranslation r(aohX[i] + 0, aohY[i] + connectY[i], aohZ[i] + shift+connectZ[i]);
      DDName child(DDSplit(aohName).first, DDSplit(aohName).second);
      cpv.position(child, centName, copyNumber, r, DDRotation());
      LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child << " number " 
			  << copyNumber << " positioned in " << centName << " at "
			  << r << " with no rotation";
      // if two copies add a copy with (-aohX,-aohZ) translation
      if(aohCopies[i] == 2) {
	copyNumber++;
	DDTranslation r(-aohX[i] + 0, aohY[i] + connectY[i], -aohZ[i] + shift+connectZ[i]);
	DDName child(DDSplit(aohName).first, DDSplit(aohName).second);
	cpv.position(child, centName, copyNumber, r, DDRotation());
	LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child << " number " 
			    << copyNumber << " positioned in " << centName << " at "
			    << r << " with no rotation";
      }
      // if four copies add 3 copies with (-aohX,+aohZ) (-aohX,-aohZ) (+aohX,+aohZ) and translations
      if(aohCopies[i] == 4) {
	for (unsigned int j = 1; j<4; j++ ) {
	  copyNumber++;
	  switch(j) {
	  case 1:
	    {
	      DDTranslation r(-aohX[i] + 0, aohY[i] + connectY[i], +aohZ[i] + shift+connectZ[i]);
	      DDName child(DDSplit(aohName).first, DDSplit(aohName).second);
	      cpv.position(child, centName, copyNumber, r, DDRotation());
	      break;
	    }
	  case 2:
	    {
	      DDTranslation r(-aohX[i] + 0, aohY[i] + connectY[i], -aohZ[i] + shift+connectZ[i]);
	      DDName child(DDSplit(aohName).first, DDSplit(aohName).second);
	      cpv.position(child, centName, copyNumber, r, DDRotation());
	      break;
	    }
	  case 3:
	    {
	      DDTranslation r(+aohX[i] + 0, aohY[i] + connectY[i], -aohZ[i] + shift+connectZ[i]);
	      DDName child(DDSplit(aohName).first, DDSplit(aohName).second);
	      cpv.position(child, centName, copyNumber, r, DDRotation());
	      break;
	    }
	  }
	  LogDebug("TOBGeom") << "DDTOBRodAlgo test: " << child << " number " 
			      << copyNumber << " positioned in " << centName << " at "
			      << r << " with no rotation";
	}
      }
    }
  }
  
  LogDebug("TOBGeom") << "<<== End of DDTOBRodAlgo construction ...";
}
