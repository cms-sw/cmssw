///////////////////////////////////////////////////////////////////////////////
// File: DDHCalFibreBundle.cc
// Description: Create & Position fibre bundles in mother
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/plugins/DDHCalFibreBundle.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDHCalFibreBundle::DDHCalFibreBundle() {
  LogDebug("HCalGeom") <<"DDHCalFibreBundle info: Creating an instance";
}

DDHCalFibreBundle::~DDHCalFibreBundle() {}

void DDHCalFibreBundle::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & ) {

  deltaPhi    = nArgs["DeltaPhi"];
  deltaZ      = nArgs["DeltaZ"];
  numberPhi   = int(nArgs["NumberPhi"]);
  material    = sArgs["Material"];
  areaSection = vArgs["AreaSection"];
  rStart      = vArgs["RadiusStart"];
  rEnd        = vArgs["RadiusEnd"];
  bundle      = dbl_to_int(vArgs["Bundles"]);
  tilt        = nArgs["TiltAngle"];

  DDCurrentNamespace ns;
  idNameSpace = *ns;
  childPrefix = sArgs["Child"]; 
  DDName parentName = parent().name();
  LogDebug("HCalGeom") << "DDHCalFibreBundle debug: Parent " << parentName
		       << " with " << bundle.size() << " children with prefix "
		       << childPrefix << ", material " << material << " with "
		       << numberPhi << " bundles along phi; width of mother "
		       << deltaZ << " along Z, " << deltaPhi/CLHEP::deg 
		       << " along phi and with " << rStart.size()
		       << " different bundle types";
  for (unsigned int i=0; i<areaSection.size(); ++i) 
    LogDebug("HCalGeom") << "DDHCalFibreBundle debug: Child[" << i << "] Area "
			 << areaSection[i] << " R at Start " << rStart[i]
			 << " R at End " << rEnd[i];
  LogDebug("HCalGeom") << "DDHCalFibreBundle debug: NameSpace " 
		       << idNameSpace << " Tilt Angle " << tilt/CLHEP::deg
		       << " Bundle type at different positions";
  for (unsigned int i=0; i<bundle.size(); ++i) {
    LogDebug("HCalGeom") << "DDHCalFibreBundle debug: Position[" << i << "] "
			 << " with Type " << bundle[i];
  }
}

void DDHCalFibreBundle::execute(DDCompactView& cpv) {

  DDName mother = parent().name();
  DDName matname(DDSplit(material).first, DDSplit(material).second);
  DDMaterial matter(matname);

  // Create the rotation matrices
  double dPhi = deltaPhi/numberPhi;
  std::vector<DDRotation> rotation;
  for (int i=0; i<numberPhi; ++i) {
    double phi    = -0.5*deltaPhi+(i+0.5)*dPhi;
    double phideg = phi/CLHEP::deg;
    std::string rotstr = "R0"+ std::to_string(phideg);
    DDRotation  rot = DDRotation(DDName(rotstr, idNameSpace));
    if (!rot) {
      LogDebug("HCalGeom") << "DDHCalFibreBundle test: Creating a new "
			   << "rotation " << rotstr << "\t" << 90 << ","
			   << phideg << ","  << 90 << "," << (phideg+90)
			   << ", 0, 0";
      rot = DDrot(DDName(rotstr, idNameSpace), 90*CLHEP::deg, phi, 
		  90*CLHEP::deg, (90*CLHEP::deg+phi), 0,  0);
    }
    rotation.emplace_back(rot);
  }

  // Create the solids and logical parts
  std::vector<DDLogicalPart> logs;
  for (unsigned int i=0; i<areaSection.size(); ++i) {
    double r0     = rEnd[i]/std::cos(tilt);
    double dStart = areaSection[i]/(2*dPhi*rStart[i]);
    double dEnd   = areaSection[i]/(2*dPhi*r0);
    std::string name = childPrefix + std::to_string(i);
    DDSolid solid = DDSolidFactory::cons(DDName(name, idNameSpace), 0.5*deltaZ,
					 rStart[i]-dStart, rStart[i]+dStart,
					 r0-dEnd, r0+dEnd, -0.5*dPhi, dPhi);
    LogDebug("HCalGeom") << "DDHCalFibreBundle test: Creating a new solid "
			 << name << " a cons with dZ " << deltaZ << " rStart "
			 << rStart[i]-dStart << ":"  << rStart[i]+dStart
			 << " rEnd " << r0-dEnd << ":" << r0+dEnd << " Phi " 
			 << -0.5*dPhi/CLHEP::deg << ":" << 0.5*dPhi/CLHEP::deg;
    DDLogicalPart log(DDName(name, idNameSpace), matter, solid);
    logs.emplace_back(log);
  }

  // Now posiiton them
  int    copy = 0;
  int    nY   = (int)(bundle.size())/numberPhi;
  for (unsigned int i=0; i<bundle.size(); i++) {
    DDTranslation tran(0,0,0);
    int ir = (int)(i)/nY;
    if (ir >= numberPhi) ir = numberPhi-1;
    int ib = bundle[i];
    copy++;
    if (ib>=0 && ib<(int)(logs.size())) {
      cpv.position(logs[ib], mother, copy, tran, rotation[ir]);
      LogDebug("HCalGeom") << "DDHCalFibreBundle test: " << logs[ib].name() 
			   << " number " << copy << " positioned in "
			   << mother << " at " << tran << " with " 
			   << rotation[ir];
    }
  }
}
