///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerLinear.cc
// Description: Position n copies at given intervals along an axis
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <string>
#include <vector>

using namespace std;

class DDTrackerLinear : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDTrackerLinear();
  ~DDTrackerLinear() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  string idNameSpace;     //Namespace of this and ALL sub-parts
  string childName;       //Child name
  int number;             //Number of positioning
  int startcn;            //Start copy no index
  int incrcn;             //Increment of copy no.
  double theta;           //Direction of translation
  double phi;             //  ......
  double offset;          //Offset    along (theta,phi) direction
  double delta;           //Increment     ................
  vector<double> centre;  //Centre
  string rotMat;          //Rotation matrix
};

DDTrackerLinear::DDTrackerLinear() : startcn(1), incrcn(1) {
  LogDebug("TrackerGeom") << "DDTrackerLinear info: Creating an instance";
}

DDTrackerLinear::~DDTrackerLinear() {}

void DDTrackerLinear::initialize(const DDNumericArguments& nArgs,
                                 const DDVectorArguments& vArgs,
                                 const DDMapArguments&,
                                 const DDStringArguments& sArgs,
                                 const DDStringVectorArguments&) {
  number = int(nArgs["Number"]);
  theta = nArgs["Theta"];
  phi = nArgs["Phi"];
  offset = nArgs["Offset"];
  delta = nArgs["Delta"];
  centre = vArgs["Center"];
  rotMat = sArgs["Rotation"];
  if (nArgs.find("StartCopyNo") != nArgs.end()) {
    startcn = size_t(nArgs["StartCopyNo"]);
  } else {
    startcn = 1;
  }
  if (nArgs.find("IncrCopyNo") != nArgs.end()) {
    incrcn = int(nArgs["IncrCopyNo"]);
  } else {
    incrcn = 1;
  }

  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];
  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerLinear debug: Parent " << parentName << "\tChild " << childName << " NameSpace "
                          << idNameSpace << "\tNumber " << number << "\tAxis (theta/phi) " << theta / CLHEP::deg << ", "
                          << phi / CLHEP::deg << "\t(Offset/Delta) " << offset << ", " << delta << "\tCentre "
                          << centre[0] << ", " << centre[1] << ", " << centre[2] << "\tRotation " << rotMat;
}

void DDTrackerLinear::execute(DDCompactView& cpv) {
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  DDTranslation direction(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
  DDTranslation base(centre[0], centre[1], centre[2]);
  string rotstr = DDSplit(rotMat).first;
  DDRotation rot;
  if (rotstr != "NULL") {
    string rotns = DDSplit(rotMat).second;
    rot = DDRotation(DDName(rotstr, rotns));
  }
  int ci = startcn;
  for (int i = 0; i < number; i++) {
    DDTranslation tran = base + (offset + double(i) * delta) * direction;
    cpv.position(child, mother, ci, tran, rot);
    LogDebug("TrackerGeom") << "DDTrackerLinear test: " << child << " number " << ci << " positioned in " << mother
                            << " at " << tran << " with " << rot;
    ++ci;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDTrackerLinear, "track:DDTrackerLinear");
