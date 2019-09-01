///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerZPosAlgo.cc
// Description: Position n copies at given z-values
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"

#include <string>
#include <vector>

using namespace std;

class DDTrackerZPosAlgo : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDTrackerZPosAlgo();
  ~DDTrackerZPosAlgo() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  vector<double> zvec;    //Z positions
  vector<string> rotMat;  //Names of rotation matrices

  string idNameSpace;  //Namespace of this and ALL sub-parts
  string childName;    //Child name
  int startCopyNo;     //Start Copy number
  int incrCopyNo;      //Increment in Copy number
};

DDTrackerZPosAlgo::DDTrackerZPosAlgo() { LogDebug("TrackerGeom") << "DDTrackerZPosAlgo info: Creating an instance"; }

DDTrackerZPosAlgo::~DDTrackerZPosAlgo() {}

void DDTrackerZPosAlgo::initialize(const DDNumericArguments& nArgs,
                                   const DDVectorArguments& vArgs,
                                   const DDMapArguments&,
                                   const DDStringArguments& sArgs,
                                   const DDStringVectorArguments& vsArgs) {
  startCopyNo = int(nArgs["StartCopyNo"]);
  incrCopyNo = int(nArgs["IncrCopyNo"]);
  zvec = vArgs["ZPositions"];
  rotMat = vsArgs["Rotations"];

  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];
  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerZPosAlgo debug: Parent " << parentName << "\tChild " << childName
                          << " NameSpace " << idNameSpace << "\tCopyNo (Start/Increment) " << startCopyNo << ", "
                          << incrCopyNo << "\tNumber " << zvec.size();
  for (int i = 0; i < (int)(zvec.size()); i++) {
    LogDebug("TrackerGeom") << "\t[" << i << "]\tZ = " << zvec[i] << ", Rot.Matrix = " << rotMat[i];
  }
}

void DDTrackerZPosAlgo::execute(DDCompactView& cpv) {
  int copy = startCopyNo;
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);

  for (int i = 0; i < (int)(zvec.size()); i++) {
    DDTranslation tran(0, 0, zvec[i]);
    string rotstr = DDSplit(rotMat[i]).first;
    DDRotation rot;
    if (rotstr != "NULL") {
      string rotns = DDSplit(rotMat[i]).second;
      rot = DDRotation(DDName(rotstr, rotns));
    }
    cpv.position(child, mother, copy, tran, rot);
    LogDebug("TrackerGeom") << "DDTrackerZPosAlgo test: " << child << " number " << copy << " positioned in " << mother
                            << " at " << tran << " with " << rot;
    copy += incrCopyNo;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDTrackerZPosAlgo, "track:DDTrackerZPosAlgo");
