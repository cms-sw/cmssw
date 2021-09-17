///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerXYZPosAlgo.cc
// Description: Position n copies at given x-values, y-values and z-values
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

class DDTrackerXYZPosAlgo : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDTrackerXYZPosAlgo();
  ~DDTrackerXYZPosAlgo() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  vector<double> xvec;    //X positions
  vector<double> yvec;    //Y positions
  vector<double> zvec;    //Z positions
  vector<string> rotMat;  //Names of rotation matrices

  string idNameSpace;  //Namespace of this and ALL sub-parts
  string childName;    //Child name
  int startCopyNo;     //Start Copy number
  int incrCopyNo;      //Increment in Copy number
};

DDTrackerXYZPosAlgo::DDTrackerXYZPosAlgo() {
  LogDebug("TrackerGeom") << "DDTrackerXYZPosAlgo info: Creating an instance";
}

DDTrackerXYZPosAlgo::~DDTrackerXYZPosAlgo() {}

void DDTrackerXYZPosAlgo::initialize(const DDNumericArguments& nArgs,
                                     const DDVectorArguments& vArgs,
                                     const DDMapArguments&,
                                     const DDStringArguments& sArgs,
                                     const DDStringVectorArguments& vsArgs) {
  startCopyNo = int(nArgs["StartCopyNo"]);
  incrCopyNo = int(nArgs["IncrCopyNo"]);
  xvec = vArgs["XPositions"];
  yvec = vArgs["YPositions"];
  zvec = vArgs["ZPositions"];
  rotMat = vsArgs["Rotations"];

  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];
  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerXYZPosAlgo debug: Parent " << parentName << "\tChild " << childName
                          << " NameSpace " << idNameSpace << "\tCopyNo (Start/Increment) " << startCopyNo << ", "
                          << incrCopyNo << "\tNumber " << xvec.size() << ", " << yvec.size() << ", " << zvec.size();
  for (int i = 0; i < (int)(zvec.size()); i++) {
    LogDebug("TrackerGeom") << "\t[" << i << "]\tX = " << xvec[i] << "\t[" << i << "]\tY = " << yvec[i] << "\t[" << i
                            << "]\tZ = " << zvec[i] << ", Rot.Matrix = " << rotMat[i];
  }
}

void DDTrackerXYZPosAlgo::execute(DDCompactView& cpv) {
  int copy = startCopyNo;
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);

  for (int i = 0; i < (int)(zvec.size()); i++) {
    DDTranslation tran(xvec[i], yvec[i], zvec[i]);
    string rotstr = DDSplit(rotMat[i]).first;
    DDRotation rot;
    if (rotstr != "NULL") {
      string rotns = DDSplit(rotMat[i]).second;
      rot = DDRotation(DDName(rotstr, rotns));
    }
    cpv.position(child, mother, copy, tran, rot);
    LogDebug("TrackerGeom") << "DDTrackerXYZPosAlgo test: " << child << " number " << copy << " positioned in "
                            << mother << " at " << tran << " with " << rot;
    copy += incrCopyNo;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDTrackerXYZPosAlgo, "track:DDTrackerXYZPosAlgo");
