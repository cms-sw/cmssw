///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerPhiAlgo.cc
// Description: Position n copies at prescribed phi values
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

class DDTrackerPhiAlgo : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDTrackerPhiAlgo();
  ~DDTrackerPhiAlgo() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  double radius;        //Radius
  double tilt;          //Tilt angle
  vector<double> phi;   //Phi values
  vector<double> zpos;  //Z positions

  string idNameSpace;  //Namespace of this and ALL sub-parts
  string childName;    //Child name

  size_t startcn;    //Start index of copy numbers.
  int incrcn;        //Increment of copy number index.
  size_t numcopies;  //Number of copies == phi.size() above.
};

DDTrackerPhiAlgo::DDTrackerPhiAlgo() : startcn(1), incrcn(1) {
  LogDebug("TrackerGeom") << "DDTrackerPhiAlgo info: Creating an instance";
}

DDTrackerPhiAlgo::~DDTrackerPhiAlgo() {}

void DDTrackerPhiAlgo::initialize(const DDNumericArguments& nArgs,
                                  const DDVectorArguments& vArgs,
                                  const DDMapArguments&,
                                  const DDStringArguments& sArgs,
                                  const DDStringVectorArguments&) {
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

  radius = nArgs["Radius"];
  tilt = nArgs["Tilt"];
  phi = vArgs["Phi"];
  zpos = vArgs["ZPos"];

  if (nArgs.find("NumCopies") != nArgs.end()) {
    numcopies = size_t(nArgs["NumCopies"]);
    if (numcopies != phi.size()) {
      edm::LogError("TrackerGeom") << "DDTrackerPhiAlgo error: Parameter "
                                   << "NumCopies does not agree with the size "
                                   << "of the Phi vector. It was adjusted to "
                                   << "be the size of the Phi vector and may "
                                   << "lead to crashes or errors.";
    }
  } else {
    numcopies = phi.size();
  }

  LogDebug("TrackerGeom") << "DDTrackerPhiAlgo debug: Parameters for position"
                          << "ing:: "
                          << " Radius " << radius << " Tilt " << tilt / CLHEP::deg << " Copies " << phi.size() << " at";
  for (int i = 0; i < (int)(phi.size()); i++)
    LogDebug("TrackerGeom") << "\t[" << i << "] phi = " << phi[i] / CLHEP::deg << " z = " << zpos[i];

  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];
  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerPhiAlgo debug: Parent " << parentName << "\tChild " << childName << " NameSpace "
                          << idNameSpace;
}

void DDTrackerPhiAlgo::execute(DDCompactView& cpv) {
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  double theta = 90. * CLHEP::deg;
  size_t i = 0;
  int ci = startcn;
  for (; i < numcopies; ++i) {
    double phix = phi[i] + tilt;
    double phiy = phix + 90. * CLHEP::deg;
    double phideg = phi[i] / CLHEP::deg;

    string rotstr = DDSplit(childName).first + to_string(phideg);
    DDRotation rotation = DDRotation(DDName(rotstr, idNameSpace));
    if (!rotation) {
      LogDebug("TrackerGeom") << "DDTrackerPhiAlgo test: Creating a new "
                              << "rotation: " << rotstr << "\t"
                              << "90., " << phix / CLHEP::deg << ", 90.," << phiy / CLHEP::deg << ", 0, 0";
      rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy, 0., 0.);
    }

    double xpos = radius * cos(phi[i]);
    double ypos = radius * sin(phi[i]);
    DDTranslation tran(xpos, ypos, zpos[i]);

    cpv.position(child, mother, ci, tran, rotation);
    LogDebug("TrackerGeom") << "DDTrackerPhiAlgo test: " << child << " number " << ci << " positioned in " << mother
                            << " at " << tran << " with " << rotation;
    ci = ci + incrcn;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDTrackerPhiAlgo, "track:DDTrackerPhiAlgo");
