// COCOA class implementation file
// Id:  Measurement.C
// CAT: Model
// ---------------------------------------------------------------------------
// History: v1.0
// Authors:
//   Pedro Arce

#include "Alignment/CocoaModel/interface/MeasurementDiffEntry.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>  // include floating-point std::abs functions
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/ALIVisLightPath.h"
#endif

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MeasurementDiffEntry::buildOptONamesList(const std::vector<ALIstring>& wl) {
  int NPairs = (wl.size() + 1) / 2;  // Number of OptO names ( pair of name and '&' )

  //--------- Fill list with names
  for (int ii = 0; ii < NPairs; ii++) {
    //--- take out Entry names from object names
    int isl = wl[ii * 2].rfind('/');
    AddOptONameListItem(wl[ii * 2].substr(0, isl));
    // Check for separating '&'
    if (ii != NPairs - 1 && wl[2 * ii + 1] != ALIstring("&")) {
      ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
      std::cerr << "!!! Measured Optical Objects should be separated by '&', not by" << wl[2 * ii + 1] << std::endl;
      exit(2);
    }
    //---- Fill entry names
    if (ii == 0) {
      theEntryNameFirst = wl[ii * 2].substr(isl + 1, 999);
    } else if (ii == 1) {
      theEntryNameSecond = wl[ii * 2].substr(isl + 1, 999);
    }
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ calculate the simulated value propagating the light ray through the OptO that take part in the Measurement
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void MeasurementDiffEntry::calculateSimulatedValue(ALIbool firstTime) {
  if (ALIUtils::debug >= 2)
    printStartCalculateSimulatedValue(this);  // important for Examples/FakeMeas

  //---------- Loop list of OptO that take part in measurement
  std::vector<OpticalObject*>::const_iterator vocite = OptOList().begin();
  if (ALIUtils::debug >= 5)
    std::cout << "OptOList size" << OptOList().size() << std::endl;

  //----- Check that there are only two objects
  if (OptOList().size() == !true) {
    std::cerr << "!!! ERROR in MeasurementDiffEntry: " << name() << " There should only be two objects " << std::endl;
    std::cerr << " 1st " << (*vocite)->name() << " 2nd " << (*vocite + 1)->name() << std::endl;
    DumpBadOrderOptOs();
    std::exception();
  }

#ifdef COCOA_VIS
  ALIVisLightPath* vispath = 0;
  if (ALIUtils::getFirstTime()) {
    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
    if (gomgr->GlobalOptions()["VisWriteIguana"] > 1) {
      vispath = IgCocoaFileMgr::getInstance().newLightPath(name());
    }
  }
#endif

  //--- This is a special 'measurement': it represents the fact that you have measured two entries one relative to the other (e.g. relative angle of two objects)
  Entry* entry1 = Model::getEntryByName((*(OptOList().begin()))->longName(), theEntryNameFirst);
  Entry* entry2 = Model::getEntryByName((*(OptOList().begin() + 1))->longName(), theEntryNameSecond);
  if (ALIUtils::debug >= 5)
    std::cout << "  entry1 " << (*(OptOList().begin()))->longName() << "/" << entry1->name() << " ->valueDisplaced() "
              << entry1->valueDisplaced() << " entry2 " << (*(OptOList().begin() + 1))->longName() << "/"
              << entry2->name() << " ->valueDisplaced() " << entry2->valueDisplaced() << std::endl;
  setValueSimulated(0, entry1->valueDisplaced() - entry2->valueDisplaced());

  if (ALIUtils::debug >= 2) {
    ALIdouble detD = 1000 * valueSimulated(0);
    if (std::abs(detD) <= 1.e-9)
      detD = 0.;
    std::cout << "REAL value: "
              << "D: " << 1000. * value()[0] << " (mm)  " << (this)->name() << "   DIFF= " << detD - 1000 * value()[0]
              << std::endl;
    std::cout << "SIMU value: "
              << "D: " << detD << " (mm)  " << (this)->name() << std::endl;
  }

  if (ALIUtils::debug >= 5)
    std::cout << "end calculateSimulatedValue" << std::endl;
}
