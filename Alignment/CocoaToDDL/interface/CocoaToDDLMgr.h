//   COCOA class header file
//Id:  CocoaToDDLMgr.h
//CAT: Model
//
//   Class to manage the sets of fitted entries (one set per each measurement data set)
//
//   History: v1.0
//   Pedro Arce

#ifndef _CocoaToDDLMgr_HH
#define _CocoaToDDLMgr_HH

#include <map>
//#include <fstream>
#include "Alignment/CocoaUtilities/interface/ALIFileOut.h"

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"

#include "CLHEP/Vector/Rotation.h"

class CocoaMaterialElementary;
class CocoaSolidShape;
class OpticalObject;

class CocoaToDDLMgr {
public:
  //---------- Constructors / Destructor
  CocoaToDDLMgr(){};
  ~CocoaToDDLMgr(){};
  static CocoaToDDLMgr* getInstance();

  void writeDDDFile(ALIstring filename);
  void writeHeader(ALIstring filename);
  void writeMaterials();
  void writeSolids();
  void writeLogicalVolumes();
  void writePhysicalVolumes();
  void writeRotations();
  void writeSpecPars();
  void measurementsAsSpecPars();

  void newPartPre(std::string name);
  void newPartPost(std::string name, std::string extension);
  void newSectPre_ma(std::string name);
  void ma(CocoaMaterialElementary* ma);
  void newSectPost_ma(std::string name);
  void newSectPre_so(std::string name);
  void so(OpticalObject* opto);
  void newSectPost_so(std::string name);
  void newSectPre_lv(std::string name);
  void lv(OpticalObject* opto);
  void newSectPost_lv(std::string name);
  void newSectPre_pv(std::string name);
  void pv(OpticalObject* opto);
  void newSectPost_pv(std::string name);
  void newSectPre_ro(std::string name);
  void ro(const CLHEP::HepRotation& ro, int n);
  void newSectPost_ro(std::string name);
  void newSectPre_specPar(std::string name);
  void specPar(OpticalObject* opto);
  void writeSpecParsCocoa();
  void newSectPost_specPar(std::string name);
  void newSectPre(std::string name, std::string type);
  void newSectPost(std::string name);
  ALIbool materialIsRepeated(CocoaMaterialElementary* ma);
  ALIint buildRotationNumber(OpticalObject* opto);

  std::string scrubString(const std::string& s);

private:
  static CocoaToDDLMgr* instance;

  ALIFileOut file_;
  std::string filename_;

  std::vector<CocoaMaterialElementary*> theMaterialList;
  std::vector<CLHEP::HepRotation> theRotationList;
};
#endif
