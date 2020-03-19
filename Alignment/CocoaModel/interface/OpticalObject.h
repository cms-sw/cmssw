//   COCOA class header file
//Id:  OpticalObject.h
//CAT: Model
//
//   Base class to describe all types of Optical Objects
//
//   History: v1.0
//   Pedro Arce

#ifndef _OPTICALOBJECT_HH
#define _OPTICALOBJECT_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <cstdint>

class LightRay;
class Measurement;
class Entry;
class ALIFileIn;
class Measurement;
class ALIPlane;

class CocoaMaterialElementary;
class CocoaSolidShape;

#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include <fstream>
#include <vector>
class OpticalAlignInfo;
class OpticalAlignParam;

enum XYZcoor { XCoor, YCoor, ZCoor };

class OpticalObject {
  friend std::ostream& operator<<(std::ostream& os, const OpticalObject& c);

public:
  //---------- Constructors / destructor
  OpticalObject(){};
  OpticalObject(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data);
  virtual ~OpticalObject();

  //----- Steering function to read OptO data from SDF file (and also start component OptOs)
  void construct();

  void constructFromOptAligInfo(const OpticalAlignInfo& oaInfo);
  std::vector<ALIstring> getCoordinateFromOptAlignParam(const OpticalAlignParam& oaParam);
  void createComponentOptOsFromOptAlignInfo();

  virtual void constructMaterial();
  virtual void constructSolidShape();

  virtual void fillVRML(){};
  virtual void fillIguana(){};

  // ACCESS DATA MEMBERS
  const ALIstring& name() const { return theName; };
  const ALIstring& type() const { return theType; };
  const OpticalObject* parent() const { return theParent; };

  // Return List of Coordinate Entries
  const std::vector<Entry*>& CoordinateEntryList() const { return theCoordinateEntryVector; }
  // Return List of Extra Entries
  const std::vector<Entry*>& ExtraEntryList() const { return theExtraEntryVector; }
  // Return List of Extra Entry Values
  std::vector<ALIdouble>& ExtraEntryValueList() { return theExtraEntryValueVector; }

  const std::vector<ALIdouble>& ExtraEntryValueOriginalList() { return theExtraEntryValueOriginalVector; }
  const std::vector<ALIdouble>& ExtraEntryValueOriginalOriginalList() {
    return theExtraEntryValueOriginalOriginalVector;
  }

  // ACCESS CENTRE AND ROTATION DATA MEMBERS
  const CLHEP::Hep3Vector& centreGlob() const { return theCentreGlob; }

  const CLHEP::Hep3Vector& centreGlobal() const { return centreGlob(); }

  const CLHEP::Hep3Vector centreLocal() const;

  const CLHEP::Hep3Vector& centreGlobOriginal() const { return theCentreGlobOriginal; }
  const CLHEP::Hep3Vector& centreGlobOriginalOriginal() const { return theCentreGlobOriginalOriginal; }
  const CLHEP::HepRotation& rmGlob() const { return theRmGlob; }

  const CLHEP::HepRotation rmLocal() const;

  const CLHEP::HepRotation& rmGlobOriginal() const { return theRmGlobOriginal; }
  const CLHEP::HepRotation& rmGlobOriginalOriginal() const { return theRmGlobOriginalOriginal; }

  const double getEntryCentre(const XYZcoor coor) const;
  const double getEntryCentre(const ALIstring& coor) const;

  const double getEntryRMangle(const XYZcoor coor) const;
  const double getEntryRMangle(const ALIstring& coor) const;

  const ALIuint ID() const { return theCmsswID; }
  const ALIuint getCmsswID() const { return theCmsswID; }
  void setCmsswID(ALIuint id) { theCmsswID = id; }
  void setID(ALIuint id) { theCmsswID = id; }

  // SET DATA METHODS
  void setRmGlobalOriginal(const CLHEP::HepRotation& rm) { theRmGlobOriginal = rm; }
  void setGlobalRMOriginalOriginal(const CLHEP::HepRotation& rmoriori);
  void propagateGlobalRMOriginalOriginalChangeToChildren(const CLHEP::HepRotation& rmorioriold,
                                                         const CLHEP::HepRotation& rmoriorinew);
  CLHEP::HepRotation buildRmFromEntryValuesOriginalOriginal();

  void setRmGlobal(const CLHEP::HepRotation& rm) { theRmGlob = rm; }

  void setType(const ALIstring& type) { theType = type; }
  void addCoordinateEntryToList(Entry* entry) { theCoordinateEntryVector.push_back(entry); }
  void addExtraEntryToList(Entry* entry) { theExtraEntryVector.push_back(entry); }
  void addExtraEntryValueToList(ALIdouble entry_value) { theExtraEntryValueVector.push_back(entry_value); }
  void addExtraEntryValueOriginalToList(ALIdouble entry_value) {
    theExtraEntryValueOriginalVector.push_back(entry_value);
  }
  void addExtraEntryValueOriginalOriginalToList(ALIdouble entry_value) {
    theExtraEntryValueOriginalOriginalVector.push_back(entry_value);
  }

  //-  void test(){};
  //@@@@@----- METHODS USED IN Fit
  //---------- Propagate the light ray with the behaviour 'behav'
  virtual void participateInMeasurement(LightRay& lightray, Measurement& meas, const ALIstring& behav);
  //---------- default behaviour (depends of subclass type). A default behaviour can be makeMeasurement(), therefore you have to pass 'meas'
  virtual void defaultBehaviour(LightRay& lightray, Measurement& meas);
  //---------- Fast simulation of deviation of the light ray (reflection, shift, ...)
  virtual void fastDeviatesLightRay(LightRay& lightray);
  //---------- Detailed simulation of the light ray traversing
  virtual void fastTraversesLightRay(LightRay& lightray);
  //---------- Detailed simulation of deviation of the light ray (reflection, shift, ...)
  virtual void detailedDeviatesLightRay(LightRay& lightray);
  //---------- Fast simulation of the light ray traversing
  virtual void detailedTraversesLightRay(LightRay& lightray);

  //---------- Fast simulation of the light ray traversing
  virtual void makeMeasurement(LightRay& lightray, Measurement& meas);

  //---------- User Defined Behaviour
  virtual void userDefinedBehaviour(LightRay& lightray, Measurement& meas, const ALIstring& behav);

  //---------- Obtain the Z Axis of the OptO (ZAxis rotated with the OptO rotation matrix)
  CLHEP::Hep3Vector getZAxis() {
    CLHEP::Hep3Vector ZAxis(0., 0., 1.);
    CLHEP::HepRotation rmt = rmGlob();
    ZAxis = rmt * ZAxis;
    return ZAxis;
  }
  //---------- Get one of the plates of an OptO made of two parallel plates
  ALIPlane getPlate(const ALIbool forwardPlate, const ALIbool applyWedge);

  //---------- Displacements to get the derivative of a measurement w.r.t an entry
  //----- Displace the centre coordinate 'coor'
  void displaceCentreGlob(const XYZcoor coor, const ALIdouble disp);
  CLHEP::Hep3Vector getDisplacementInLocalCoordinates(const XYZcoor coor, const ALIdouble disp);
  void displaceCentreGlob(const CLHEP::Hep3Vector& dispVec);
  //----- Rotate around axis 'coor'
  void displaceRmGlobAroundGlobal(OpticalObject* opto1stRotated, const XYZcoor coor, const ALIdouble disp);
  void displaceRmGlobAroundLocal(OpticalObject* opto1stRotated, const XYZcoor coor, const ALIdouble disp);
  //----- Displace extra entry number 'entryNo'
  void displaceExtraEntry(const ALIuint entryNo, const ALIdouble disp);
  void setExtraEntryValue(const ALIuint entryNo, const ALIdouble disp);

  //----------- Displacements of original coordinates: do it to reset the original data
  //----------- every new iteration of the non linear fit
  //----- Displace the centre coordinate 'coor'
  void displaceCentreGlobOriginal(const XYZcoor coor, const ALIdouble disp);
  void displaceCentreGlobOriginal(const CLHEP::Hep3Vector& dispVec);
  void displaceCentreGlobOriginalOriginal(const XYZcoor coor, const ALIdouble disp);
  void displaceCentreGlobOriginalOriginal(const CLHEP::Hep3Vector& dispVec);
  //----- Rotate around axis 'coor'
  void displaceRmGlobOriginal(const OpticalObject* opto1stRotated, const XYZcoor coor, const ALIdouble disp);
  void displaceRmGlobOriginalOriginal(const OpticalObject* opto1stRotated, const XYZcoor coor, const ALIdouble disp);
  //----- Displace extra entry number 'entryNo'
  void displaceExtraEntryOriginal(const ALIuint entryNo, const ALIdouble disp);
  void displaceExtraEntryOriginalOriginal(const ALIuint entryNo, const ALIdouble disp);

  //---------- Reset the global coordinates and extra entries (after derivative is finished)
  void resetGlobalCoordinates();
  void resetOriginalOriginalCoordinates();

  // Find position of an extra entry in ExtraEntryList (and therefore in ExtraEntryValueList)
  const ALIint extraEntryNo(const ALIstring& entry_name) const;

  // //@@ Find an extra Entry by name and return its value. If entry not found, stop.
  const ALIdouble findExtraEntryValueMustExist(const ALIstring& eename) const;
  // Find an extra Entry by name and return its value, if it is not found return 0
  const ALIdouble findExtraEntryValue(const ALIstring& eename) const;

  // Find an extra Entry by name and pass its value. Return if entry is found or not
  const ALIbool findExtraEntryValueIfExists(const ALIstring& eename, ALIdouble& value) const;

  // Return the name of the OptO without its path
  const ALIstring shortName() const;
  // Return the name of the OptO with its path
  const ALIstring longName() const { return name(); }

  //! set current measurement
  void setMeas(Measurement* meas) { theCurrentMeas = meas; }
  Measurement* meas() { return theCurrentMeas; }

  std::vector<double> getRotationAnglesInOptOFrame(const OpticalObject* optoAncestor,
                                                   const std::vector<Entry*>& entries) const;
  std::vector<double> getLocalRotationAngles(const std::vector<Entry*>& entries) const;
  std::vector<double> getRotationAnglesFromMatrix(CLHEP::HepRotation& rmLocal,
                                                  const std::vector<Entry*>& entries) const;
  double diff2pi(double ang1, double ang2);
  bool eq2ang(double ang1, double ang2);
  double approxTo0(double val);
  double addPii(double val);
  int checkMatrixEquations(double angleX, double angleY, double angleZ, CLHEP::HepRotation* rot = nullptr);

  //-  CLHEP::Hep3Vector GetAxisForDisplacement( const XYZcoor coor );
  void setGlobalCoordinates();
  void setOriginalEntryValues();

  CocoaMaterialElementary* getMaterial() const { return theMaterial; }
  CocoaSolidShape* getSolidShape() const { return theSolidShape; }

  //private DATA METHODS
private:
  // Reads the data
  void readData(ALIFileIn& filein);
  // Copy the data from last OptO found of the same type
  void copyData();

  // Read extra entries
  void readExtraEntries(ALIFileIn& filein);

protected:
  // Create and fill an extra entry
  virtual void fillExtraEntry(std::vector<ALIstring>& wordlist);

private:
  // Read centre or angles
  void readCoordinates(const ALIstring& coor_type_read, const ALIstring& coor_type_expected, ALIFileIn& filein);
  // Create and fill a coordinate entry
  void fillCoordinateEntry(const ALIstring& coor_name, const std::vector<ALIstring>& wordlist);
  // Set angles null
  void setAnglesNull();

  //------ Build wordlist copying data of entry 'entry'
  void buildWordList(const Entry* entry, std::vector<ALIstring>& wordlist);

  // start Optical Objects that are components of current
  void createComponentOptOs(ALIFileIn& filein);

  OpticalObject* createNewOptO(OpticalObject* parent, ALIstring optoType, ALIstring optoName, ALIbool fcopyComponents);

  // Set global centre and rotation matrix
  void setGlobalCentre();
  void setGlobalRM();
  void setGlobalCoordinatesOfComponents();

  void SetCentreLocalFromEntryValues();
  void SetCentreGlobFromCentreLocal();
  void SetRMLocalFromEntryValues();
  void SetRMGlobFromRMLocal();
  void SetRMGlobFromRMLocalOriginalOriginal(const CLHEP::HepRotation& rmoriori);
  // Calculate local rot axis with new rm glob
  void calculateLocalRotationAxisInGlobal();

  void transformCylindrical2Cartesian();
  void transformSpherical2Cartesian();
  void transformGlobal2Local();

  template <class T>
  void rotateItAroundGlobal(T& object, const XYZcoor coor, const double disp);

  CLHEP::Hep3Vector getDispVec(const XYZcoor coor, const ALIdouble disp);

  void SetCentreIsGlobal(ALIbool isG) { centreIsGlobal = isG; }
  void SetAnglesIsGlobal(ALIbool isG) { anglesIsGlobal = isG; }

private:
  // private DATA MEMBERS
  OpticalObject* theParent;
  ALIstring theType;
  ALIstring theName;
  //----- Boolean to mark if data is going to be read or copied from closest previous OptO with same type
  ALIbool fcopyData;

  //----- Global centre and rotation matrix
  CLHEP::Hep3Vector theCentreGlob;
  CLHEP::HepRotation theRmGlob;
  //----- Original global centre and rotation matrix (for backup when they are changed to get derivatives)
  CLHEP::Hep3Vector theCentreGlobOriginal;
  CLHEP::HepRotation theRmGlobOriginal;
  CLHEP::Hep3Vector theCentreGlobOriginalOriginal;
  CLHEP::HepRotation theRmGlobOriginalOriginal;

  // Lists of entries
  std::vector<Entry*> theCoordinateEntryVector;
  std::vector<Entry*> theExtraEntryVector;

  // Lists of values of entries
  std::vector<ALIdouble> theExtraEntryValueVector;
  std::vector<ALIdouble> theExtraEntryValueOriginalVector;
  std::vector<ALIdouble> theExtraEntryValueOriginalOriginalVector;

  // centre and angles are global
  ALIbool centreIsGlobal;
  ALIbool anglesIsGlobal;

  Measurement* theCurrentMeas;

  CLHEP::Hep3Vector axisXLocalInGlobal;
  CLHEP::Hep3Vector axisYLocalInGlobal;
  CLHEP::Hep3Vector axisZLocalInGlobal;

  ALIuint theCmsswID;

protected:
  CocoaMaterialElementary* theMaterial;
  CocoaSolidShape* theSolidShape;

  ALIint verbose;
};

#endif
