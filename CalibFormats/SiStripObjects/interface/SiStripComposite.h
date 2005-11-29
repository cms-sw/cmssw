#ifndef CALIBTRACKER_SISTRIPCONNECTIVITY_SISTRIPCOMPOSITE_H
#define CALIBTRACKER_SISTRIPCONNECTIVITY_SISTRIPCOMPOSITE_H

#include "CalibTracker/SiStripConnectivity/interface/ModuleConnection.h"
#include "CalibTracker/SiStripConnectivity/interface/CompositeHeader.h"

#include<string>
using namespace std;

class SiStripComposite {
 public:
  typedef vector<SiStripComposite*> SiStripCompositeVectorType;
  typedef vector<ModuleConnection*> ModuleConnectionVectorType;

  SiStripComposite(ModuleConnection, string, string,  string, CompositeHeader*);
  SiStripComposite(SiStripCompositeVectorType, string, string, string, CompositeHeader*);
  SiStripComposite();
  SiStripComposite(string, string,  string, CompositeHeader*);
  ~SiStripComposite();

  bool containsStructures(){return theSiStripCompositeList.size()>0;}
  bool containsModules();

  ModuleConnection* moduleConnection(){return theModuleConnection;}
  SiStripCompositeVectorType structures(){return theSiStripCompositeList;}
  
  void print();

  SiStripCompositeVectorType deepStructures();
  ModuleConnectionVectorType deepModules();
  //
  // add, set
  //
  void addStructure(SiStripComposite*);
  void addStructures(SiStripCompositeVectorType);
  void setModuleConnection(ModuleConnection);

  //
  // get
  //
  SiStripComposite* getModule(string in);
  SiStripComposite* getModule(int in);
  ModuleConnection*  getModuleConnection();

  //
  // name, header
  //
  void setName(string name){theName = name;}
  void setType(string name){theType = name;}
  void setPosition(string name){thePosition = name;}
  string name(){return theName;}
  string type(){return theType;}
  string position(){return thePosition;}
  void setHeader(CompositeHeader* in){theHeader = in;}
  CompositeHeader* header(){return theHeader;}

  SiStripComposite* findCompositeChild(string name, string type, string position);
 private:
  string theName;
  string theType;
  string thePosition;
  CompositeHeader* theHeader;
  //
  // used if that contains real modules
  //
  ModuleConnection* theModuleConnection;
  //
  // used if it is a bigger structure
  //
  SiStripCompositeVectorType theSiStripCompositeList;
};

#endif
