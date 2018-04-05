#ifndef OpticalObjectMgr_h
#define OpticalObjectMgr_h
/*---------------------------------------------------------------------------
ClassName:   OpticalObjectMgr
Author:      P. Arce
Changes:     02/05/01: creation  
---------------------------------------------------------------------------*/ 
// Description:
// Manages the set of optical objects 


#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <map>
#include "Alignment/CocoaModel/interface/OpticalObject.h"

typedef std::map<ALIstring, OpticalObject*, std::less<ALIstring> > msopto;

class OpticalObject;

class OpticalObjectMgr 
{
 public:    

  OpticalObjectMgr(){ };
  ~OpticalObjectMgr(){ };
  
  /// Get the only instance 
  static OpticalObjectMgr* getInstance();  
  
  // register an OpticalObject
  void registerMe( OpticalObject* opto ){
    theOptODict[opto->longName()] = opto;
  }
  // find an OpticalObject by long name (its name + name of its ancestors)
  OpticalObject* findOptO( const ALIstring& longName, bool exists = false ) const;  
  // find a list of OpticalObject's by name 
  std::vector<OpticalObject*> findOptOs( const ALIstring& name, bool exists = false ) const;  

  void dumpOptOs( std::ostream& out= std::cout ) const;

  ALIuint buildCmsSwID();

 private:
  static OpticalObjectMgr* theInstance;
  msopto theOptODict;
  ALIuint theLastCmsSwID;
};

#endif
