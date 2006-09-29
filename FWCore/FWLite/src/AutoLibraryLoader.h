#ifndef LibraryLoader_AutoLibraryLoader_h
#define LibraryLoader_AutoLibraryLoader_h
/**\class AutoLibraryLoader
 *
 * ROOT helper class which can automatically load the 
 * proper shared library when ROOT needs a new class dictionary
 *
 * \author Chirs Jones, Cornell
 *
 * $Id: AutoLibraryLoader.h,v 1.3 2006/08/11 19:41:49 chrjones Exp $
 *
 */
#include "TClassGenerator.h"

class DummyClassToStopCompilerWarning;

class AutoLibraryLoader : public TClassGenerator {
  friend class DummyClassToStopCompilerWarning;
public:
  /// return class type
  virtual TClass *GetClass(const char* classname, Bool_t load);
  /// return class type
  virtual TClass *GetClass(const type_info& typeinfo, Bool_t load);
  /// interface for TClass generators
  ClassDef(AutoLibraryLoader,1);
  /// enable automatic library loading  
  static void enable();
  
  /// load all known libraries holding dictionaries
  static void loadAll();

private:
  const char* classNameAttemptingToLoad_; //!
  AutoLibraryLoader();
  AutoLibraryLoader(const AutoLibraryLoader&); // stop default
  const AutoLibraryLoader& operator=(const AutoLibraryLoader&); // stop default
};


#endif
