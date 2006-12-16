#ifndef LibraryLoader_RootAutoLibraryLoader_h
#define LibraryLoader_RootAutoLibraryLoader_h
/**\class RootAutoLibraryLoader
 *
 * ROOT helper class which can automatically load the 
 * proper shared library when ROOT needs a new class dictionary
 *
 * \author Chris Jones, Cornell
 *
 * $Id: RootAutoLibraryLoader.h,v 1.4 2006/09/29 00:54:16 chrjones Exp $
 *
 */
#include "TClassGenerator.h"

class DummyClassToStopCompilerWarning;

namespace edm {
class RootAutoLibraryLoader : public TClassGenerator {
  friend class DummyClassToStopCompilerWarning;
public:
  /// return class type
  virtual TClass *GetClass(const char* classname, Bool_t load);
  /// return class type
  virtual TClass *GetClass(const type_info& typeinfo, Bool_t load);
  /// interface for TClass generators
  //ClassDef(RootAutoLibraryLoader,1);
  /// enable automatic library loading  
  static void enable();
  
  /// load all known libraries holding dictionaries
  static void loadAll();

private:
  const char* classNameAttemptingToLoad_; //!
  RootAutoLibraryLoader();
  RootAutoLibraryLoader(const RootAutoLibraryLoader&); // stop default
  const RootAutoLibraryLoader& operator=(const RootAutoLibraryLoader&); // stop default
};

}
#endif
