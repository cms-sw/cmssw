#ifndef LibraryLoader_AutoLibraryLoader_h
#define LibraryLoader_AutoLibraryLoader_h
/**\class AutoLibraryLoader
 *
 * ROOT helper class which can automatically load the 
 * proper shared library when ROOT needs a new class dictionary
 *
 * \author Chirs Jones, Cornell
 *
 * $Id: AutoLibraryLoader.h,v 1.1 2006/05/17 16:43:44 wmtan Exp $
 *
 */
#include "TClassGenerator.h"

class AutoLibraryLoader : public TClassGenerator {
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
  AutoLibraryLoader();
  AutoLibraryLoader(const AutoLibraryLoader&); // stop default
  const AutoLibraryLoader& operator=(const AutoLibraryLoader&); // stop default
};


#endif
