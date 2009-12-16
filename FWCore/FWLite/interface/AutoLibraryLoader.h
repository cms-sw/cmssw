#ifndef FWCore_FWLite_AutoLibraryLoader_h
#define FWCore_FWLite_AutoLibraryLoader_h
/**\class AutoLibraryLoader
 *
 * ROOT helper class which can automatically load the 
 * proper shared library when ROOT needs a new class dictionary
 *
 * \author Chris Jones, Cornell
 *
 * $Id: AutoLibraryLoader.h,v 1.2 2008/06/12 22:17:22 dsr Exp $
 *
 */
class DummyClassToStopCompilerWarning;

class AutoLibraryLoader {
  friend class DummyClassToStopCompilerWarning;
public:
  /// enable automatic library loading  
  static void enable();
  
  /// load all known libraries holding dictionaries
  static void loadAll();

private:
  static bool enabled_;
  AutoLibraryLoader();
  AutoLibraryLoader(const AutoLibraryLoader&); // stop default
  const AutoLibraryLoader& operator=(const AutoLibraryLoader&); // stop default
};


#endif
