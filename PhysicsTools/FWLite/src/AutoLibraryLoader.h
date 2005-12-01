#ifndef LibraryLoader_AutoLibraryLoader_h
#define LibraryLoader_AutoLibraryLoader_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     AutoLibraryLoader
// 
/**\class AutoLibraryLoader AutoLibraryLoader.h AnalysisTools/FWLite/interface/AutoLibraryLoader.h

 Description: ROOT helper class which can automatically load the proper shared library when ROOT needs a new class dictionary

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Wed Nov 30 14:59:33 EST 2005
// $Id$
//

// system include files

// user include files
#include "TClassGenerator.h"

// forward declarations

class AutoLibraryLoader : public TClassGenerator
{

   public:

      virtual TClass *GetClass(const char* classname, Bool_t load);
      virtual TClass *GetClass(const type_info& typeinfo, Bool_t load);
      
      ClassDef(AutoLibraryLoader,1);  // interface for TClass generators

      static void enable();
   private:
      AutoLibraryLoader();
      AutoLibraryLoader(const AutoLibraryLoader&); // stop default

      const AutoLibraryLoader& operator=(const AutoLibraryLoader&); // stop default

      // ---------- member data --------------------------------

};


#endif
