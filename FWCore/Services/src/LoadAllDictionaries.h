#ifndef Services_LoadAllDictionaries_h
#define Services_LoadAllDictionaries_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     LoadAllDictionaries
// 
/**\class LoadAllDictionaries LoadAllDictionaries.h FWCore/Services/interface/LoadAllDictionaries.h

 Description: Loads all seal Capability dictionaries

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Sep 15 09:47:42 EDT 2005
// $Id: LoadAllDictionaries.h,v 1.1 2005/09/15 15:22:59 chrjones Exp $
//

// system include files

// user include files

// forward declarations
namespace edm {
   class ParameterSet;
   namespace service {
      class LoadAllDictionaries
   {
      
   public:
      LoadAllDictionaries(const edm::ParameterSet&);
      //virtual ~LoadAllDictionaries();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      LoadAllDictionaries(const LoadAllDictionaries&); // stop default

      const LoadAllDictionaries& operator=(const LoadAllDictionaries&); // stop default

      // ---------- member data --------------------------------

   };
   }
}

#endif
