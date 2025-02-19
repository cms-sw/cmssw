#ifndef FWCore_Services_LoadAllDictionaries_h
#define FWCore_Services_LoadAllDictionaries_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     LoadAllDictionaries
// 
/**\class LoadAllDictionaries LoadAllDictionaries.h FWCore/Services/interface/LoadAllDictionaries.h

 Description: Loads all Capability dictionaries

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Sep 15 09:47:42 EDT 2005
// $Id: LoadAllDictionaries.h,v 1.5 2010/03/09 16:24:55 wdd Exp $
//

// system include files

// user include files

// forward declarations
namespace edm {
   class ParameterSet;
   class ConfigurationDescriptions;
   namespace service {
      class LoadAllDictionaries
   {
      
   public:
      LoadAllDictionaries(const edm::ParameterSet&);
      //virtual ~LoadAllDictionaries();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

      // ---------- member functions ---------------------------

   private:
      LoadAllDictionaries(const LoadAllDictionaries&); // stop default

      const LoadAllDictionaries& operator=(const LoadAllDictionaries&); // stop default

      // ---------- member data --------------------------------

   };
   }
}

#endif
