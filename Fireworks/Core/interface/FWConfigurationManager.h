#ifndef Fireworks_Core_FWConfigurationManager_h
#define Fireworks_Core_FWConfigurationManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWConfigurationManager
// 
/**\class FWConfigurationManager FWConfigurationManager.h Fireworks/Core/interface/FWConfigurationManager.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun Feb 24 14:38:41 EST 2008
// $Id$
//

// system include files
#include <map>

// user include files

// forward declarations
class FWConfigurable;
class FWConfiguration;

class FWConfigurationManager
{

   public:
      FWConfigurationManager();
      virtual ~FWConfigurationManager();

      // ---------- const member functions ---------------------
      void setFrom(const FWConfiguration&) const;
      void to(FWConfiguration&) const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      ///does not take ownership
      void add(const std::string& iName, FWConfigurable*);
   
   private:
      FWConfigurationManager(const FWConfigurationManager&); // stop default

      const FWConfigurationManager& operator=(const FWConfigurationManager&); // stop default

      // ---------- member data --------------------------------
      std::map<std::string, FWConfigurable*> m_configurables;
};


#endif
