#ifndef Fireworks_Core_FWPhysicsObjectDesc_h
#define Fireworks_Core_FWPhysicsObjectDesc_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWPhysicsObjectDesc
// 
/**\class FWPhysicsObjectDesc FWPhysicsObjectDesc.h Fireworks/Core/interface/FWPhysicsObjectDesc.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jan 15 15:04:58 EST 2008
// $Id: FWPhysicsObjectDesc.h,v 1.1 2008/01/15 22:39:42 chrjones Exp $
//

// system include files
#include <string>
#include "Reflex/Type.h"

// user include files
#include "Fireworks/Core/interface/FWDisplayProperties.h"

// forward declarations

class FWPhysicsObjectDesc
{

   public:
      FWPhysicsObjectDesc(const std::string& iName,
                          const TClass* iClass,
                          const FWDisplayProperties& iProperties =
                          FWDisplayProperties(),
                          const std::string& iModuleLabel = std::string(),
                          const std::string& iProductInstanceLabel = std::string(),
                          const std::string& iProcessName = std::string(),
                          const std::string& iFilterExpression = std::string());
      //virtual ~FWPhysicsObjectDesc();

      // ---------- const member functions ---------------------
      const FWDisplayProperties& displayProperties() const;
      const std::string& name() const;

      const TClass* type() const;
   
      const std::string& moduleLabel() const;
      const std::string& productInstanceLabel() const;
      const std::string& processName() const;
   
      const std::string& filterExpression() const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   
      void setLabels(const std::string& iModule,
                     const std::string& iProductInstance,
                     const std::string& iProcess);
      void setName(const std::string& iName);
   
   private:
      //FWPhysicsObjectDesc(const FWPhysicsObjectDesc&); // stop default

      //const FWPhysicsObjectDesc& operator=(const FWPhysicsObjectDesc&); // stop default

      // ---------- member data --------------------------------
      std::string m_name;
      const TClass* m_type;
      FWDisplayProperties m_displayProperties;
   
      std::string m_moduleLabel;
      std::string m_productInstanceLabel;
      std::string m_processName;
      
      std::string m_filterExpression;
};


#endif
