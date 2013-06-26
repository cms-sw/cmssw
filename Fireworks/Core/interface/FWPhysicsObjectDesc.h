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
// $Id: FWPhysicsObjectDesc.h,v 1.9 2012/08/03 18:20:27 wmtan Exp $
//

// system include files
#include <string>
#include "FWCore/Utilities/interface/TypeWithDict.h"

// user include files
#include "Fireworks/Core/interface/FWDisplayProperties.h"

// forward declarations

class FWPhysicsObjectDesc
{

public:
   FWPhysicsObjectDesc(const std::string& iName,
                       const TClass* iClass,
                       const std::string& iPurpose,
                       const FWDisplayProperties& iProperties =
                          FWDisplayProperties::defaultProperties,
                       const std::string& iModuleLabel = std::string(),
                       const std::string& iProductInstanceLabel = std::string(),
                       const std::string& iProcessName = std::string(),
                       const std::string& iFilterExpression = std::string(),
                       unsigned int iLayer=1);
   //virtual ~FWPhysicsObjectDesc();

   // ---------- const member functions ---------------------
   const FWDisplayProperties& displayProperties() const;
   const std::string& name() const;

   const TClass* type() const;
   const std::string& purpose() const;

   const std::string& moduleLabel() const;
   const std::string& productInstanceLabel() const;
   const std::string& processName() const;

   //objects with a larger layer number are draw on top of objects with a lower layer number
   unsigned int layer() const;

   const std::string& filterExpression() const;
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   void setLabels(const std::string& iModule,
                  const std::string& iProductInstance,
                  const std::string& iProcess);
   void setName(const std::string& iName);

   void setDisplayProperties( const FWDisplayProperties&);
private:
   //FWPhysicsObjectDesc(const FWPhysicsObjectDesc&); // stop default

   //const FWPhysicsObjectDesc& operator=(const FWPhysicsObjectDesc&); // stop default

   // ---------- member data --------------------------------
   std::string m_name;
   const TClass* m_type;
   const std::string m_purpose;
   FWDisplayProperties m_displayProperties;

   std::string m_moduleLabel;
   std::string m_productInstanceLabel;
   std::string m_processName;

   unsigned int m_layer;

   std::string m_filterExpression;
};


#endif
