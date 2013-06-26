// -*- C++ -*-
//
// Package:     Core
// Class  :     FWPhysicsObjectDesc
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Jan 15 15:05:02 EST 2008
// $Id: FWPhysicsObjectDesc.cc,v 1.7 2009/04/07 15:58:40 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWPhysicsObjectDesc::FWPhysicsObjectDesc(const std::string& iName,
                                         const TClass* iClass,
                                         const std::string& iPurpose,
                                         const FWDisplayProperties& iProperties,
                                         const std::string& iModuleLabel,
                                         const std::string& iProductInstanceLabel,
                                         const std::string& iProcessName,
                                         const std::string& iFilterExpression,
                                         unsigned int iLayer) :
   m_name(iName),
   m_type(iClass),
   m_purpose(iPurpose),
   m_displayProperties(iProperties),
   m_moduleLabel(iModuleLabel),
   m_productInstanceLabel(iProductInstanceLabel),
   m_processName(iProcessName),
   m_layer(iLayer),
   m_filterExpression(iFilterExpression)
{
}

// FWPhysicsObjectDesc::FWPhysicsObjectDesc(const FWPhysicsObjectDesc& rhs)
// {
//    // do actual copying here;
// }

//FWPhysicsObjectDesc::~FWPhysicsObjectDesc()
//{
//}

//
// assignment operators
//
// const FWPhysicsObjectDesc& FWPhysicsObjectDesc::operator=(const FWPhysicsObjectDesc& rhs)
// {
//   //An exception safe implementation is
//   FWPhysicsObjectDesc temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWPhysicsObjectDesc::setLabels(const std::string& iModule,
                               const std::string& iProductInstance,
                               const std::string& iProcess)
{
   m_moduleLabel = iModule;
   m_productInstanceLabel = iProductInstance;
   m_processName = iProcess;
}

void
FWPhysicsObjectDesc::setName(const std::string& iName)
{
   m_name = iName;
}

void 
FWPhysicsObjectDesc::setDisplayProperties( const FWDisplayProperties& iProperties)
{
   m_displayProperties = iProperties;
}

//
// const member functions
//
const FWDisplayProperties&
FWPhysicsObjectDesc::displayProperties() const
{
   return m_displayProperties;
}

const std::string&
FWPhysicsObjectDesc::name() const
{
   return m_name;
}

const TClass*
FWPhysicsObjectDesc::type() const
{
   return m_type;
}

const std::string&
FWPhysicsObjectDesc::purpose() const
{
   return m_purpose;
}

const std::string&
FWPhysicsObjectDesc::moduleLabel() const
{
   return m_moduleLabel;
}
const std::string&
FWPhysicsObjectDesc::productInstanceLabel() const
{
   return m_productInstanceLabel;
}

const std::string&
FWPhysicsObjectDesc::processName() const
{
   return m_processName;
}

unsigned int
FWPhysicsObjectDesc::layer() const
{
   return m_layer;
}

const std::string&
FWPhysicsObjectDesc::filterExpression() const
{
   return m_filterExpression;
}
//
// static member functions
//
