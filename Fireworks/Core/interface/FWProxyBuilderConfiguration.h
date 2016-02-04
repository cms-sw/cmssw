#ifndef Fireworks_Core_FWProxyBuilderConfiguration_h
#define Fireworks_Core_FWProxyBuilderConfiguration_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWProxyBuilderConfiguration
// 
/**\class FWProxyBuilderConfiguration FWProxyBuilderConfiguration.h Fireworks/Core/interface/FWProxyBuilderConfiguration.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Wed Jul 27 00:58:35 CEST 2011
// $Id: FWProxyBuilderConfiguration.h,v 1.4 2011/08/16 01:16:05 amraktad Exp $
//

#include <string>
#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"

#include "Fireworks/Core/interface/FWParameters.h"
#include "Fireworks/Core/interface/FWGenericParameterWithRange.h"

#ifndef __CINT__
#include <boost/shared_ptr.hpp>
#include <sigc++/sigc++.h>
#endif

class TGCompositeFrame;
 
class FWParameterBase;
class FWConfiguration;
class FWEventItem;

//==============================================================================
class FWProxyBuilderConfiguration : public FWConfigurableParameterizable,
                                    public FWParameterSetterEditorBase
{
public:
   FWProxyBuilderConfiguration(const FWConfiguration* c, const FWEventItem* item);
   virtual ~FWProxyBuilderConfiguration();


   template <class T> FWGenericParameter<T>* assertParam(const std::string& name, T def);
   template <class T> FWGenericParameterWithRange<T>* assertParam(const std::string& name, T def, T min, T max);
   template <class T> T value(const std::string& name);


   virtual void setFrom(const FWConfiguration& iFrom);
   virtual void addTo(FWConfiguration& iTo) const;

   void populateFrame(TGCompositeFrame* frame);

private:
   void makeSetter(TGCompositeFrame*, FWParameterBase*);

   const FWConfiguration*  m_txtConfig;
   const FWEventItem*      m_item;

#ifndef __CINT__
   std::vector<boost::shared_ptr<FWParameterSetterBase> > m_setters;
#endif

};
#endif
