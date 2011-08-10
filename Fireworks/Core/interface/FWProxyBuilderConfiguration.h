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
// $Id: FWProxyBuilderConfiguration.h,v 1.1 2011/07/30 04:45:25 amraktad Exp $
//

#include <string>
#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"

#include "Fireworks/Core/interface/FWParameters.h"

#ifndef __CINT__
#include <boost/shared_ptr.hpp>
#include <sigc++/sigc++.h>
#endif

class TGCompositeFrame;
 
class FWParameterBase;
class FWConfiguration;
class FWEventItem;

//==============================================================================
//==============================================================================

class FWProxyBuilderConfiguration : public FWConfigurableParameterizable,
                                    public FWParameterSetterEditorBase
{
public:
   struct StyleParameters
   {
      FWDoubleParameter* m_pointSize;
      //FWLongParameter* m_lineWidth;
      const FWConfiguration*  m_config;

      StyleParameters( const FWConfiguration* c) : m_pointSize(0), m_config(c) {};
      ~StyleParameters() {};
   };

   FWProxyBuilderConfiguration(const FWConfiguration* c, const FWEventItem* item);
   virtual ~FWProxyBuilderConfiguration() {}

   FWParameterBase*  getVarParameter(const std::string& name, FWViewType::EType type = FWViewType::kTypeSize);

   double  getPointSize();

   virtual void setFrom(const FWConfiguration& iFrom);
   virtual void addTo(FWConfiguration& iTo) const;

   void populateFrame(TGCompositeFrame* frame);

private:
   void makeSetter(TGCompositeFrame*, FWParameterBase*);
   void assertStyleParameters();

   const FWConfiguration*  m_txtConfig;
   const FWEventItem*      m_item;

   StyleParameters*        m_styleParameters;

#ifndef __CINT__
   std::vector<boost::shared_ptr<FWParameterSetterBase> > m_setters;
#endif

};
#endif
