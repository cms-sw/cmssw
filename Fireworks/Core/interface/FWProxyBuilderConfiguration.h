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
// $Id$
//

#include <string>
#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"

#ifndef __CINT__
#include <boost/shared_ptr.hpp>
#include <sigc++/sigc++.h>
#endif
class FWParameterBase;
class FWConfiguration;
class TGCompositeFrame;
class FWEventItem;

//==============================================================================
//==============================================================================

class FWProxyBuilderConfiguration : public FWConfigurableParameterizable,
                                    public FWParameterSetterEditorBase
{
public:
   /*
     struct StyleParams
     {
     FWLongParam m_lineWidth;
     FWLongParam m_lineStyle;
     FWLongParam m_pointSize;
     };*/

   FWProxyBuilderConfiguration(const FWConfiguration* c);
   virtual ~FWProxyBuilderConfiguration() {}

   FWParameterBase*  getVarParameter(const std::string& name, const FWEventItem*, FWViewType::EType type = FWViewType::kTypeSize);
   // StyleParams* getStyleParams();

   virtual void setFrom(const FWConfiguration& iFrom);
   virtual void addTo(FWConfiguration& iTo) const;

   void populateFrame(TGCompositeFrame* frame);

private:
   void buildVarParameters(const FWEventItem*, FWViewType::EType type = FWViewType::kTypeSize);

   const FWConfiguration*  m_txtConfig;

   std::vector<FWParameterBase* > m_varParameters;
   //  StyleParams*   m_lineParams;

#ifndef __CINT__
   std::vector<boost::shared_ptr<FWParameterSetterBase> > m_setters;
#endif

};
#endif
