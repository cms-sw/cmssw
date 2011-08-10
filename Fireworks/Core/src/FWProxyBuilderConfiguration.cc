// -*- C++ -*-
//
// Package:     Core
// Class  :     FWProxyBuilderConfiguration
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Wed Jul 27 00:58:43 CEST 2011
// $Id: FWProxyBuilderConfiguration.cc,v 1.2 2011/08/04 09:55:39 yana Exp $
//

// system include files

// user include files
#include <iostream>
#include <boost/bind.hpp>
#include "TGFrame.h"

#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWItemChangeSignal.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWParameterBase.h"
#include "Fireworks/Core/interface/FWGenericParameter.h"
#include "Fireworks/Core/interface/FWEnumParameter.h"


FWProxyBuilderConfiguration::FWProxyBuilderConfiguration(const FWConfiguration* c, const FWEventItem* item):
   m_txtConfig(c),
   m_item(item),
   m_styleParameters(0)
{
}

//==============================================================================
void
FWProxyBuilderConfiguration::assertStyleParameters()
{
   if ( ! m_styleParameters )
   {
      const FWConfiguration* c = m_txtConfig ? m_txtConfig->valueForKey("Style") : 0;
      m_styleParameters = new StyleParameters(c);

   }
}


double
FWProxyBuilderConfiguration::getPointSize()
{ 
   assertStyleParameters();
   if (!m_styleParameters->m_pointSize)
   {
      m_styleParameters->m_pointSize =  new FWDoubleParameter(0, "Point size", 1.0, 0.1, 10.0);
      if (m_styleParameters->m_config)
         m_styleParameters->m_pointSize->setFrom(*(m_styleParameters->m_config));

      m_styleParameters->m_pointSize->changed_.connect(boost::bind(&FWEventItem::proxyConfigChanged, (FWEventItem*)m_item));
   }

   return m_styleParameters->m_pointSize->value();
}


//______________________________________________________________________________


FWParameterBase* FWProxyBuilderConfiguration::getVarParameter(const std::string& name, FWViewType::EType type)
{
   for (FWConfigurableParameterizable::const_iterator i = begin(); i != end(); ++i)
   {
      if ((*i)->name() == name)
         return *i;
   }

   // std::cout << "FWProxyBuilderConfiguration::getVarParameter(). No parameter with name " << name << std::endl;
 
   const FWConfiguration* varConfig = m_txtConfig ? m_txtConfig->valueForKey("Var") : 0;

   // AMT:: the following could be templated

   if (m_item->purpose() == "Candidates")
   {
      FWBoolParameter*  mode = new FWBoolParameter(this, "Mode", false);
      if (varConfig) mode->setFrom(*varConfig);
      mode->changed_.connect(boost::bind(&FWEventItem::proxyConfigChanged, (FWEventItem*)m_item));
      return mode;
   }
   else {
      throw std::runtime_error("Invalid parameter request.");
      return 0;
   }
}


//______________________________________________________________________________

void
FWProxyBuilderConfiguration::addTo( FWConfiguration& iTo) const
{
   if (begin() != end()) {
      FWConfiguration vTmp;
      FWConfigurableParameterizable::addTo(vTmp);
      iTo.addKeyValue("Var",vTmp, true);
   }
   if (m_styleParameters) {
      FWConfiguration sTmp;

      if (m_styleParameters->m_pointSize) 
         m_styleParameters->m_pointSize->addTo(sTmp);

      iTo.addKeyValue("Style", sTmp, true);
   }
}


void
FWProxyBuilderConfiguration::setFrom(const FWConfiguration& iFrom)
{
   /*
     for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it!= keyVals->end(); ++it)
     std::cout << it->first << "FWProxyBuilderConfiguration::setFrom  " << std::endl;
     }*/
}

//______________________________________________________________________________

void 
FWProxyBuilderConfiguration::makeSetter(TGCompositeFrame* frame, FWParameterBase* pb )
{
   //  std::cout << "make setter " << pb->name() << std::endl;

   boost::shared_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(pb) );
   ptr->attach(pb, this); 
   TGFrame* tmpFrame = ptr->build(frame, false);
   frame->AddFrame(tmpFrame, new TGLayoutHints(kLHintsExpandX));
   m_setters.push_back(ptr);
}

void
FWProxyBuilderConfiguration::populateFrame(TGCompositeFrame* settersFrame)
{
   // std::cout << "populate \n";

   TGHorizontalFrame* frame =  new TGHorizontalFrame(settersFrame);
   settersFrame->AddFrame(frame, new TGLayoutHints(kLHintsExpandX) );
  
   for(const_iterator it =begin(); it != end(); ++it)
      makeSetter(frame, *it);

   if (m_styleParameters) {
      if (m_styleParameters->m_pointSize)  makeSetter(frame, m_styleParameters->m_pointSize );
   }

   settersFrame->MapSubwindows();   
}
