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
// $Id: FWProxyBuilderConfiguration.cc,v 1.1 2011/07/30 04:45:25 amraktad Exp $
//

// system include files

// user include files
#include <iostream>
//#include <sigc++/signal.h>
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

FWProxyBuilderConfiguration::FWProxyBuilderConfiguration(const FWConfiguration* c):
   m_txtConfig(c)
{
}

//______________________________________________________________________________


FWParameterBase* FWProxyBuilderConfiguration::getVarParameter(const std::string& name, const FWEventItem* item, FWViewType::EType type)
{
   if (begin() == end())
      buildVarParameters(item, type);

   for (FWConfigurableParameterizable::const_iterator i = begin(); i != end(); ++i)
   {
      if ((*i)->name() == name)
         return *i;
   }

   fwLog(fwlog::kDebug) << "FWProxyBuilderConfiguration::getVarParameter(). No parameter with name " << name << std::endl;
   return 0;
}

void FWProxyBuilderConfiguration::buildVarParameters(const FWEventItem* item, FWViewType::EType)
{
   const FWConfiguration* varConfig = m_txtConfig ? m_txtConfig->valueForKey("Var") : 0;
   if (item->purpose() == "Candidates")
   {
      FWBoolParameter*  mode = new FWBoolParameter(this, "Mode", false);
      if (varConfig) mode->setFrom(*varConfig);
      mode->changed_.connect(boost::bind(&FWEventItem::proxyConfigChanged, (FWEventItem*)item));
         
   }
}

//______________________________________________________________________________

void
FWProxyBuilderConfiguration::addTo( FWConfiguration& iTo) const
{
   FWConfiguration pbTmp;
   FWConfigurableParameterizable::addTo(pbTmp);
   iTo.addKeyValue("Var",pbTmp, true);

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
FWProxyBuilderConfiguration::populateFrame(TGCompositeFrame* settersFrame)
{
   if (begin() != end())
   {
      TGHorizontalFrame* frame =  new TGHorizontalFrame(settersFrame);
      settersFrame->AddFrame(frame, new TGLayoutHints(kLHintsExpandX) );
      // std::cout << "ADD setter " << frame << std::endl;
      for(const_iterator it =begin(); it != end(); ++it)
      {
         boost::shared_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(*it) );
         ptr->attach(*it, this);

         TGFrame* tmpFrame = ptr->build(frame, false);
         frame->AddFrame(tmpFrame, new TGLayoutHints(kLHintsExpandX));

         m_setters.push_back(ptr);
      }
      settersFrame->MapSubwindows();
   }
}
