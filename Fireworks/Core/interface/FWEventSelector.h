// -*- C++ -*-
#ifndef Fireworks_Core_FWEventSelector_h
#define Fireworks_Core_FWEventSelector_h
//
// Package:     newVersion
// Class  :     FWEventSelector
// $Id: FWEventSelector.h,v 1.5 2009/12/07 20:29:52 amraktad Exp $
//

// system include files
#include <string>

struct FWEventSelector
{
   FWEventSelector(FWEventSelector* s)
   {
      *this = *s;
   }

   FWEventSelector(): m_enabled(false), m_selected (-1), m_updated(false) {}

   std::string m_expression;
   std::string m_description;
   std::string m_triggerProcess;
   bool        m_enabled;
   int         m_selected;
   bool        m_updated;
};
#endif
