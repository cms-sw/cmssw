// -*- C++ -*-
#ifndef Fireworks_Core_FWEventSelector_h
#define Fireworks_Core_FWEventSelector_h
//
// Package:     newVersion
// Class  :     FWEventSelector
// $Id: FWEventSelector.h,v 1.2 2009/11/13 20:58:17 amraktad Exp $
//

// system include files
#include <string>

struct FWEventSelector
{
   FWEventSelector(const char* iSelection, const char* iTitle, bool enable):
      m_expression(iSelection), m_description(iTitle), m_enabled(enable) {}

   FWEventSelector(FWEventSelector* s)
   {
      m_expression  = s->m_expression;
      m_description = s->m_description;
      m_enabled     = s->m_enabled;
   }

   FWEventSelector():m_enabled(false) {}

   std::string m_expression;
   std::string m_description;
   bool        m_enabled;
};
#endif
