// -*- C++ -*-
#ifndef Fireworks_Core_FWEventSelector_h
#define Fireworks_Core_FWEventSelector_h
//
// Package:     newVersion
// Class  :     FWEventSelector
// $Id: FWEventSelector.h,v 1.3 2009/11/17 22:24:34 amraktad Exp $
//

// system include files
#include <string>

struct FWEventSelector
{
   FWEventSelector(const char* iSelection, const char* iTitle, bool enable):
      m_expression(iSelection), m_description(iTitle), m_enabled(enable), m_selected (-1){}

   FWEventSelector(FWEventSelector* s)
   {
      m_expression  = s->m_expression;
      m_description = s->m_description;
      m_enabled     = s->m_enabled;
      m_selected    = s->m_selected;
   }

   FWEventSelector():m_enabled(false) {}

   std::string m_expression;
   std::string m_description;
   bool        m_enabled;
   int         m_selected;
};
#endif
