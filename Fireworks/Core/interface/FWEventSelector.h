// -*- C++ -*-
#ifndef Fireworks_Core_FWEventSelector_h
#define Fireworks_Core_FWEventSelector_h
//
// Package:     newVersion
// Class  :     FWEventSelector
// $Id: FWEventSelector.h,v 1.4 2009/11/30 12:37:33 amraktad Exp $
//

// system include files
#include <string>

struct FWEventSelector
{
   FWEventSelector(const char* iSelection, const char* iTitle, bool enable):
      m_expression(iSelection), m_description(iTitle), m_enabled(enable), m_selected (-1), m_updated(false){}

   FWEventSelector(FWEventSelector* s)
   {
      m_expression  = s->m_expression;
      m_description = s->m_description;
      m_enabled     = s->m_enabled;
      m_selected    = s->m_selected;
      m_updated     = s->m_updated;
   }

   FWEventSelector():m_enabled(false), m_selected (-1), m_updated(false) {}

   std::string m_expression;
   std::string m_description;
   bool        m_enabled;
   int         m_selected;
   bool        m_updated;
};
#endif
