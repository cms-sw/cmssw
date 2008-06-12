#ifndef Fireworks_Core_FWModelFilter_h
#define Fireworks_Core_FWModelFilter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelFilter
// 
/**\class FWModelFilter FWModelFilter.h Fireworks/Core/interface/FWModelFilter.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Feb 29 13:39:51 PST 2008
// $Id: FWModelFilter.h,v 1.1 2008/03/01 02:14:18 chrjones Exp $
//

// system include files

// user include files
#include <string>

// forward declarations

class FWModelFilter
{

   public:
      FWModelFilter(const std::string& iExpression,
                   const std::string& iClassName);
      virtual ~FWModelFilter();

      // ---------- const member functions ---------------------
      const std::string& expression() const;
   
      bool passesFilter(const void*) const;
   
      const bool trivialFilter() const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void setExpression(const std::string& );
      void setClassName(const std::string& );
   
   private:
      //FWModelFilter(const FWModelFilter&); // stop default

      //const FWModelFilter& operator=(const FWModelFilter&); // stop default

      // ---------- member data --------------------------------
      std::string m_expression;
      std::string m_className;   
      std::string m_fullExpression;
};


#endif
