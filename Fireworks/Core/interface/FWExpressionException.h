#ifndef Fireworks_Core_FWExpressionException_h
#define Fireworks_Core_FWExpressionException_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWExpressionException
//
/**\class FWExpressionException FWExpressionException.h Fireworks/Core/interface/FWExpressionException.h

   Description: Holds information about an expression parsing failure

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Aug 21 14:22:22 EDT 2008
// $Id: FWExpressionException.h,v 1.3 2009/01/23 21:35:41 amraktad Exp $
//

// system include files
#include <string>

// user include files

// forward declarations

class FWExpressionException {

public:
   FWExpressionException(const std::string& iWhat, long iColumn) :
      m_what(iWhat), m_column(iColumn) {
   }
   //virtual ~FWExpressionException();

   // ---------- const member functions ---------------------
   const std::string& what() const {
      return m_what;
   }

   long column() const {
      return m_column;
   }
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   //FWExpressionException(const FWExpressionException&); // stop default

   //const FWExpressionException& operator=(const FWExpressionException&); // stop default

   // ---------- member data --------------------------------
   std::string m_what;
   long m_column;
};


#endif
