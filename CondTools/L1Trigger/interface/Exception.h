#ifndef CondTools_L1Trigger_Exception_h
#define CondTools_L1Trigger_Exception_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     Exception
// 
/**\class Exception Exception.h CondTools/L1Trigger/interface/Exception.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Mon Mar 24 21:27:21 CET 2008
// $Id: Exception.h,v 1.2 2009/08/14 19:58:10 wsun Exp $
//

// system include files
#include <string>

// user include files
#include "CondCore/DBCommon/interface/Exception.h"

// forward declarations

namespace l1t {

  class DataAlreadyPresentException : public cond::Exception
  {

  public:
    explicit DataAlreadyPresentException( const std::string& message );
    virtual ~DataAlreadyPresentException() throw();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
    //DataAlreadyPresentException(const DataAlreadyPresentException&); // stop default

    //const DataAlreadyPresentException& operator=(const DataAlreadyPresentException&); // stop default

      // ---------- member data --------------------------------

  };

  class DataInvalidException : public cond::Exception
  {

  public:
    explicit DataInvalidException( const std::string& message );
    virtual ~DataInvalidException() throw();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
    //DataInvalidException(const DataInvalidException&); // stop default

    //const DataInvalidException& operator=(const DataInvalidException&); // stop default

      // ---------- member data --------------------------------

  };

}

#endif
