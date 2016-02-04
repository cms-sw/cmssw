// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     Exception
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Werner Sun
//         Created:  Mon Mar 24 21:38:43 CET 2008
// $Id: Exception.cc,v 1.2 2009/08/14 19:58:11 wsun Exp $
//

// system include files

// user include files
#include "CondTools/L1Trigger/interface/Exception.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
l1t::DataAlreadyPresentException::DataAlreadyPresentException(
  const std::string& message )
  : cond::Exception( message )
{
}

// DataAlreadyPresentException::DataAlreadyPresentException(const DataAlreadyPresentException& rhs)
// {
//    // do actual copying here;
// }

l1t::DataAlreadyPresentException::~DataAlreadyPresentException() throw()
{
}

l1t::DataInvalidException::DataInvalidException(
  const std::string& message )
  : cond::Exception( message )
{
}

l1t::DataInvalidException::~DataInvalidException() throw()
{
}

//
// assignment operators
//
// const DataAlreadyPresentException& DataAlreadyPresentException::operator=(const DataAlreadyPresentException& rhs)
// {
//   //An exception safe implementation is
//   DataAlreadyPresentException temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
