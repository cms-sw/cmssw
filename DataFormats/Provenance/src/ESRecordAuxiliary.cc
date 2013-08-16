// -*- C++ -*-
//
// Package:     Provenance
// Class  :     ESRecordAuxiliary
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu Dec  3 16:17:49 CST 2009
//

// system include files

// user include files
#include "DataFormats/Provenance/interface/ESRecordAuxiliary.h"

using namespace edm;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ESRecordAuxiliary::ESRecordAuxiliary()
{
}

ESRecordAuxiliary::ESRecordAuxiliary(const edm::EventID& iID, const edm::Timestamp& iTime):
eventID_(iID),
timestamp_(iTime)
{
}
// ESRecordAuxiliary::ESRecordAuxiliary(const ESRecordAuxiliary& rhs)
// {
//    // do actual copying here;
// }

//ESRecordAuxiliary::~ESRecordAuxiliary()
//{
//}

//
// assignment operators
//
// const ESRecordAuxiliary& ESRecordAuxiliary::operator=(const ESRecordAuxiliary& rhs)
// {
//   //An exception safe implementation is
//   ESRecordAuxiliary temp(rhs);
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
