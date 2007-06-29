// -*- C++ -*-
//
// Package:     <FWCore/Framework>
// Module:      EDLooperHelper
// 
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Valentin Kuznetsov
// Created:     Wed Jul 12 11:38:09 EDT 2006
// $Id: EDLooperHelper.cc,v 1.4 2007/06/22 23:26:33 wmtan Exp $
//
// Revision history
//
// $Log: EDLooperHelper.cc,v $
// Revision 1.4  2007/06/22 23:26:33  wmtan
// Add Run and Lumi loops to the EventProcessor
//
// Revision 1.3  2007/06/14 17:52:18  wmtan
// Remove unnecessary includes
//
// Revision 1.2  2006/10/13 01:47:35  wmtan
// Remove unnecessary argument from runOnce()
//
// Revision 1.1  2006/07/23 01:24:34  valya
// Add looper support into framework. The base class is EDLooper. All the work done in EventProcessor and EventHelperLooper
//

// user include files
#include "FWCore/Framework/interface/EDLooperHelper.h"
#include "FWCore/Framework/interface/EventProcessor.h"

namespace edm {

//
// constants, enums and typedefs
//

static const char* const kFacilityString = "FWCore.Framework.EDLooperHelper" ;

// ---- cvs-based strings (Id and Tag with which file was checked out)
static const char* const kIdString  = "$Id: EDLooperHelper.cc,v 1.4 2007/06/22 23:26:33 wmtan Exp $";
static const char* const kTagString = "$Name:  $";

//
// static data member definitions
//

//
// constructors and destructor
//
//EDLooperHelper::EDLooperHelper()
//{
//}

// EDLooperHelper::EDLooperHelper( const EDLooperHelper& rhs )
// {
//    // do actual copying here; if you implemented
//    // operator= correctly, you may be able to use just say      
//    *this = rhs;
// }

EDLooperHelper::~EDLooperHelper()
{
}

//
// assignment operators
//
// const EDLooperHelper& EDLooperHelper::operator=( const EDLooperHelper& rhs )
// {
//   if( this != &rhs ) {
//      // do actual copying here, plus:
//      // "SuperClass"::operator=( rhs );
//   }
//
//   return *this;
// }

//
// member functions
//
EventHelperDescription
EDLooperHelper::runOnce(boost::shared_ptr<edm::LuminosityBlockPrincipal> lbp)
{
    return eventProcessor_->runOnce(lbp);
}

void
EDLooperHelper::rewind(const std::set<edm::eventsetup::EventSetupRecordKey>& keys)
{
   std::for_each(keys.begin(),keys.end(), 
        boost::bind(&eventsetup::EventSetupProvider::resetRecordPlusDependentRecords,
                     eventProcessor_->esp_.get(), _1));
   return eventProcessor_->rewind();
}

//
// const member functions
//

//
// static member functions
//

} // end of namespace

