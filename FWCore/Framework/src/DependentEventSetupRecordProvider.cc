// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     DependentEventSetupRecordProvider
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Sun May  1 17:15:52 EDT 2005
// $Id: DependentEventSetupRecordProvider.cc,v 1.1 2005/05/03 19:33:40 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/CoreFramework/interface/DependentEventSetupRecordProvider.h"
#include "FWCore/CoreFramework/interface/DependentRecordIntervalFinder.h"

#include "boost/bind.hpp"

//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {
//
// static data member definitions
//

//
// constructors and destructor
//
//DependentEventSetupRecordProvider::DependentEventSetupRecordProvider()
//{
//}

// DependentEventSetupRecordProvider::DependentEventSetupRecordProvider( const DependentEventSetupRecordProvider& rhs )
// {
//    // do actual copying here;
// }

//DependentEventSetupRecordProvider::~DependentEventSetupRecordProvider()
//{
//}

//
// assignment operators
//
// const DependentEventSetupRecordProvider& DependentEventSetupRecordProvider::operator=( const DependentEventSetupRecordProvider& rhs )
// {
//   //An exception safe implementation is
//   DependentEventSetupRecordProvider temp(rhs);
//   swap( rhs );
//
//   return *this;
// }

//
// member functions
//
void 
DependentEventSetupRecordProvider::setDependentProviders( const std::vector< boost::shared_ptr<EventSetupRecordProvider> >& iProviders)
{
   boost::shared_ptr< DependentRecordIntervalFinder > newFinder( 
                                       new DependentRecordIntervalFinder( key() ) );

   addFinder( newFinder );
   std::for_each( iProviders.begin(),
                  iProviders.end(),
                  boost::bind( std::mem_fun( &DependentRecordIntervalFinder::addProviderWeAreDependentOn), &(*newFinder), _1 ) );
}

//
// const member functions
//

//
// static member functions
//
   }
}
