// -*- C++ -*-
//
// Package:     Core
// Class  :     unittest_fwmodelid
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jan 18 10:19:07 EST 2008
// $Id: unittest_fwmodelid.cc,v 1.1 2009/03/05 22:01:53 chrjones Exp $
//

// system include files
#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>

// user include files
#include "Fireworks/Core/interface/FWModelId.h"


//
// constants, enums and typedefs
//
BOOST_AUTO_TEST_CASE( fwmodelid )
{
   FWModelId one(0,1);
   FWModelId one2(0,1);
   
   BOOST_CHECK(not (one<one2));
   BOOST_CHECK(not (one2<one));
   
   FWModelId two(0,2);
   BOOST_CHECK(one < two);
   BOOST_CHECK(not (two < one) );
   
   FWModelId otherOne( reinterpret_cast<const FWEventItem*>(1),1);
   BOOST_CHECK(one < otherOne);
   BOOST_CHECK(not (otherOne < one));
   BOOST_CHECK(two < otherOne);
   BOOST_CHECK(not (otherOne < two));
}
