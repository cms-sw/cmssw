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
//

// system include files
#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <stdexcept>
#include <iostream>
#include <ios>
#include <fstream>

// user include files
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWConfigurable.h"
#include "Fireworks/Core/interface/FWConfigurationManager.h"

namespace {
   struct Conf : public FWConfigurable {

      virtual void addTo(FWConfiguration& iTop) const {
         iTop = m_config;
      }
      
      virtual void setFrom(const FWConfiguration& iFrom) {
         m_config = iFrom;
      }
      
      FWConfiguration m_config;
   };
}
//
// constants, enums and typedefs
//
BOOST_AUTO_TEST_CASE( fwconfiguration )
{
   FWConfiguration config;
   BOOST_CHECK( 1 == config.version());
   BOOST_CHECK( 0 == config.stringValues());
   BOOST_CHECK( 0 == config.keyValues());
   
   const std::string kValue("1.0");
   config.addValue(kValue);
   BOOST_REQUIRE( 0 != config.stringValues() );
   BOOST_CHECK( 1 == config.stringValues()->size() );
   BOOST_CHECK( 0 == config.keyValues() );
   BOOST_CHECK( kValue == config.value() );
   BOOST_CHECK_THROW(config.addKeyValue("one",FWConfiguration("two")), std::runtime_error);

   //copy constructor
   FWConfiguration config2(config);
   BOOST_CHECK( 1 == config2.version());
   BOOST_REQUIRE( 0 != config2.stringValues() );
   BOOST_CHECK( 1 == config2.stringValues()->size() );
   BOOST_CHECK( 0 == config2.keyValues() );
   BOOST_CHECK( kValue == config2.value() );
   
   //operator=
   FWConfiguration config3;
   config3 = config;
   BOOST_CHECK( 1 == config3.version());
   BOOST_REQUIRE( 0 != config3.stringValues() );
   BOOST_CHECK( 1 == config3.stringValues()->size() );
   BOOST_CHECK( 0 == config3.keyValues() );
   BOOST_CHECK( kValue == config3.value() );
   
   FWConfiguration valueForConst(kValue);
   BOOST_CHECK( 0 == valueForConst.version());
   BOOST_REQUIRE( 0 != valueForConst.stringValues() );
   BOOST_CHECK( 1 == valueForConst.stringValues()->size() );
   BOOST_CHECK( 0 == valueForConst.keyValues() );
   BOOST_CHECK( kValue == valueForConst.value() );
   
   FWConfiguration topConfig;
   topConfig.addKeyValue("first",config);
   BOOST_REQUIRE( 0 != topConfig.keyValues() );
   BOOST_CHECK( 1 == topConfig.keyValues()->size() );
   BOOST_CHECK( 0 == topConfig.stringValues() );
   BOOST_CHECK( std::string("first") == topConfig.keyValues()->front().first );
   BOOST_CHECK( kValue == topConfig.keyValues()->front().second.value() );
   BOOST_CHECK_THROW(topConfig.addValue("one"), std::runtime_error);
   const FWConfiguration* found = topConfig.valueForKey("second");
   BOOST_CHECK(0 == found);
   found = topConfig.valueForKey("first");
   BOOST_REQUIRE(0!=found);
   BOOST_CHECK(found->value()==kValue);
   BOOST_CHECK_THROW(config.valueForKey("blah"), std::runtime_error);
   
   FWConfiguration::streamTo(std::cout, topConfig, "top");

   //Test manager
   std::auto_ptr<Conf> pConf(new Conf() );
   
   FWConfigurationManager confMgr;
   confMgr.add("first", pConf.get() );
   
   confMgr.setFrom(topConfig);
   BOOST_REQUIRE( 0 != pConf->m_config.stringValues() );
   BOOST_CHECK( 1 == pConf->m_config.stringValues()->size() );
   BOOST_CHECK( 0 == pConf->m_config.keyValues() );
   BOOST_CHECK( kValue == pConf->m_config.value() );

   {
      FWConfiguration topConfig;
      confMgr.to(topConfig);
      
      BOOST_REQUIRE( 0 != topConfig.keyValues() );
      BOOST_CHECK( 1 == topConfig.keyValues()->size() );
      BOOST_CHECK( 0 == topConfig.stringValues() );
      BOOST_CHECK( std::string("first") == topConfig.keyValues()->front().first );
      BOOST_CHECK( kValue == topConfig.keyValues()->front().second.value() );
      found = topConfig.valueForKey("first");
      BOOST_REQUIRE(0!=found);
      BOOST_CHECK(found->value()==kValue);      
   }
   confMgr.writeToFile("testConfig");
   {
      FWConfiguration temp;
      pConf->m_config.swap(temp);
   }
   confMgr.readFromFile("testConfig");
   BOOST_REQUIRE( 0 != pConf->m_config.stringValues() );
   BOOST_CHECK( 1 == pConf->m_config.stringValues()->size() );
   BOOST_CHECK( 0 == pConf->m_config.keyValues() );
   BOOST_CHECK( kValue == pConf->m_config.value() );
   
   std::ofstream log("testConfig", std::ios_base::app | std::ios_base::out);
   log << "line\n"; log.close();
   try 
   {
      confMgr.readFromFile("testConfig");
   }
   catch (...) { std::cerr << "OK, checked parser exception \n"; }
}
