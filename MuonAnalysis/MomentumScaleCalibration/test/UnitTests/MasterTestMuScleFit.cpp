/**
 * This file defines the executable that will run the tests.
 *
 * Author: M. De Mattia demattia@.pd.infn.it
 *
 * class description:
 *
 *
 * Version: $Id: MasterTestMuScleFit.cpp,v 1.1 2010/01/11 09:32:33 demattia Exp $
 */

// In case this does not work anymore uncomment the rest
// #include <Utilities/Testing/interface/CppUnit_testdriver.icpp>

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestRunner.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/CompilerOutputter.h>
// #include <cppunit/TextTestProgressListener.h>
#include <cppunit/BriefTestProgressListener.h>

/**
 * Main function used to run all tests.
 * We are not using the one in #include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
 * because we use the BriefTestProgressListener to output the name of each test.
 */

int main( int argc, char* argv[] )
{
  std::string testPath = (argc > 1) ? std::string(argv[1]) : "";
  CppUnit::TextUi::TestRunner runner;
  CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
  runner.addTest( registry.makeTest() );

  // Outputs the name of each test when it is executed.
  CppUnit::BriefTestProgressListener progress;
  runner.eventManager().addListener( &progress );
  runner.run();
}
