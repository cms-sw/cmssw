// -*- C++ -*-
//
// Package:     PhysicsTools/MVAComputer
// Class  :     testMVAComputer.cppunit
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 23 Jan 2015 18:54:27 GMT
//

// system include files

// user include files
#include <cppunit/extensions/HelperMacros.h>

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

class testMVAComputer: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testMVAComputer);
  
  CPPUNIT_TEST(multTest);
  CPPUNIT_TEST(optionalTest);
  CPPUNIT_TEST(foreachTest);
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  void setUp() {}
  void tearDown() {}
  
  void multTest();
  void optionalTest();
  void foreachTest();
  
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testMVAComputer);

using namespace PhysicsTools;

void
testMVAComputer::multTest() 
{
  {
    Calibration::MVAComputer calib;
    //this will be assigned to 'bit' 0
    calib.inputSet = {Calibration::Variable{AtomicId("x")}};
    //want to read out bit '1'
    calib.output = 1;
    
    //
    Calibration::ProcMultiply square;
    //we only want to read 1 input
    square.in = 1;
    //we will read bit '0' and multiply it by itself
    square.out = std::vector<Calibration::ProcMultiply::Config>{{0,0}};
    //input to use comes from bit '0' 
    square.inputVars.store={0b1};
    //number of bits stored in the last char (?)
    square.inputVars.bitsInLast = 1;
    
    calib.addProcessor(&square);
    
    MVAComputer mva(&calib,false);

    {
      std::vector<Variable::Value> input;
      input.emplace_back("x",2);

      CPPUNIT_ASSERT( 4 == mva.eval(input));
    }

    {
      std::vector<Variable::Value> input;
      input.emplace_back("x",3);

      CPPUNIT_ASSERT( 9 == mva.eval(input));
    }
  }

  {
    Calibration::MVAComputer calib;
    //this will be assigned to 'bit' 0 and 1
    calib.inputSet = {Calibration::Variable{AtomicId("x")}, Calibration::Variable{AtomicId("y")} };
    //want to read out bit '1'
    calib.output = 2;
    
    //
    Calibration::ProcMultiply square;
    //we only want to read 2 input
    square.in = 2;
    //we will read bit '0' and multiply it by bit '1'
    square.out = std::vector<Calibration::ProcMultiply::Config>{{0,1}};
    //input comes from bits '0' and '1'
    square.inputVars.store={0b11};
    //number of bits stored in the last char (?)
    square.inputVars.bitsInLast = 2;
    
    calib.addProcessor(&square);
    
    MVAComputer mva(&calib,false);
    
    std::vector<Variable::Value> input;
    input.emplace_back("x",2);
    input.emplace_back("y",3);

    CPPUNIT_ASSERT( 6 == mva.eval(input));
  }

}

void
testMVAComputer::optionalTest() 
{
  {
    Calibration::MVAComputer calib;
    //this will be assigned to 'bit' 0
    calib.inputSet = {Calibration::Variable{AtomicId("x")}};
    //want to read out bit '1'
    calib.output = 1;
    
    //
    Calibration::ProcOptional optional;
    //default
    optional.neutralPos = {1.};
    //input to use comes from bit '0' 
    optional.inputVars.store={0b1};
    //number of bits stored in the last char (?)
    optional.inputVars.bitsInLast = 1;
    
    calib.addProcessor(&optional);
    
    MVAComputer mva(&calib,false);

    {
      std::vector<Variable::Value> input;
      input.emplace_back("x",2);

      CPPUNIT_ASSERT( 2 == mva.eval(input));
    }

    {
      std::vector<Variable::Value> input;

      CPPUNIT_ASSERT( 1 == mva.eval(input));
    }

    {
      std::vector<Variable::Value> input;
      input.emplace_back("y",2);

      CPPUNIT_ASSERT( 1 == mva.eval(input));
    }

  }
}

void
testMVAComputer::foreachTest() 
{
  {
    Calibration::MVAComputer calib;
    //this will be assigned to 'bit' 0
    calib.inputSet = {Calibration::Variable{AtomicId("x")}};
    //want to read out bit '2'
    calib.output = 6;
    
    //
    Calibration::ProcForeach foreach;
    //we only want to read 1 input
    foreach.nProcs = 1;

    //input to use comes from bit '0' 
    foreach.inputVars.store={0b1};
    //number of bits stored in the last char (?)
    foreach.inputVars.bitsInLast = 1;
    
    calib.addProcessor(&foreach);
    

    //
    Calibration::ProcMultiply square;
    //we only want to read 1 input
    square.in = 1;
    //we will read bit '0' and multiply it by itself
    square.out = std::vector<Calibration::ProcMultiply::Config>{{0,0}};
    //input comes from bits '2' 
    square.inputVars.store={0b100};
    //number of bits stored in the last char (?)
    square.inputVars.bitsInLast = 3;
    
    calib.addProcessor(&square);

    //Need to break apart the output int different elements
    Calibration::ProcSplitter splitter;
    splitter.nFirst = 2;
    //input comes from bits '3' 
    splitter.inputVars.store={0b1000};
    //number of bits stored in the last char (?)
    splitter.inputVars.bitsInLast = 4;

    calib.addProcessor(&splitter);

    //
    Calibration::ProcMultiply join;
    //we only want to read 2 inputs
    join.in = 2;
    //we will read bit '4' and '5'
    join.out = std::vector<Calibration::ProcMultiply::Config>{{0,1}};
    //input comes from bits '4' and '5'
    join.inputVars.store={0b110000};
    //number of bits stored in the last char (?)
    join.inputVars.bitsInLast = 7;
    calib.addProcessor(&join);

    MVAComputer mva(&calib,false);

    {
      std::vector<Variable::Value> input;
      input.emplace_back("x",2);
      input.emplace_back("x",2);

      CPPUNIT_ASSERT( 4*4 == mva.eval(input));
    }

    {
      std::vector<Variable::Value> input;
      input.emplace_back("x",3);
      input.emplace_back("x",3);

      CPPUNIT_ASSERT( 9*9 == mva.eval(input));
    }
  }
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
