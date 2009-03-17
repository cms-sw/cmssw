// Unit test for L1GctHtMiss class.
// Trying to do proper TDD from now on!
// Author Robert Frazier


#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHtMiss.h"

#include <iostream>
#include <cstdlib>

using namespace std;

// Simple class that holds test input data and expected test output data
class TestIO
{
public:
  TestIO():rawInput(0), bxInput(0), etInput(0), phiInput(0), overflowInput(0),
           rawOutput(0), bxOutput(0), etOutput(0), phiOutput(0), overflowOutput(0) {}

  uint32_t rawInput;
  int16_t bxInput;
  unsigned etInput;
  unsigned phiInput;
  bool overflowInput;

  uint32_t rawOutput;
  int16_t bxOutput;
  unsigned etOutput;
  unsigned phiOutput;
  bool overflowOutput;  
};

// Function prototypes
bool testL1GctHtMiss(const std::string& testLabel,
                     const TestIO& testIO);

bool testL1GctHtMissInstance(const std::string& testLabel,
                             L1GctHtMiss& testObj,
                             const TestIO& testIO,
                             bool bxIsZeroNotValueInTestIO);  // A hack, as I'm sick of writing this goddamn test.

bool doObjTest(L1GctHtMiss& testObj,
               const TestIO& testIO,
               bool bxIsZeroNotValueInTestIO);  // A hack, as I'm sick of writing this goddamn test.


int main()
{
  cout << "---------------------------------------" << endl;
  cout << "RUNNING UNIT TEST FOR L1GctHtMiss CLASS" << endl;
  cout << "---------------------------------------" << endl;
    
  bool unitTestPassed = true;  // Try and prove this wrong...
  
  // Max values test input and expected output.
  TestIO maxTestData;
  maxTestData.rawInput = 0xffffffff;
  maxTestData.bxInput = 0x3fff;
  maxTestData.etInput = 0xffffffff;
  maxTestData.phiInput = 0xffffffff;
  maxTestData.overflowInput = true;
  maxTestData.rawOutput = maxTestData.rawInput;
  maxTestData.bxOutput = maxTestData.bxInput;
  maxTestData.etOutput = 0x7f;
  maxTestData.phiOutput = 0x1f;
  maxTestData.overflowOutput = maxTestData.overflowInput;
  
  // Random test input and expected output.
  TestIO rndTestData;
  rndTestData.rawInput = 0xd3b7a88e;
  rndTestData.bxInput = -17;
  rndTestData.etInput = 4;  // Corresponds with value that is packed in rawInput above
  rndTestData.phiInput = 14;  // Corresponds with value that is packed in rawInput above
  rndTestData.overflowInput = false;  // Corresponds with value that is packed in rawInput above
  rndTestData.rawOutput = rndTestData.rawInput;
  rndTestData.bxOutput = rndTestData.bxInput;
  rndTestData.etOutput = rndTestData.etInput;
  rndTestData.phiOutput = rndTestData.phiInput;
  rndTestData.overflowOutput = rndTestData.overflowInput;
  
  // Null test data for testing default constructor
  TestIO nullTestData;


  // NOW DO THE TESTS

  if(!testL1GctHtMiss("MAX VALUES", maxTestData)) { unitTestPassed = false; }
  if(!testL1GctHtMiss("RANDOM VALUES", rndTestData)) { unitTestPassed = false; }

  cout << "\nAND FINALLY, TEST THE DEFAULT CONSTRUCTOR..." << endl;
  // Default constructor test object.
  L1GctHtMiss defaultConstructorTestObj;
  if(!testL1GctHtMissInstance("DEFAULT CONSTRUCTOR", defaultConstructorTestObj, nullTestData, false)) { unitTestPassed = false; }

  
  // DISPLAY OVERALL RESULT  

  if(unitTestPassed) { cout << "\n\n----------------\nUnit test passed\n----------------" << endl; }
  else { cout << "\n\n-----------------\nUnit test FAILED!\n-----------------" << endl; }

  return 0;
}

bool testL1GctHtMiss(const std::string& testLabel,
                     const TestIO& testIO)
{
  bool testsPassed = true; // Try and prove wrong...

  cout << "\nSTART OF " << testLabel << " TESTS\n" << endl;

  // Constructor for the unpacker that takes only the raw data. 
  L1GctHtMiss rawOnlyConstructorTestObj(testIO.rawInput);
  if(!testL1GctHtMissInstance("RAW ONLY CONSTRUCTOR", rawOnlyConstructorTestObj, testIO, true)) { testsPassed = false; }

  // Constructor for the unpacker that takes the raw data and the bx. 
  L1GctHtMiss rawAndBxConstructorTestObj(testIO.rawInput, testIO.bxInput);
  if(!testL1GctHtMissInstance("RAW AND BX CONSTRUCTOR", rawAndBxConstructorTestObj, testIO, false)) { testsPassed = false; }

  // Constructor that takes Et, Phi, and overflow.
  L1GctHtMiss etPhiOverflowConstructorTestObj(testIO.etInput, testIO.phiInput, testIO.overflowInput);
  if(!testL1GctHtMissInstance("ET/PHI/OVERFLOW CONSTRUCTOR", etPhiOverflowConstructorTestObj, testIO, true)) { testsPassed = false; }

  // Constructor that takes Et, Phi, overflow, and bx.
  L1GctHtMiss etPhiOverflowBxConstructorTestObj(testIO.etInput, testIO.phiInput, testIO.overflowInput, testIO.bxInput);
  if(!testL1GctHtMissInstance("ET/PHI/OVERFLOW/BX CONSTRUCTOR", etPhiOverflowBxConstructorTestObj, testIO, false)) { testsPassed = false; }
  
  return testsPassed;
}



bool testL1GctHtMissInstance(const std::string& testLabel,
                             L1GctHtMiss& testObj,
                             const TestIO& testIO,
                             bool bxIsZeroNotValueInTestIO)
{
  bool testsPassed = true; // Try and prove wrong...
  
  cout << "\n  START OF " << testLabel << " SUB-TESTS **" << endl;

  // For testing the the copy ctor and assignment operators.
  L1GctHtMiss copyCtorTestObj(testObj);
  L1GctHtMiss assignmentOperatorTestObj;  assignmentOperatorTestObj = testObj;

  cout << "\n    1) Testing original object:" << endl;
  if(!doObjTest(testObj, testIO, bxIsZeroNotValueInTestIO)) { testsPassed = false; }

  cout << "\n    2) Testing copy constructed version of original object:" << endl;
  if(!doObjTest(copyCtorTestObj, testIO, bxIsZeroNotValueInTestIO)) { testsPassed = false; }
  
  cout << "\n    3) Test assignment operator version of original object:" << endl;
  if(!doObjTest(assignmentOperatorTestObj, testIO, bxIsZeroNotValueInTestIO)) { testsPassed = false; }  

  return testsPassed;
}

bool doObjTest(L1GctHtMiss& testObj,
               const TestIO& testIO,
               bool bxIsZeroNotValueInTestIO)
{
  bool allTestsPassed = true; // Try and prove wrong...
  
  bool testPassed;  // Reused for each individual test.
  
  testPassed = (testObj.name() == "HtMiss");
  if(!testPassed) { allTestsPassed = false; }
  cout << "      Test name(): \t" << (testPassed ? "passed." : "FAILED!") << endl;

  testPassed = (testObj.empty() == false);
  if(!testPassed) { allTestsPassed = true; }
  cout << "      Test empty(): \t" << (testPassed ? "passed." : "FAILED!") << endl;

  testPassed = (testObj.raw() == testIO.rawOutput);
  if(!testPassed) { allTestsPassed = false; }
  cout << "      Test raw(): \t" << (testPassed ? "passed." : "FAILED!") << endl;

  testPassed = (testObj.et() == testIO.etOutput);
  if(!testPassed) { allTestsPassed = false; }
  cout << "      Test et(): \t" << (testPassed ? "passed." : "FAILED!") << endl;

  testPassed = (testObj.phi() == testIO.phiOutput);
  if(!testPassed) { allTestsPassed = false; }
  cout << "      Test phi(): \t" << (testPassed ? "passed." : "FAILED!") << endl;

  testPassed = (testObj.overFlow() == testIO.overflowOutput);
  if(!testPassed) { allTestsPassed = false; }
  cout << "      Test overFlow(): \t" << (testPassed ? "passed." : "FAILED!") << endl;

  if(bxIsZeroNotValueInTestIO) { testPassed = (testObj.bx() == 0); }
  else { testPassed = (testObj.bx() == testIO.bxOutput); }
  if(!testPassed) { allTestsPassed = false; }
  cout << "      Test bx(): \t" << (testPassed ? "passed." : "FAILED!") << endl;
  
  return allTestsPassed;
}
