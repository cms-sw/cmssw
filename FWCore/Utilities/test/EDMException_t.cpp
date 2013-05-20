
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>
#include <string>
#include <iomanip>
#include <typeinfo>
#include <memory>

void func3()
{
  edm::Exception ex(edm::errors::NotFound);
  ex << "This is just a test";
  ex.addContext("new1");
  ex.addAdditionalInfo("info1");
  if (ex.returnCode() != 8026) {
    abort();
  }
  if (ex.category() != std::string("NotFound")) {
    abort();
  }
  throw ex;
}

void func2()
{
  func3();
}

void func1()
{
  try {
      func2();
  }
  catch (edm::Exception& e) {
    edm::Exception toThrow(edm::errors::Unknown, "In func2", e);
    edm::Exception toThrowString(edm::errors::Unknown, std::string("In func2"), e);
    if (toThrow.explainSelf() != toThrowString.explainSelf()) {
      abort();
    }
    toThrow << "\nGave up";
    toThrow.addContext("new2");
    toThrow.addAdditionalInfo("info2");
    if (toThrow.returnCode() != 8003) {
      abort();
    }
    if (toThrow.categoryCode() != edm::errors::Unknown) {
      abort();
    }
    cms::Exception* ptr = &toThrow;
    ptr->raise();
  }
}

const char answer[] = 
  "An exception of category 'Unknown' occurred while\n"
  "   [0] new2\n"
  "   [1] new1\n"
  "Exception Message:\n"
  "In func2\n"
  "This is just a test\n"
  "Gave up\n"
  "   Additional Info:\n"
  "      [a] info2\n"
  "      [b] info1\n";

int main()
{
  try {
    func1();
  }
  catch (edm::Exception& e) {
    if(e.explainSelf() != answer) {
	std::cerr << "Exception message incorrect.\n"
               "==expected==\n"
      << answer <<
      "\n==message==\n"
      <<e.explainSelf()
	     << std::endl;
	abort();
    }
    edm::Exception ecopy(e);
    if (e.explainSelf() != ecopy.explainSelf()) {
      abort();
    }
  }
  catch (cms::Exception& e) {
    abort();
  }

  edm::Exception e1(edm::errors::Unknown, "blah");
  edm::Exception e1String(edm::errors::Unknown, std::string("blah"));
  if (e1.explainSelf() != e1String.explainSelf()) {
    abort();
  }
  if (e1.returnCode() != 8003) {
    abort();
  }
  if (e1.category() != std::string("Unknown")) {
    abort();
  }
  if (e1.message() != std::string("blah ")) {
    abort();
  }
  cms::Exception* ptr = &e1;
  cms::Exception* ptrCloneCopy = ptr->clone();
  if (ptrCloneCopy->returnCode() != 8003) {
    abort();
  }
  try {
    edm::Exception::throwThis(edm::errors::ProductNotFound,
			      "a", "b", "c", "d", "e");
  }
  catch (edm::Exception & ex) {
    if (ex.explainSelf() != std::string(
      "An exception of category 'ProductNotFound' occurred.\n"
      "Exception Message:\n"
      "a bcde\n")) {
      abort();
    }
  }

  try {
    edm::Exception::throwThis(edm::errors::ProductNotFound, "a", 1, "b");
  }
  catch (edm::Exception & ex) {
    if (ex.explainSelf() != std::string(
      "An exception of category 'ProductNotFound' occurred.\n"
      "Exception Message:\n"
      "a 1b\n")) {
      abort();
    }
  }


  return 0; 
}
