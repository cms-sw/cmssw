
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <memory>

struct Thing
{
  Thing():x() { }
  explicit Thing(int xx):x(xx) { }
  int x;
};

std::ostream& operator<<(std::ostream& os, const Thing& t)
{
  os << "Thing(" << t.x << ")";
  return os;
}

const char expected[] =   "An exception of category 'InfiniteLoop' occurred.\n"
                           "Exception Message:\n"
			   "In func1\n"
			   "This is just a test: \n"
			   "double: 1.11111\n"
			   "float:  2.22222\n"
			   "uint:   75\n"
			   "string: a string\n"
			   "char*:  a nonconst pointer\n"
			   "char[]: a c-style array\n"
			   "Thing:  Thing(4)\n"
			   "\n"
			   "double: 1.111110e+00\n"
			   "float:  2.22e+00\n"
			   "char*:  ..a nonconst pointer\n"
			   "\n"
			   "Gave up\n";

void func3()
{
  double d = 1.11111;
  float f = 2.22222;
  unsigned int i = 75U;
  std::string s("a string");
  char* c1 = const_cast<char *>("a nonconst pointer");
  char c2[] = "a c-style array";
  Thing thing(4);

  //  throw cms::Exception("DataCorrupt") 
  cms::Exception e("DataCorrupt");
  e << "This is just a test: \n"
    << "double: " << d << "\n"
    << "float:  " << f << "\n"
    << "uint:   " << i << "\n"
    << "string: " << s << "\n"
    << "char*:  " << c1 << "\n"
    << "char[]: " << c2 << "\n"
    << "Thing:  " << thing << "\n"
    << std::endl
    << "double: " << std::scientific << d << "\n"
    << "float:  " << std::setprecision(2) << f << "\n"
    << "char*:  " << std::setfill('.') << std::setw(20) << c1 << "\n"
    << std::endl;

  throw e;
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
  catch (cms::Exception& e) {
    cms::Exception toThrow("InfiniteLoop", "In func1", e);
    toThrow << "Gave up";
    throw toThrow;
  }  
}

int main()
{
  try {
    func1();
  }
  catch (cms::Exception& e) {
    std::cerr << "*** main caught Exception, output is ***\n"
              << e.explainSelf()
	      << "*** After exception output ***" << std::endl;

    if (e.explainSelf() != expected ||
        e.explainSelf() != std::string(e.what())) {
      std::cerr << "The output does not match the expected output.\n"
                << "The expected output is:\n"
                << expected << std::endl;
      abort();
    }
  }

  cms::Exception e1("ABC");
  if (e1.alreadyPrinted()) abort();
  e1.setAlreadyPrinted(true);
  if (!e1.alreadyPrinted()) abort();

  cms::Exception e1s(std::string("ABC"));
  if (e1s.alreadyPrinted()) abort();
  std::cerr << e1.what() << std::endl;
  if (e1.explainSelf() != std::string("An exception of category 'ABC' occurred.\n")) {
    abort();
  }
  if (e1.explainSelf() != e1s.explainSelf()) {
    abort();
  }

  cms::Exception e2("ABC", "foo");
  cms::Exception e2cs("ABC", std::string("foo"));
  cms::Exception e2sc(std::string("ABC"), "foo");
  cms::Exception e2ss(std::string("ABC"), std::string("foo"));
  if (e2.alreadyPrinted()) abort();
  if (e2cs.alreadyPrinted()) abort();
  if (e2sc.alreadyPrinted()) abort();
  if (e2ss.alreadyPrinted()) abort();
  e2 << "bar";
  e2cs << "bar";
  e2sc << "bar";
  e2ss << "bar";
  std::cerr << e2.what() << std::endl;
  if (e2.explainSelf() != std::string("An exception of category 'ABC' occurred.\n"
                                      "Exception Message:\n"
                                      "foo bar\n")) {
    abort();
  }
  if (e2.explainSelf() != e2cs.explainSelf()) {
    abort();
  }
  if (e2.explainSelf() != e2sc.explainSelf()) {
    abort();
  }
  if (e2.explainSelf() != e2ss.explainSelf()) {
    abort();
  }

  cms::Exception e3("ABC", "foo ");
  e3 << "bar\n";
  std::cerr << "e3\n" << e3.explainSelf() << std::endl;
  if (e3.explainSelf() != std::string("An exception of category 'ABC' occurred.\n"
                                      "Exception Message:\n"
                                      "foo bar\n")) {
    abort();
  }

  cms::Exception e4("ABC", "foo\n");
  e4 << "bar";
  std::cerr << "e4\n" << e4.explainSelf() << std::endl;
  if (e4.explainSelf() != std::string("An exception of category 'ABC' occurred.\n"
                                      "Exception Message:\n"
                                      "foo\nbar\n")) {
    abort();
  }

  e2.addContext("context1");
  e2.addContext(std::string("context2"));
  e2.addAdditionalInfo("info1");
  e2.addAdditionalInfo(std::string("info2"));
  std::cerr << e2.explainSelf() << std::endl;
  if (e2.explainSelf() != std::string("An exception of category 'ABC' occurred while\n"
                                      "   [0] context2\n"
                                      "   [1] context1\n"
                                      "Exception Message:\n"
                                      "foo bar\n"
                                      "   Additional Info:\n"
                                      "      [a] info2\n"
                                      "      [b] info1\n")) {
    abort();
  }

  cms::Exception e5("DEF", "start\n", e2);
  if (e5.alreadyPrinted()) abort();
  cms::Exception e6("DEF", "start", e2);
  std::string expected5("An exception of category 'DEF' occurred while\n"
                        "   [0] context2\n"
                        "   [1] context1\n"
                        "Exception Message:\n"
                        "start\n"
                        "foo bar"
                        "finish\n"
                        "   Additional Info:\n"
                        "      [a] info2\n"
                        "      [b] info1\n");
  e5 << "finish";
  e6 << "finish";
  std::cerr << e5.explainSelf() << std::endl;
  cms::Exception e7(e6);
  if (e7.alreadyPrinted()) abort();
  e6.setAlreadyPrinted(true);
  cms::Exception e9(e6);
  if (!e9.alreadyPrinted()) abort();
  
  if (e7.explainSelf() != expected5) {
    abort();
  }

  if (e7.category() != std::string("DEF")) {
    abort();
  }
  if (e7.message() != std::string("start\n"
                                  "foo bar"
                                  "finish")) {
    abort();
  }
  e7.clearContext();
  std::string expected7_1("An exception of category 'DEF' occurred.\n"
                          "Exception Message:\n"
                          "start\n"
                          "foo bar"
                          "finish\n"
                          "   Additional Info:\n"
                          "      [a] info2\n"
                          "      [b] info1\n");
  if (e7.explainSelf() != expected7_1) {
    abort();
  }
  std::list<std::string> newContext;
  newContext.push_back("new1");
  newContext.push_back("new2");
  newContext.push_back("new3");
  e7.setContext(newContext);
  std::cerr << e7;
  if (e7.context() != newContext) {
    abort();
  }

  e7.clearAdditionalInfo();
  std::string expected7_2("An exception of category 'DEF' occurred while\n"
                          "   [0] new3\n"
                          "   [1] new2\n"
                          "   [2] new1\n"
                          "Exception Message:\n"
                          "start\n"
                          "foo bar"
                          "finish\n");
  if (e7.explainSelf() != expected7_2) {
    abort();
  }
  std::list<std::string> newAdditionalInfo;
  newAdditionalInfo.push_back("newInfo1");
  newAdditionalInfo.push_back("newInfo2");
  newAdditionalInfo.push_back("newInfo3");
  e7.setAdditionalInfo(newAdditionalInfo);
  std::cerr << e7;
  if (e7.additionalInfo() != newAdditionalInfo) {
    abort();
  }
  std::string expected7_3("An exception of category 'DEF' occurred while\n"
                          "   [0] new3\n"
                          "   [1] new2\n"
                          "   [2] new1\n"
                          "Exception Message:\n"
                          "start\n"
                          "foo bar"
                          "finish\n"
                          "   Additional Info:\n"
                          "      [a] newInfo3\n"
                          "      [b] newInfo2\n"
                          "      [c] newInfo1\n");
  if (e7.explainSelf() != expected7_3) {
    abort();
  }
  if (e7.returnCode() != 8001) {
    abort();
  }
  e7.append(std::string(" X"));
  e7.append("Y");
  cms::Exception e8("ZZZ", "Z");
  e7.append(e8);
  std::string expected7_4("An exception of category 'DEF' occurred while\n"
                          "   [0] new3\n"
                          "   [1] new2\n"
                          "   [2] new1\n"
                          "Exception Message:\n"
                          "start\n"
                          "foo bar"
                          "finish XYZ \n"
                          "   Additional Info:\n"
                          "      [a] newInfo3\n"
                          "      [b] newInfo2\n"
                          "      [c] newInfo1\n");
  std::cerr << e7;
  if (e7.explainSelf() != expected7_4) {
    abort();
  }
  std::auto_ptr<cms::Exception> ptr(e7.clone());
  e7.clearMessage();
  std::string expected7_5("An exception of category 'DEF' occurred while\n"
                          "   [0] new3\n"
                          "   [1] new2\n"
                          "   [2] new1\n"
                          "   Additional Info:\n"
                          "      [a] newInfo3\n"
                          "      [b] newInfo2\n"
                          "      [c] newInfo1\n");
  std::cerr << e7;
  if (e7.explainSelf() != expected7_5) {
    abort();
  }

  try {
    ptr->raise();
  }
  catch (cms::Exception & ex) {
    ex << "last one ";
    std::cerr << ex;
    std::string expected7_6("An exception of category 'DEF' occurred while\n"
                            "   [0] new3\n"
                            "   [1] new2\n"
                            "   [2] new1\n"
                            "Exception Message:\n"
                            "start\n"
                            "foo bar"
                            "finish XYZ last one \n"
                            "   Additional Info:\n"
                            "      [a] newInfo3\n"
                            "      [b] newInfo2\n"
                            "      [c] newInfo1\n");
    if (ex.explainSelf() != expected7_6) {
      abort();
    }
  }
  return 0; 
}
