
#include "FWCore/Utilities/interface/CodedException.h"

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <assert.h>
#include <limits>

//using namespace cms;
namespace edmtest
{

  // This is the list of error codes we shall use.
  enum ToyErrorCodes
    {
      Bad=3,
      Worse,
      Horrific,
      Amusing
    };


  // This is the kind of exception we shall throw.
  // I was confused at first, and thought we were supposed to use
  //     edm::CodedException<ToyErrorCodes>
  typedef edm::CodedException<ToyErrorCodes> ToyException;

}

// we must write this specialization.  it is somewhat awkward
// because of the edm namespace
namespace {
  struct FilledMap
  {
    FilledMap() : trans_()
    {
      std::cerr << "my loadmap got called" << std::endl;
      EDM_MAP_ENTRY(trans_,edmtest,Bad);
      EDM_MAP_ENTRY(trans_,edmtest,Worse);
      EDM_MAP_ENTRY(trans_,edmtest,Horrific);
      EDM_MAP_ENTRY(trans_,edmtest,Amusing);
    }

    edmtest::ToyException::CodeMap trans_;
  };
}

namespace edm {
void getCodeTable(edmtest::ToyException::CodeMap*& setme)
{
  static FilledMap fm;
  setme = &fm.trans_;
}
}

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

void simple()
{
  edmtest::ToyException h(edmtest::Horrific);
}

void func3()
{
  double d = 1.11111;
  float f = 2.22222;
  unsigned int i = std::numeric_limits<unsigned int>::max();
  std::string s("a string");
  char* c1 = const_cast<char *>("a nonconst pointer");
  char c2[] = "a c-style array";
  Thing thing(4);

  throw edmtest::ToyException(edmtest::Horrific)
    << "This is just a test: \n"
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
    << "uint:   " << i << "\n"
    << "string: " << s << "\n"
    << "char*:  " << std::setfill('.') << std::setw(20) << c1 << "\n"
    << "char[]: " << c2 << "\n"
    << "Thing:  " << thing
    << std::endl;
}

void func2()
{
  func3();
}

void func1()
{
  try 
    {
      func2();
    }
  catch (cms::Exception& e)
    {
      throw cms::Exception("Worse","In func2",e) << "Gave up";
    }
  
}

const char answer[] = 
  "---- Worse BEGIN\n"
  "In func2\n"
  "---- Horrific BEGIN\n"
  "This is just a test: \n" 
  "double: 1.11111\n"
  "float:  2.22222\n"
  "uint:   4294967295\n"
  "string: a string\n"
  "char*:  a nonconst pointer\n"
  "char[]: a c-style array\n"
  "Thing:  Thing(4)\n"
  "\n"
  "double: 1.111110e+00\n"
  "float:  2.22e+00\n"
  "uint:   4294967295\n"
  "string: a string\n"
  "char*:  ..a nonconst pointer\n"
  "char[]: a c-style array\n"
  "Thing:  Thing(4)\n"
  "---- Horrific END\n"
  "Gave up\n"
  "---- Worse END\n"
  ;

const char* correct[] = { "Worse","Horrific" };

int main()
{
  edmtest::ToyException ex(edmtest::Amusing, "Rats! Foiled again!\n");
  std::ostringstream oss;
  oss << ex;
  std::string s = oss.str();

  std::string expected("---- Amusing BEGIN\n"
		       "Rats! Foiled again!\n" 
		       "---- Amusing END\n");

  std::cerr << "ToyException message is:\n" << s << std::endl;
  assert ( s == expected );
  try {
    func1();
  }
  catch (cms::Exception& e) {
    std::cerr << "*** main caught Exception, output is ***\n"
	 << "(" << e.explainSelf() << ")"
	 << "*** After exception output ***"
	 << std::endl;

    std::cerr << "\nCategory name list:\n";

    if(e.explainSelf() != answer) {
      std::cerr << "not right answer\n(" << answer << ")\n"
	   << std::endl;
      abort();
    }

    cms::Exception::CategoryList::const_iterator i(e.history().begin()),
	b(e.history().end());

    if(e.history().size() !=2) abort();

    for(int j=0; i != b; ++i, ++j) {
      std::cout << "  " << *i << "\n";
      if(*i != correct[j])
	{ std::cerr << "bad category " << *i << std::endl; abort(); }
    }
  }
  return 0;
}
