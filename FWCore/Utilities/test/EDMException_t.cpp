
#include "FWCore/FWUtilities/interface/EDMException.h"

#include <iostream>
#include <string>
#include <iomanip>
#include <typeinfo>

//using namespace cms;
using namespace std;

void func3()
{
  throw edm::Exception(edm::errors::GreatBadness)
    << "This is just a test"
    << endl;
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
  catch (edm::Exception& e)
    {
      //cerr << "GOT HERE" << endl;
      throw edm::Exception(edm::errors::Badness,"In func2",e)
	<< "Gave up";
    }
  catch (cms::Exception& e)
    {
      //cerr << "GOT HERE2 " << typeid(e).name() << endl;
      throw cms::Exception("edm::errors::Badness","In func2 bad place",e)
	<< "Gave up";
    }
  
}

const char answer[] = 
  "---- Badness BEGIN\n"
  "In func2\n"
  "---- GreatBadness BEGIN\n"
  "This is just a test\n" 
  "---- GreatBadness END\n"
  "Gave up\n"
  "---- Badness END\n"
  ;

const char* correct[] = { "Badness","GreatBadness" };

int main()
{
  try
    {
      func1();
    }
  catch (cms::Exception& e)
    {
      cerr << "*** main caught Exception, output is ***\n"
	   << "(" << e.what() << ")"
	   << "*** After exception output ***"
	   << endl;

      cerr << "\nCategory name list:\n";

#if 1
      if(e.what() != answer)
	{
	  cerr << "not right answer\n(" << answer << ")\n"
	       << endl;
	  abort();
	}
#endif

      cms::Exception::CategoryList::const_iterator i(e.history().begin()),
	b(e.history().end());

      if(e.history().size() !=2) 
	{
	  cerr << "Exception history is bad"  << endl;
	  abort();
	}

      for(int j=0;i!=b;++i,++j)
	{
	  cout << "  " << *i << "\n";
	  if(*i != correct[j])
	    { cerr << "bad category " << *i << endl; abort(); }
	}
    }
  return 0; 
}
