#include <iostream>
#include <string>

#include "DetectorDescription/DDCMS/interface/DDSingleton.h"

using namespace std;

struct StrDDSingleton : public cms::DDSingleton< string, StrDDSingleton> {};

class TestClass
{
public:

  void test()
  {
    *m_singleton = "Hello world";
  }

  friend ostream &operator << ( ostream &o, const TestClass &t )
  {
    o << *t.m_singleton;
    return o;
  }

private:

  StrDDSingleton m_singleton;
};

int main(int argc, char *argv[])
{
  TestClass tst;
  for (int i = 0; i < 100000; ++i)
    tst.test();
  cout << tst << endl;
  return 0;
}
