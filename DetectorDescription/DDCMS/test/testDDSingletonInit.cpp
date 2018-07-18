#include <iostream>
#include <string>

#include "DetectorDescription/DDCMS/interface/DDSingleton.h"

using namespace std;
using namespace cms;

struct StrSingleton : public DDSingleton<string, StrSingleton>
{
  StrSingleton() : DDSingleton(1) {}
  static std::unique_ptr<string> init() { return std::make_unique<string>( "Initialized" ); }
};

int main( int argc, char *argv[] )
{
  StrSingleton strSingleton;
  
  cout << *strSingleton << endl;
  
  *strSingleton = "Hello";
  
  cout << *strSingleton << endl;
  return 0;
}
