#include <algorithm>
#include <string>
#include <cctype>

using namespace std;

/** A simple C++ replacement for C's strupper */
string upcaseString(std::string aString) 
{
  transform(aString.begin(), aString.end(), aString.begin(), (int (*) (int))toupper);
  return aString;
}
