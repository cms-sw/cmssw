#include "DetectorDescription/RegressionTest/src/TinyDom2.h"

using namespace std;

void TinyDom2PrettyPrint(ostream & os , const TinyDom2 & dom)
{
  TinyDom2Walker walker(dom);
  //  unsigned int level = 0;
  //printTinyDom2(os, walker, level); // recursive
}




// void printTinyDom2(ostream & os, const TinyDom2Walker & w, unsigned int level)
// {
//   string space(level,' ');
//   os << space << "<" << w.current().first.str();
//   if (w.firstChild()) {
//     os << space << ">" << endl;
//     ++level;
//     printTinyDom2(os, w, level);
//     --level
//     os << space << "<" << w.current().first.str() << "/>" << endl;
//   }
//   else if (w.nextSibling()) {
//     os << space << ">" << endl;
//     //++level;
//     printTinyDom2(os, w, level);
//     //--level
//     os << space << "<" << w.current().first.str() << "/>" << endl;    
//   }
// }


