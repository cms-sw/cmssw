#include "DetectorDescription/RegressionTest/src/TinyDom.h"

using namespace std;

void TinyDomPrettyPrint(ostream & os , const TinyDom & dom)
{
  TinyDomWalker walker(dom);
  //  unsigned int level = 0;
  //printTinyDom(os, walker, level); // recursive
}




// void printTinyDom(ostream & os, const TinyDomWalker & w, unsigned int level)
// {
//   string space(level,' ');
//   os << space << "<" << w.current().first.str();
//   if (w.firstChild()) {
//     os << space << ">" << endl;
//     ++level;
//     printTinyDom(os, w, level);
//     --level
//     os << space << "<" << w.current().first.str() << "/>" << endl;
//   }
//   else if (w.nextSibling()) {
//     os << space << ">" << endl;
//     //++level;
//     printTinyDom(os, w, level);
//     //--level
//     os << space << "<" << w.current().first.str() << "/>" << endl;    
//   }
// }
