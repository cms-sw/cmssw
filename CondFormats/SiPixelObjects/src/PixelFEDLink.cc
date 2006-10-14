#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include <iostream>
using namespace std;

void PixelFEDLink::clearRocs()
{
  typedef ROCs::const_iterator CIR;
  for (CIR it = theROCs.begin(); it != theROCs.end(); it++) delete (*it);
  theROCs.clear();
}

bool PixelFEDLink::checkRocNumbering() const
{
  bool result = true;
  int idx_expected = -1;
  typedef ROCs::const_iterator CIR;
  for (CIR it = theROCs.begin(); it != theROCs.end(); it++) {
    idx_expected++;
    if (idx_expected != (*it)->idInLink() ) {
      result = false;
      cout << "** PixelFEDLink, idInLink in ROC, expected: "
           << idx_expected <<" has: "<<(*it)->idInLink() << endl;
    }
    if (this != (*it)->link()) {
      result = false;
      cout << "** PixelFEDLink, wrong Link addres in ROC" << endl;
    }
  }
  return result;
}

ostream & operator<<(
    ostream& out, const PixelFEDLink & l)
{
  typedef PixelFEDLink::Connections Con;
  typedef Con::const_iterator IC;
  const Con & con = l.connected();

  int numberOfROCs = l.numberOfROCs();
  
  int idx = -1;
  out <<"id="<<l.id();
  for (IC ic = con.begin(); ic != con.end(); ic++) {
    out <<" "<<(*ic).name->name()
//      <<",r="<< (*ic).rocs 
        <<",ids:";
    for(int i = (*ic).rocs.first; i <= (*ic).rocs.second; i++) {
      idx++;
      if (idx < numberOfROCs ) out <<"_"
//                                 << l.roc(idx)->idInLink() 
//                                 <<"("
                                   << l.roc(idx)->idInDetUnit()
//                                 <<")"
                                   ;
    }
  }
  if (idx != numberOfROCs-1) out << "PROBLEM, sizes: " 
                                << idx <<" "<< numberOfROCs <<endl;

  return out;
}
