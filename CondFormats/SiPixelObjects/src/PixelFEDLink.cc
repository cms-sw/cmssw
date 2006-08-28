#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <sstream>
#include <iostream>

using namespace std;


bool PixelFEDLink::checkRocNumbering() const
{
  bool result = true;
  int idx_expected = -1;
  typedef ROCs::const_iterator CIR;
  for (CIR it = theROCs.begin(); it != theROCs.end(); it++) {
    idx_expected++;
    if (idx_expected != (*it).idInLink() ) {
      result = false;
      cout << "** PixelFEDLink, idInLink in ROC, expected: "
           << idx_expected <<" has: "<<(*it).idInLink() << endl;
    }
  }
  return result;
}

string PixelFEDLink::print(int depth) const
{
  ostringstream out;
  typedef Connections::const_iterator IT;

  int idx=-1;
  if (depth-- >=0 ) {
    out <<"====== PixelFEDLink, ID: "<<id()<< endl;
    for (IT ic=theConnections.begin(); ic != theConnections.end(); ic++) {
      out <<"       "<<(*ic).name
          <<",r=("<< (*ic).rocs.first<<","<<(*ic).rocs.second<<")"
          <<"  ids:";
      int idx_tmp = idx;
      for (int i = (*ic).rocs.first; i <= (*ic).rocs.second; i++) {
        idx++;
        if (idx < numberOfROCs() ) out <<"_" << roc(idx)->idInLink();
      }
      out << endl;
      if (idx != numberOfROCs()) edm::LogError(" problem with ROC numbering!"); 
      idx = idx_tmp;
      for (int i = (*ic).rocs.first; i <= (*ic).rocs.second; i++) {
        idx++;
        if (idx < numberOfROCs()) out << roc(idx)->print(depth);
      }
    }
    out <<"       total number of ROCs: "<< numberOfROCs() << endl;
  }
  out << endl;
  return out.str();

}

ostream & operator<<(
    ostream& out, const PixelFEDLink & l)
{
/*  
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

*/
  return out;
}
