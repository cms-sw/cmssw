#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <sstream>
#include <iostream>

using namespace std;
using namespace sipixelobjects;

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

void PixelFEDLink::add(const ROCs & rocs)
{
  theROCs.insert( theROCs.end(), rocs.begin(), rocs.end() );
}

string PixelFEDLink::print(int depth) const
{
  ostringstream out;

  if (depth-- >=0 ) {
    out <<"====== PixelFEDLink, ID: "<<id()<< endl;
    typedef ROCs::const_iterator CIR;
    for (CIR ir = theROCs.begin(); ir != theROCs.end(); ir++) out<< (ir)->print(depth); 
    out <<"       total number of ROCs: "<< numberOfROCs() << endl;
  }
  out << endl;
  return out.str();

}

