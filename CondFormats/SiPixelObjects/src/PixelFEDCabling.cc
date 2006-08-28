#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include <sstream>
using namespace std;

void PixelFEDCabling::setLinks(Links & links) 
{
  theLinks = links;
}

void PixelFEDCabling::addLink(const PixelFEDLink & link)
{
  theLinks.push_back(link);
}

bool PixelFEDCabling::checkLinkNumbering() const
{
  bool result = true;
  typedef Links::const_iterator IL;
  int idx_expected = -1;
  for (IL il = theLinks.begin(); il != theLinks.end(); il++) {
    idx_expected++;
    if (idx_expected != (*il).id() ) {
      result = false;
      cout << " ** PixelFEDCabling ** link numbering inconsistency, expected id: "
           << idx_expected <<" has: " << (*il).id() << endl;
    } 
    if (! (*il).checkRocNumbering() ) {
      result = false;
      cout << "** PixelFEDCabling ** inconsistent ROC numbering in link id: "
           << (*il).id() << endl;
    }
  }
  return result;
}

string PixelFEDCabling::print(int depth) const
{
  ostringstream out;
  typedef vector<PixelFEDLink>::const_iterator IT; 
  if (depth-- >=0 ) {
    out <<"==== PixelFED, ID: "<<id()<< endl;
    for (IT it=theLinks.begin(); it != theLinks.end(); it++)
         out << (*it).print(depth);
    out <<"     total number of Links: "<< numberOfLinks() << endl;
  }
  out << endl;
  return out.str();
}

