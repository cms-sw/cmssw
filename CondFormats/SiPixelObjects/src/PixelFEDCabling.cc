#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include <sstream>
using namespace std;
using namespace sipixelobjects;

void PixelFEDCabling::setLinks(Links & links) 
{
  theLinks = links;
}

void PixelFEDCabling::addLink(const PixelFEDLink & link)
{
  if ( link.id() < 1) return;
  if ( theLinks.size() < link.id() ) theLinks.resize(link.id());
  theLinks[link.id()-1] = link;
}

void PixelFEDCabling::addItem(unsigned int linkId, const PixelROC & roc)
{
  if ( linkId < 1) return;
  if ( theLinks.size() < linkId ) theLinks.resize(linkId);
  if ( theLinks[linkId-1].id() != linkId)  theLinks[linkId-1] = PixelFEDLink(linkId);
  theLinks[linkId-1].addItem(roc);
}

bool PixelFEDCabling::checkLinkNumbering() const
{
  bool result = true;
  typedef Links::const_iterator IL;
  unsigned int idx_expected = 0;
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
    out <<"FED: "<<id()<< endl;
    for (IT it=theLinks.begin(); it != theLinks.end(); it++)
         out << (*it).print(depth);
    out <<"# total number of Links: "<< numberOfLinks() << endl;
  }
  out << endl;
  return out.str();
}

