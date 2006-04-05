#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

using namespace std;
PixelFEDCabling::PixelFEDCabling(int id,  ModuleNames & names) 
  : theFedId(id),  theModuleNames(names) 
{ }

PixelFEDCabling::~PixelFEDCabling()
{
  cout << "PixelFEDCabling DTOR" << endl;
  for (ModuleNames::const_iterator it=theModuleNames.begin();
       it != theModuleNames.end(); it++) delete (*it);
  clearLinks( false );
}

void PixelFEDCabling::setLinks(Links & links) 
{
  theLinks = links;
//  cout << " ** PixelFEDCabling ("<<this->id()<<") links dump: " << endl;
//  typedef Links::const_iterator CIT;
//  for (CIT it = theLinks.begin(); it != theLinks.end(); it++) 
//      cout << (**it) << endl;
  if( !checkLinkNumbering() ) clearLinks();
}

bool PixelFEDCabling::checkLinkNumbering() const
{
  bool result = true;
  typedef Links::const_iterator IL;
  int idx_expected = -1;
  for (IL il = theLinks.begin(); il != theLinks.end(); il++) {
    idx_expected++;
    if (idx_expected != (*il)->id() ) {
      result = false;
      cout << " ** PixelFEDCabling ** link numbering inconsistency, expected id: "
           << idx_expected <<" has: " << (*il)->id() << endl;
    } 
    if (! (*il)->checkRocNumbering() ) {
      result = false;
      cout << "** PixelFEDCabling ** inconsistent ROC numbering in link id: "
           << (*il)->id() << endl;
    }
  }
  return result;
}

void PixelFEDCabling::clearLinks( bool warn)
{
  if (warn) cout << "** PixelFEDCabling, clear links" << endl; 
  for (Links::const_iterator it = theLinks.begin();
      it != theLinks.end(); it++) delete (*it);
   theLinks.clear();
}
