#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/DetId/interface/DetId.h"


#include <sstream>

using namespace std;
using namespace sipixelobjects;

void PixelFEDLink::addItem(const PixelROC & roc)
{
  // INFO roc numbering vs vector has offset=1
  if(roc.idInLink() > theROCs.size() ) theROCs.resize(roc.idInLink());
  theROCs[roc.idInLink()-1] = roc;
}

bool PixelFEDLink::checkRocNumbering() const
{
  bool result = true;
  unsigned int idx_expected = 0;
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
  // if (id() < 0) return  out.str(); // id() >= 0, since it returns an unsigned

  if (depth-- >=0 ) {
    if(id()<10) out <<"  LNK:  "<<id(); else  out <<"  LNK: "<<id();
    if (depth==0) out << printForMap();
    else { 
      out << endl;
      typedef ROCs::const_iterator CIR;
      for (CIR ir = theROCs.begin(); ir != theROCs.end(); ir++) out<< (ir)->print(depth); 
      out <<"#  total number of ROCs: "<< numberOfROCs() << endl;
    }
  }
  return out.str();

}

string PixelFEDLink::printForMap() const 
{
  typedef ROCs::const_iterator CIR;
  ostringstream out;

// barrel
{
  int minroc = 9999;
  int maxroc = -1;
  bool first = true;
  PixelBarrelName prev;
  for (CIR ir = theROCs.begin(); ir < theROCs.end(); ir++) {
    DetId detid = DetId(ir->rawId());
    bool barrel = PixelModuleName::isBarrel(detid.rawId()); if (!barrel) continue;
    PixelBarrelName curr( detid);
    if (first) prev = curr; 

    int idRoc = ir->idInDetUnit();
    if (curr==prev) {
      minroc = min(idRoc, minroc);
      maxroc = max(idRoc, maxroc);
    }

    if ( !(curr==prev) ) {
    out <<"    MOD: "<< prev.name() <<" ROC: "<< minroc<<", "<<maxroc<< std::endl;
      prev = curr;
      maxroc = minroc = idRoc;
 //     minroc = idRoc;
 //     maxroc = idRoc;
    }

    if ( ir==theROCs.end()-1) {
    out <<"    MOD: "<< curr.name() <<" ROC: "<< minroc<<", "<<maxroc<< std::endl;
    }
  }
}

// same for endcpap
{
  bool first = true;
  PixelEndcapName prev;
  for (CIR ir = theROCs.begin(); ir < theROCs.end(); ir++) {
    DetId detid = DetId(ir->rawId());
    bool barrel = PixelModuleName::isBarrel(detid.rawId()); if (barrel) continue;
    PixelEndcapName tmp( detid);
    PixelEndcapName curr( tmp.halfCylinder(), tmp.diskName(), tmp.bladeName(), tmp.pannelName() );
    if (first) prev = curr;
    if ( !(curr==prev) ) out <<"    MOD: "<< prev.name() << std::endl;
    if ( ir==theROCs.end()-1) out <<"    MOD: "<< curr.name() << std::endl;
  }
}
  return out.str();
}

