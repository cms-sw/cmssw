#include "DetectorDescription/DDCMS/interface/MuonNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

using namespace cms;
using namespace std;
using namespace edm;

const MuonBaseNumber
MuonNumbering::geoHistoryToBaseNumber(const DDGeoHistory &history, MuonConstants& values) const {
  MuonBaseNumber num;

  int levelPart = values["level"];
  int superPart = values["super"];
  int basePart = values["base"];
  int startCopyNo = values["xml_starts_with_copyno"];

  // some consistency checks
  if(basePart != 1) {
    LogError("Geometry") << "MuonNumbering finds unusual base constant:"
			 << basePart;
  }
  if(superPart < 100) {
    LogError("Geometry") << "MuonNumbering finds unusual super constant:"
			 << superPart;
  }
  if(levelPart < 10*superPart) {
    LogError("Geometry") << "MuonNumbering finds unusual level constant:"
			 << levelPart;
  }
  if((startCopyNo !=0 ) && (startCopyNo != 1)) {
    LogError("Geometry") << "MuonNumbering finds unusual start value for copy numbers:"
			 << startCopyNo;
  }
  
  for(auto const& it : history) {
    int tag = it.specpar.dblValue("CopyNoTag")/levelPart;
    if(tag > 0) {
      int offset = it.specpar.dblValue("CopyNoOffset");
      int copyno = it.copyNo + offset%superPart;
      int super = offset/superPart;
      num.addBase(tag, super, copyno - startCopyNo);
    }
  }
  return num;
}
