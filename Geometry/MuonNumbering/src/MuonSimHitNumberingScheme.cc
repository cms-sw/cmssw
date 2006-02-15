#include "Geometry/MuonNumbering/interface/MuonSimHitNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/CSCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonSubDetector.h"

//#define DEBUG

MuonSimHitNumberingScheme::MuonSimHitNumberingScheme(MuonSubDetector* d) {
  theDetector=d;
  if (theDetector->isBarrel()) {
    theNumbering=new DTNumberingScheme();
  } else if (theDetector->isEndcap()) {
    theNumbering=new CSCNumberingScheme();
  } else if (theDetector->isRpc()) {
    theNumbering=new RPCNumberingScheme();
  } 
}

MuonSimHitNumberingScheme::~MuonSimHitNumberingScheme() {
  delete theNumbering;
}

int MuonSimHitNumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber num) {
  if (theNumbering) {
    return theNumbering->baseNumberToUnitNumber(num);
  } else {
    return 0;
  }
}

