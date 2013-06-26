#include "Geometry/MuonNumbering/interface/MuonSimHitNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/CSCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonSubDetector.h"

//#define LOCAL_DEBUG

MuonSimHitNumberingScheme::MuonSimHitNumberingScheme(MuonSubDetector* d, const DDCompactView& cpv) {
  theDetector=d;
  if (theDetector->isBarrel()) {
    theNumbering=new DTNumberingScheme(cpv);
  } else if (theDetector->isEndcap()) {
    theNumbering=new CSCNumberingScheme(cpv);
  } else if (theDetector->isRpc()) {
    theNumbering=new RPCNumberingScheme(cpv);
  } else if (theDetector->isGem()) {
    theNumbering=new GEMNumberingScheme(cpv);
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

