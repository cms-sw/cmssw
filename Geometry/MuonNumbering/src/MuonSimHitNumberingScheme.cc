#include "Geometry/MuonNumbering/interface/MuonSimHitNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/CSCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/ME0NumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonSubDetector.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

MuonSimHitNumberingScheme::MuonSimHitNumberingScheme(MuonSubDetector* d, const DDCompactView& cpv) :
  MuonSimHitNumberingScheme(d,MuonDDDConstants(cpv)){ }

MuonSimHitNumberingScheme::MuonSimHitNumberingScheme(MuonSubDetector* d, const MuonDDDConstants& muonConstants) {
  theDetector=d;
  if (theDetector->isBarrel()) {
    theNumbering=new DTNumberingScheme(muonConstants);
  } else if (theDetector->isEndcap()) {
    theNumbering=new CSCNumberingScheme(muonConstants);
  } else if (theDetector->isRPC()) {
    theNumbering=new RPCNumberingScheme(muonConstants);
  } else if (theDetector->isGEM()) {
    theNumbering=new GEMNumberingScheme(muonConstants);
  } else if (theDetector->isME0()) {
    theNumbering=new ME0NumberingScheme(muonConstants);
  } 
}


MuonSimHitNumberingScheme::~MuonSimHitNumberingScheme() {
  delete theNumbering;
}

int MuonSimHitNumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber& num) {
  if (theNumbering) {
    return theNumbering->baseNumberToUnitNumber(num);
  } else {
    return 0;
  }
}

