#include "DataFormats/CSCDigi/interface/CSCShowerDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <iostream>

using namespace std;

/// Constructors
CSCShowerDigi::CSCShowerDigi(const uint16_t bitsInTime,
                             const uint16_t bitsOutOfTime,
                             const uint16_t cscID,
                             const uint16_t bx,
                             const uint16_t showerType,
                             const uint16_t wireNHits,
                             const uint16_t compNHits)
    : bitsInTime_(bitsInTime),
      bitsOutOfTime_(bitsOutOfTime),
      cscID_(cscID),
      bx_(bx),
      showerType_(showerType),
      wireNHits_(wireNHits),
      comparatorNHits_(compNHits) {}

/// Default
CSCShowerDigi::CSCShowerDigi()
    : bitsInTime_(0), bitsOutOfTime_(0), cscID_(0), bx_(0), showerType_(0), wireNHits_(0), comparatorNHits_(0) {}

void CSCShowerDigi::clear() {
  bitsInTime_ = 0;
  bitsOutOfTime_ = 0;
  cscID_ = 0;
  bx_ = 0;
  showerType_ = 0;
  wireNHits_ = 0;
  comparatorNHits_ = 0;
}

bool CSCShowerDigi::isValid() const {
  // any loose shower is valid
  // isLooseOutofTime() is removed as out-of-time shower is not used for Run3
  return isLooseInTime() and isValidShowerType();
}

bool CSCShowerDigi::isLooseInTime() const { return bitsInTime() >= kLoose; }

bool CSCShowerDigi::isNominalInTime() const { return bitsInTime() >= kNominal; }

bool CSCShowerDigi::isTightInTime() const { return bitsInTime() >= kTight; }

bool CSCShowerDigi::isLooseOutOfTime() const { return bitsOutOfTime() >= kLoose; }

bool CSCShowerDigi::isNominalOutOfTime() const { return bitsOutOfTime() >= kNominal; }

bool CSCShowerDigi::isTightOutOfTime() const { return bitsOutOfTime() >= kTight; }

bool CSCShowerDigi::isValidShowerType() const { return showerType_ > kInvalidShower; }

std::ostream& operator<<(std::ostream& o, const CSCShowerDigi& digi) {
  unsigned int showerType = digi.getShowerType();
  std::string compHitsStr(", comparatorHits ");
  compHitsStr += std::to_string(digi.getComparatorNHits());
  std::string wireHitsStr(", wireHits ");
  wireHitsStr += std::to_string(digi.getWireNHits());
  std::string showerStr;
  switch (showerType) {
    case 0:
      showerStr = "Invalid ShowerType";
      break;
    case 1:
      showerStr = "AnodeShower";
      break;
    case 2:
      showerStr = "CathodeShower";
      break;
    case 3:
      showerStr = "MatchedShower";
      break;
    case 4:
      showerStr = "EMTFShower";
      break;
    case 5:
      showerStr = "GMTShower";
      break;
    default:
      showerStr = "UnknownShowerType";
  }

  return o << showerStr << ": bx " << digi.getBX() << ", in-time bits " << digi.bitsInTime() << ", out-of-time bits "
           << digi.bitsOutOfTime() << ((showerType == 1 or showerType == 3) ? wireHitsStr : "")
           << ((showerType == 2 or showerType == 3) ? compHitsStr : "") << ";";
}
