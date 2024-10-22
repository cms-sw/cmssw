#ifndef DataFormats_GEMDigi_GEMAMC13Status_h
#define DataFormats_GEMDigi_GEMAMC13Status_h
#include "GEMAMC13.h"
#include "GEMAMC.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include <bitset>
#include <ostream>

class GEMAMC13Status {
public:
  union Errors {
    uint8_t codes;
    struct {
      uint8_t InValidSize : 1;
      uint8_t failTrailerCheck : 1;
      uint8_t failFragmentLength : 1;
      uint8_t failTrailerMatch : 1;
      uint8_t moreTrailers : 1;
      uint8_t crcModified : 1;
      uint8_t slinkError : 1;
      uint8_t wrongFedId : 1;
    };
  };
  union Warnings {
    uint8_t wcodes;
    struct {
      uint8_t InValidAMC : 1;  // error for AMC but cant display when not found.
    };
  };

  GEMAMC13Status() {}
  GEMAMC13Status(const FEDRawData& fedData) {
    Errors error{0};
    if ((fedData.size() / sizeof(uint64_t)) < 5) {
      error.InValidSize = 1;
    } else {
      FEDTrailer trailer(fedData.data() + fedData.size() - FEDTrailer::length);
      error.failTrailerCheck = !trailer.check();
      error.failFragmentLength = (trailer.fragmentLength() * sizeof(uint64_t) != fedData.size());
      error.moreTrailers = trailer.moreTrailers();
      error.crcModified = trailer.crcModified();
      error.slinkError = trailer.slinkError();
      error.wrongFedId = trailer.wrongFedId();
    }
    errors_ = error.codes;

    Warnings warn{0};
    warnings_ = warn.wcodes;
  }
  void inValidAMC() {
    Warnings warn{warnings_};
    warn.InValidAMC = 1;
    warnings_ = warn.wcodes;
  }

  bool isBad() const { return errors_ != 0; }
  uint8_t errors() const { return errors_; }
  uint8_t warnings() const { return warnings_; }

private:
  uint8_t errors_;
  uint8_t warnings_;
};

inline std::ostream& operator<<(std::ostream& out, const GEMAMC13Status& status) {
  out << "GEMAMC13Status errors " << std::bitset<8>(status.errors()) << " warnings "
      << std::bitset<8>(status.warnings());
  return out;
}

#endif
