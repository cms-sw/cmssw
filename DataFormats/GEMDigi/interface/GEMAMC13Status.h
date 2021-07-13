#ifndef DataFormats_GEMDigi_GEMAMC13Status_h
#define DataFormats_GEMDigi_GEMAMC13Status_h
#include "AMC13Event.h"
#include "AMCdata.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

namespace gem {

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

    GEMAMC13Status(const FEDRawData& fedData) {
      FEDTrailer trailer(fedData.data() + fedData.size() - FEDTrailer::length);
      Errors ferror{0};
      ferror.InValidSize = ( (fedData.size() / sizeof(uint64_t)) < 5);
      ferror.failTrailerCheck = trailer.check();
      ferror.failFragmentLength = (trailer.fragmentLength() * sizeof(uint64_t) != fedData.size());
      ferror.moreTrailers = trailer.moreTrailers();
      ferror.crcModified = trailer.crcModified();
      ferror.slinkError = trailer.slinkError();
      ferror.wrongFedId = trailer.wrongFedId();
      errors_ = ferror.codes;
    }

    bool isGood() { return errors_ == 0;}
    bool isBad() { return errors_ != 0;}
    uint16_t errors() { return errors_; }

    private:

    uint8_t errors_;

    };
}
#endif
