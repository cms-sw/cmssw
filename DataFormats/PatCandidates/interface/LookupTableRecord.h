#ifndef DataFormats_PatCandidates_interface_LookupTableRecord_h
#define DataFormats_PatCandidates_interface_LookupTableRecord_h

/** \class    pat::LookupTableRecord LookupTableRecord.h "DataFormats/PatCandidates/interface/LookupTableRecord.h"
 *
 *  \brief    Class to store the result of a lookup table fetch, e.g. for efficiencies. 
 *
 *  Stores a value, an uncertainty and a bin index (needed to take into account correlations from multiple lookups) 
 *
 *  \author   Giovanni Petrucciani
 *
 *  \version  $Id: LookupTableRecord.h,v 1.1 2008/09/29 09:42:01 gpetrucc Exp $
 *
 */

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1DFloat.h"
#include <boost/cstdint.hpp>

namespace pat {
    class LookupTableRecord {
        public:
            LookupTableRecord() : value_(0), error_(0), bin_(0) {}
            LookupTableRecord(float value, float error, uint16_t bin=0) :
                value_(value), error_(error), bin_(bin) {}
            LookupTableRecord(float value, uint16_t bin=0) :
                value_(value), error_(0), bin_(bin) {}
            LookupTableRecord(const Measurement1DFloat &meas, uint16_t bin=0) :
                value_(meas.value()), error_(meas.error()), bin_(bin) {}

            // Get the stored value
            float    value() const { return value_; }
            // Get the uncertainty on the stored value (note: it CAN be ZERO)
            float    error() const { return error_; }
            // Get the bin of the table used to compute this value (if available, otherwise 0)
            uint16_t bin()   const { return bin_; }
        private:
            float    value_, error_;
            uint16_t bin_;
    };
}

#endif
