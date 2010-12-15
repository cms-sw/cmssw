#ifndef DataFormats_Provenance_LuminosityBlockRange_h
#define DataFormats_Provenance_LuminosityBlockRange_h
// -*- C++ -*-
//
// Package:     DataFormats/Provenance
// Class  :     LuminosityBlockRange
//
/**\class LuminosityBlockRange LuminosityBlockRange.h DataFormats/Provenance/interface/LuminosityBlockRange.h

 Description: Holds run and luminosityBlock range.

 Usage:
    <usage>

*/
//
//

// system include files
#include <functional>
#include <iosfwd>
#include <vector>

// user include files
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

// forward declarations
namespace edm {

//   typedef unsigned int LuminosityBlockNumber_t;

  class LuminosityBlockRange {
    public:
      LuminosityBlockRange();

      LuminosityBlockRange(RunNumber_t startRun, LuminosityBlockNumber_t startLuminosityBlock,
                           RunNumber_t endRun,   LuminosityBlockNumber_t endLuminosityBlock);

      LuminosityBlockRange(LuminosityBlockID const& begin, LuminosityBlockID const& end);

      //virtual ~LuminosityBlockID();

      // ---------- const member functions ---------------------
      LuminosityBlockID     startLumiID() const {return startLumiID_;}
      LuminosityBlockID       endLumiID() const {return endLumiID_;}
      RunNumber_t              startRun() const {return startLumiID_.run();}
      RunNumber_t                endRun() const {return endLumiID_.run();}
      LuminosityBlockNumber_t startLumi() const {return startLumiID_.luminosityBlock();}
      LuminosityBlockNumber_t   endLumi() const {return endLumiID_.luminosityBlock();}

      // ---------- static functions ---------------------------

      // ---------- member functions ---------------------------

    private:
      // ---------- member data --------------------------------
      //RunNumber_t             startRun_;
      //RunNumber_t             endRun_;
      //LuminosityBlockNumber_t startLumi_;
      //LuminosityBlockNumber_t endLumi_;
      LuminosityBlockID       startLumiID_;
      LuminosityBlockID       endLumiID_;
  };

  std::ostream& operator<<(std::ostream& oStream, LuminosityBlockRange const& iID);
  bool contains(LuminosityBlockRange const& lh, LuminosityBlockID const& rh);
  bool contains(LuminosityBlockRange const& lh, LuminosityBlockRange const& rh);
  bool lessThan(LuminosityBlockRange const& lh, LuminosityBlockRange const& rh);
  bool overlaps(LuminosityBlockRange const& lh, LuminosityBlockRange const& rh);
  bool distinct(LuminosityBlockRange const& lh, LuminosityBlockRange const& rh);
  bool merge(LuminosityBlockRange& lh, LuminosityBlockRange& rh);
  std::vector<LuminosityBlockRange>& sortAndRemoveOverlaps(std::vector<LuminosityBlockRange>& lumiRange);
}
#endif

