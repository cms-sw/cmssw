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

// user include files
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

// forward declarations
namespace edm {

//   typedef unsigned int LuminosityBlockNumber_t;


class LuminosityBlockRange
{

   public:


      LuminosityBlockRange():
        startRun_(0), endRun_(0), 
        startLumi_(edm::LuminosityBlockID::maxLuminosityBlockNumber()),
        endLumi_(edm::LuminosityBlockID::maxLuminosityBlockNumber()),
        startLumiID_(startRun_,startLumi_),
        endLumiID_(endRun_,endLumi_){
      }

      LuminosityBlockRange(RunNumber_t startRun, LuminosityBlockNumber_t startLuminosityBlock,
                           RunNumber_t endRun,   LuminosityBlockNumber_t endLuminosityBlock) :
	startRun_(startRun), endRun_(endRun),  
        // Special cases since 0 means maximum
        startLumi_(startLuminosityBlock!=0? startLuminosityBlock : edm::LuminosityBlockID::maxLuminosityBlockNumber()), 
        endLumi_(endLuminosityBlock!=0? endLuminosityBlock : edm::LuminosityBlockID::maxLuminosityBlockNumber()),
        startLumiID_(startRun_,startLumi_),
        endLumiID_(endRun_,endLumi_) {
      }

      //virtual ~LuminosityBlockID();

      // ---------- const member functions ---------------------
      LuminosityBlockID     startLumiID() const {return startLumiID_; }
      LuminosityBlockID       endLumiID() const {return endLumiID_; }
      RunNumber_t              startRun() const {return startRun_; }
      RunNumber_t                endRun() const {return endRun_; }
      LuminosityBlockNumber_t startLumi() const {return startLumi_; }
      LuminosityBlockNumber_t   endLumi() const {return endLumi_; }

      // ---------- static functions ---------------------------

      // ---------- member functions ---------------------------

   private:

      // ---------- member data --------------------------------
      RunNumber_t             startRun_;
      RunNumber_t             endRun_;
      LuminosityBlockNumber_t startLumi_;
      LuminosityBlockNumber_t endLumi_;
      LuminosityBlockID       startLumiID_,endLumiID_;
};

std::ostream& operator<<(std::ostream& oStream, LuminosityBlockRange const& iID);
bool contains(LuminosityBlockRange const& lh, LuminosityBlockID const& rh);
bool contains(LuminosityBlockRange const& lh, LuminosityBlockRange const& rh);
bool overlaps(LuminosityBlockRange const& lh, LuminosityBlockRange const& rh);
bool distinct(LuminosityBlockRange const& lh, LuminosityBlockRange const& rh);

}



#endif
