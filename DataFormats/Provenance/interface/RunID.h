#ifndef DataFormats_Provenance_RunID_h
#define DataFormats_Provenance_RunID_h
// -*- C++ -*-
//
// Package:     DataFormats/Provenance
// Class  :     RunID
// 
/**\class RunID RunID.h DataFormats/Provenance/interface/RunID.h

 Description: Holds run and luminosityBlock number.

 Usage:
    <usage>

*/
//
// $Id: RunID.h,v 1.2 2007/06/14 03:38:30 wmtan Exp $
//

// system include files
#include <iosfwd>

// user include files

// forward declarations
namespace edm {

   typedef unsigned int RunNumber_t;

   
class RunID
{

   public:
   
   
      RunID() : run_(0) {}
      explicit RunID(RunNumber_t iRun) :
	run_(iRun) {}
      
      //virtual ~RunID();

      // ---------- const member functions ---------------------
      RunNumber_t run() const { return run_; }
   
      //moving from one RunID to another one
      RunID next() const {
         return RunID(run_+1);
      }
      RunID previous() const {
         if(run_ != 0) {
            return RunID(run_-1);
         }
         return RunID(0);
      }
      
      bool operator==(RunID const& iRHS) const {
         return iRHS.run_ == run_;
      }
      bool operator!=(RunID const& iRHS) const {
         return !(*this == iRHS);
      }
      
      bool operator<(RunID const& iRHS) const {
         return run_ < iRHS.run_;
      }
      bool operator<=(RunID const& iRHS) const {
         return run_ <= iRHS.run_;
      }
      bool operator>(RunID const& iRHS) const {
         return run_ > iRHS.run_;
      }
      bool operator>=(RunID const& iRHS) const {
         return run_ >= iRHS.run_;
      }
      // ---------- static functions ---------------------------

      static RunNumber_t maxRunNumber() {
         return 0xFFFFFFFFU;
      }
   
      static RunID firstValidRun() {
         return RunID(1);
      }
      // ---------- member functions ---------------------------
   
   private:
      //RunID(RunID const&); // stop default

      //RunID const& operator=(RunID const&); // stop default

      // ---------- member data --------------------------------
      RunNumber_t run_;
};

std::ostream& operator<<(std::ostream& oStream, RunID const& iID);

}
#endif
