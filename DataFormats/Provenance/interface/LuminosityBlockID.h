#ifndef DataFormats_Provenance_LuminosityBlockID_h
#define DataFormats_Provenance_LuminosityBlockID_h
// -*- C++ -*-
//
// Package:     DataFormats/Provenance
// Class  :     LuminosityBlockID
//
/**\class edm::LuminosityBlockID

 Description: Holds run and luminosityBlock number.

 Usage:
    <usage>

*/
//
//

// system include files
#include <functional>
#include <iosfwd>
#include "boost/cstdint.hpp"

// user include files
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

// forward declarations
namespace edm {

  class LuminosityBlockID {
   public:
      LuminosityBlockID() : run_(invalidRunNumber), luminosityBlock_(invalidLuminosityBlockNumber) {}
      explicit LuminosityBlockID(boost::uint64_t id);
      LuminosityBlockID(RunNumber_t iRun, LuminosityBlockNumber_t iLuminosityBlock) :
        run_(iRun), luminosityBlock_(iLuminosityBlock) {}

      //virtual ~LuminosityBlockID();

      // ---------- const member functions ---------------------
      RunNumber_t run() const { return run_; }
      LuminosityBlockNumber_t luminosityBlock() const { return luminosityBlock_; }

      boost::uint64_t value() const;

      //moving from one LuminosityBlockID to another one
      LuminosityBlockID next() const {
         if(luminosityBlock_ != maxLuminosityBlockNumber()) {
            return LuminosityBlockID(run_, luminosityBlock_+1);
         }
         return LuminosityBlockID(run_+1, 1);
      }
      LuminosityBlockID nextRun() const {
         return LuminosityBlockID(run_+1, 0);
      }
      LuminosityBlockID nextRunFirstLuminosityBlock() const {
         return LuminosityBlockID(run_+1, 1);
      }
      LuminosityBlockID previousRunLastLuminosityBlock() const {
         if(run_ > 1) {
            return LuminosityBlockID(run_-1, maxLuminosityBlockNumber());
         }
         return LuminosityBlockID(0,0);
      }

      LuminosityBlockID previous() const {
         if(luminosityBlock_ > 1) {
            return LuminosityBlockID(run_, luminosityBlock_-1);
         }
         if(run_ != 0) {
            return LuminosityBlockID(run_ -1, maxLuminosityBlockNumber());
         }
         return LuminosityBlockID(0,0);
      }

      bool operator==(LuminosityBlockID const& iRHS) const {
         return iRHS.run_ == run_ && iRHS.luminosityBlock_ == luminosityBlock_;
      }
      bool operator!=(LuminosityBlockID const& iRHS) const {
         return ! (*this == iRHS);
      }

      bool operator<(LuminosityBlockID const& iRHS) const {
         return doOp<std::less>(iRHS);
      }
      bool operator<=(LuminosityBlockID const& iRHS) const {
         return doOp<std::less_equal>(iRHS);
      }
      bool operator>(LuminosityBlockID const& iRHS) const {
         return doOp<std::greater>(iRHS);
      }
      bool operator>=(LuminosityBlockID const& iRHS) const {
         return doOp<std::greater_equal>(iRHS);
      }

      // ---------- static functions ---------------------------

      static LuminosityBlockNumber_t maxLuminosityBlockNumber() {
         return 0xFFFFFFFFU;
      }

      static LuminosityBlockID firstValidLuminosityBlock() {
         return LuminosityBlockID(1, 1);
      }
      // ---------- member functions ---------------------------

   private:
      template<template <typename> class Op>
      bool doOp(LuminosityBlockID const& iRHS) const {
         //Run takes presidence for comparisions
         if(run_ == iRHS.run_) {
            Op<LuminosityBlockNumber_t> op_e;
            return op_e(luminosityBlock_, iRHS.luminosityBlock_);
         }
         Op<RunNumber_t> op;
         return op(run_, iRHS.run_) ;
      }
      //LuminosityBlockID(LuminosityBlockID const&); // stop default

      //LuminosityBlockID const& operator=(LuminosityBlockID const&); // stop default

      // ---------- member data --------------------------------
      RunNumber_t run_;
      LuminosityBlockNumber_t luminosityBlock_;
  };

  std::ostream& operator<<(std::ostream& oStream, LuminosityBlockID const& iID);

  inline
  LuminosityBlockID const& min(LuminosityBlockID const& lh, LuminosityBlockID const& rh) {
    return (rh < lh ? rh : lh);
  }

  inline
  LuminosityBlockID const& max(LuminosityBlockID const& lh, LuminosityBlockID const& rh) {
    return (rh < lh ? lh : rh);
  }

}
#endif
