#ifndef EVENTSETUP_VALIDITYINTERVAL_H
#define EVENTSETUP_VALIDITYINTERVAL_H
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ValidityInterval
// 
/**\class ValidityInterval ValidityInterval.h FWCore/Framework/interface/ValidityInterval.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Tue Mar 29 14:47:25 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/Timestamp.h"

// forward declarations
namespace edm {
class ValidityInterval
{

   public:
      ValidityInterval();
      ValidityInterval(const Timestamp& iFirst,
                       const Timestamp& iLast);
      //virtual ~ValidityInterval();

      // ---------- const member functions ---------------------
      bool validFor(const Timestamp&) const;
      
      const Timestamp& first() const { return first_; }
      const Timestamp& last() const { return last_; }
      
      bool operator==(const ValidityInterval& iRHS) const {
         return iRHS.first_ == first_ && 
         iRHS.last_ == last_ ;
      }
      bool operator!=(const ValidityInterval& iRHS) const {
         return ! (*this == iRHS);
      }
      
      // ---------- static member functions --------------------
      static const ValidityInterval& invalidInterval();
      
      // ---------- member functions ---------------------------
      void setFirst(const Timestamp& iTime) {
         first_ = iTime;
      }
      void setLast(const Timestamp& iTime) {
         last_ = iTime;
      }
      
   private:
      //ValidityInterval(const ValidityInterval&); // stop default

      //const ValidityInterval& operator=(const ValidityInterval&); // stop default

      // ---------- member data --------------------------------
      Timestamp first_;
      Timestamp last_;
};

}
#endif /* EVENTSETUP_VALIDITYINTERVAL_H */
