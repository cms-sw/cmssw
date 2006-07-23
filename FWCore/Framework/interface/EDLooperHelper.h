#if !defined(FWCORE_FRAMEWORK_EDLOOPERHELPER_H)
#define FWCORE_FRAMEWORK_EDLOOPERHELPER_H
// -*- C++ -*-
//
// Package:     <package>
// Module:      EDLooperHelper
// 
/**\class EDLooperHelper EDLooperHelper.h package/EDLooperHelper.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Valentin Kuznetsov
// Created:     Wed Jul 12 11:26:26 EDT 2006
// $Id$
//
// Revision history
//
// $Log$

// system include files

// user include files
#include "FWCore/Framework/interface/EventHelperDescription.h"
#include "FWCore/Framework/interface/EventProcessor.h"

// forward declarations

namespace edm {
class EventProcessor;

namespace eventsetup {
class EventSetupRecordKey;
}

class EDLooperHelper
{
      // ---------- friend classes and functions ---------------
      friend class edm::EventProcessor;

   public:
      // ---------- constants, enums and typedefs --------------

      // ---------- Constructors and destructor ----------------
      virtual ~EDLooperHelper();

      // ---------- member functions ---------------------------
      EventHelperDescription runOnce(unsigned long numberToProcess);
      void rewind(const std::set<edm::eventsetup::EventSetupRecordKey>& keys);

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

   protected:
      // ---------- protected member functions -----------------

      // ---------- protected const member functions -----------

   private:
      // ---------- Constructors and destructor ----------------
      EDLooperHelper(EventProcessor* p) : eventProcessor_(p) {}
      EDLooperHelper( const EDLooperHelper& ); // stop default

      // ---------- assignment operator(s) ---------------------
      const EDLooperHelper& operator=( const EDLooperHelper& ); // stop default

      // ---------- private member functions -------------------

      // ---------- private const member functions -------------

      // ---------- data members -------------------------------
      EventProcessor* eventProcessor_;

      // ---------- static data members ------------------------

};

// inline function definitions

} // end of namespace

#endif /* FWCORE_FRAMEWORK_EDLOOPERHELPER_H */
