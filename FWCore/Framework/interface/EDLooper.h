#ifndef FWCore_Framework_EDLooper_h
#define FWCore_Framework_EDLooper_h
// -*- C++ -*-
//
// Package:     Framework
// Module:      EDLooper
// 
/**\class EDLooper EDLooper.h package/EDLooper.h

 Description: Base class for all looping components

 Usage:
    <usage>

*/
//
// Author:      Valentin Kuznetsov
// Created:     Wed Jul  5 11:42:17 EDT 2006
// $Id: EDLooper.h,v 1.4 2007/03/04 06:00:22 wmtan Exp $
//
// Revision history
//
// $Log: EDLooper.h,v $
// Revision 1.4  2007/03/04 06:00:22  wmtan
// Move Provenance classes to DataFormats/Provenance
//
// Revision 1.3  2006/12/19 00:28:17  wmtan
// changed (u)long to (u)int so that data is the same size on 32 and 64 bit machines
//
// Revision 1.2  2006/07/28 13:24:34  valya
// Modified endOfLoop, now it accepts counter as a second argument. Add EDLooper calls to beginOfJob/endOfJob in EventProcessor
//
// Revision 1.1  2006/07/23 01:24:33  valya
// Add looper support into framework. The base class is EDLooper. All the work done in EventProcessor and EventHelperLooper
//

// system include files
#include <string>
#include <set>

// user include files
#include "DataFormats/Provenance/interface/PassID.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDLooperHelper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations

namespace edm {
  namespace eventsetup {
    class EventSetupRecordKey;
  }
class EDLooper
{
      // ---------- friend classes and functions ---------------

   public:
      // ---------- constants, enums and typedefs --------------
      enum Status {kContinue, kStop};

      // ---------- Constructors and destructor ----------------
      EDLooper();
      virtual ~EDLooper();

      // ---------- member functions ---------------------------
      virtual void beginOfJob(const edm::EventSetup&); 
      virtual void startingNewLoop(unsigned int ) = 0; 
      virtual Status duringLoop(const edm::Event&, const edm::EventSetup&) = 0; 
      virtual Status endOfLoop(const edm::EventSetup&, unsigned int iCounter) = 0; 
      virtual void endOfJob();
      void loop(EDLooperHelper& iHelper, unsigned int numberToProcess); 
      void setLooperName(const std::string& name) {name_=name;};
      void setLooperPassID(const PassID& id) {passID_=id; processID_=passID_; }
      PassID getLooperPassID() {return passID_;}
      std::string getLooperName() {return name_;}

      // ---------- const member functions ---------------------
      virtual std::set<eventsetup::EventSetupRecordKey> modifyingRecords() const;

      // ---------- static member functions --------------------

   protected:
      // ---------- protected member functions -----------------

      // ---------- protected const member functions -----------

   private:
      // ---------- Constructors and destructor ----------------
      EDLooper( const EDLooper& ); // stop default

      // ---------- assignment operator(s) ---------------------
      const EDLooper& operator=( const EDLooper& ); // stop default

      // ---------- private member functions -------------------

      // ---------- private const member functions -------------

      // ---------- data members -------------------------------
      std::string name_;
      PassID passID_, processID_;

      // ---------- static data members ------------------------

};

// inline function definitions

}

#endif /* FWCORE_FRAMEWORK_EDLOOPER_H */
