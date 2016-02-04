#ifndef DataFormats_FWLite_EventBase_h
#define DataFormats_FWLite_EventBase_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     EventBase
//
/**\class EventBase EventBase.h DataFormats/FWLite/interface/EventBase.h

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:  Charles Plager
//         Created:  Tue May  8 15:01:20 EDT 2007
//
#if !defined(__CINT__) && !defined(__MAKECINT__)
// system include files
#include <string>
#include <typeinfo>
//
// // user include files
#include "FWCore/Common/interface/EventBase.h"

#include "Rtypes.h"

namespace fwlite
{
   class EventBase : public edm::EventBase
   {
      public:
         EventBase();

         virtual ~EventBase();

         virtual bool getByLabel (const std::type_info&,
                                  const char*,
                                  const char*,
                                  const char*,
                                  void*) const = 0;
         using edm::EventBase::getByLabel;

         virtual const std::string getBranchNameFor (const std::type_info&,
                                                     const char*,
                                                     const char*,
                                                     const char*) const = 0;

         virtual bool atEnd() const = 0;

         virtual const EventBase& operator++() = 0;

         virtual const EventBase& toBegin() = 0;

         virtual Long64_t fileIndex()          const { return -1; }
         virtual Long64_t secondaryFileIndex() const { return -1; }

      private:

         virtual edm::BasicHandle getByLabelImpl(const std::type_info&, const std::type_info&, const edm::InputTag&) const;
   };
} // fwlite namespace

#endif /*__CINT__ */
#endif
