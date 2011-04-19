#ifndef DataFormats_FWLite_RunBase_h
#define DataFormats_FWLite_RunBase_h
// -*- C++ -*-
//
// Package:     DataFormats/FWLite
// Class  :     RunBase
//
/**\class RunBase RunBase.h DataFormats/FWLite/interface/RunBase.h

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:  Eric Vaandering
//         Created:  Wed Jan  13 15:01:20 EDT 2007
//
#if !defined(__CINT__) && !defined(__MAKECINT__)
// system include files
#include <string>
#include <typeinfo>
//
// // user include files
#include "FWCore/Common/interface/RunBase.h"

#include "Rtypes.h"

namespace fwlite
{
   class RunBase : public edm::RunBase
   {
      public:
         RunBase();

         virtual ~RunBase();

         virtual bool getByLabel(
                                  std::type_info const&,
                                  char const*,
                                  char const*,
                                  char const*,
                                  void*) const = 0;
         using edm::RunBase::getByLabel;

//          virtual std::string const getBranchNameFor (std::type_info const&,
//                                                      char const*,
//                                                      char const*,
//                                                      char const*) const = 0;

         virtual bool atEnd() const = 0;

         virtual const RunBase& operator++() = 0;

         virtual const RunBase& toBegin() = 0;

         virtual Long64_t fileIndex()          const { return -1; }
         virtual Long64_t secondaryFileIndex() const { return -1; }

      private:

         virtual edm::BasicHandle getByLabelImpl(edm::WrapperInterfaceBase const*, std::type_info const&, std::type_info const&, const edm::InputTag&) const;
   };
} // fwlite namespace

#endif /*__CINT__ */
#endif
