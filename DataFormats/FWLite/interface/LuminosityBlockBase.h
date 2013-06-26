#ifndef DataFormats_FWLite_LuminosityBlockBase_h
#define DataFormats_FWLite_LuminosityBlockBase_h
// -*- C++ -*-
//
// Package:     DataFormats/FWLite
// Class  :     LuminosityBlockBase
//
/**\class LuminosityBlockBase LuminosityBlockBase.h DataFormats/FWLite/interface/LuminosityBlockBase.h

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
#include "FWCore/Common/interface/LuminosityBlockBase.h"

#include "Rtypes.h"

namespace fwlite
{
   class LuminosityBlockBase : public edm::LuminosityBlockBase
   {
      public:
         LuminosityBlockBase();

         virtual ~LuminosityBlockBase();

         virtual bool getByLabel(
                                  std::type_info const&,
                                  char const*,
                                  char const*,
                                  char const*,
                                  void*) const = 0;
         virtual bool getByLabel(
                                  std::type_info const&,
                                  char const*,
                                  char const*,
                                  char const*,
                                  edm::WrapperHolder&) const = 0;

         using edm::LuminosityBlockBase::getByLabel;

         virtual bool atEnd() const = 0;

         virtual const LuminosityBlockBase& operator++() = 0;

         virtual const LuminosityBlockBase& toBegin() = 0;

         virtual Long64_t fileIndex()          const { return -1; }
         virtual Long64_t secondaryFileIndex() const { return -1; }

      private:

         virtual edm::BasicHandle getByLabelImpl(std::type_info const&, std::type_info const&, const edm::InputTag&) const;
   };
} // fwlite namespace

#endif /*__CINT__ */
#endif
