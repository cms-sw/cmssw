#ifndef FWCore_Framework_ComponentDescription_h
#define FWCore_Framework_ComponentDescription_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ComponentDescription
//
/**\struct edm::eventsetup::ComponentDescription

 Description: minimal set of information to describe an EventSetup component (ESSource or ESProducer)

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Dec 15 14:07:57 EST 2005
//

// user include files
#include "DataFormats/Provenance/interface/ParameterSetID.h"

// system include files
#include <string>

// forward declarations
namespace edm {
   namespace eventsetup {
      struct ComponentDescription {
         std::string label_; // A human friendly string that uniquely identifies the label
         std::string type_;  // A human friendly string that uniquely identifies the name
         bool isSource_;
         bool isLooper_;

         // ID of parameter set of the creator
         ParameterSetID pid_;

         /* ----------- end of provenance information ------------- */

         ComponentDescription() :
             label_(),
             type_(),
             isSource_(false),
             isLooper_(false),
             pid_() {}

         ComponentDescription(std::string const& iType,
                              std::string const& iLabel,
                              bool iIsSource,
                              bool iIsLooper = false) :
                                label_(iLabel),
                                type_(iType),
                                isSource_(iIsSource),
                                isLooper_(iIsLooper),
                                pid_() {}

         bool operator<(ComponentDescription const& iRHS) const {
            return (type_ == iRHS.type_) ? (label_ < iRHS.label_) : (type_<iRHS.type_);
         }
         bool operator==(ComponentDescription const& iRHS) const {
            return label_ == iRHS.label_ &&
            type_ == iRHS.type_ &&
            isSource_ == iRHS.isSource_;
         }
      };
   }
}
#endif
