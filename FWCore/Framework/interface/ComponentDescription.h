#ifndef Framework_ComponentDescription_h
#define Framework_ComponentDescription_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ComponentDescription
// 
/**\class ComponentDescription ComponentDescription.h FWCore/Framework/interface/ComponentDescription.h

 Description: minimal set of information to describe an EventSetup component (ESSource or ESProducer)

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Dec 15 14:07:57 EST 2005
// $Id$
//

// system include files
#include <string>

// user include files

// forward declarations
namespace edm {
   namespace eventsetup {
      struct ComponentDescription {
         std::string label_;
         std::string type_;
         bool isSource_;
         ComponentDescription() : isSource_(false){}
         ComponentDescription(const std::string& iType,
                              const std::string& iLabel,
                              bool iIsSource) : label_(iLabel),type_(iType),isSource_(iIsSource){}
         bool operator<( const ComponentDescription& iRHS) const {
            return (type_ == iRHS.type_) ? (label_ < iRHS.label_) : (type_<iRHS.type_);
         }
         bool operator==(const ComponentDescription& iRHS) const {
            return label_ == iRHS.label_ &&
            type_ == iRHS.type_ &&
            isSource_ == iRHS.isSource_;
         }
      };
      
   }
}



#endif
