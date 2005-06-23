#ifndef EVENTSETUP_DATAKEYTAGS_H
#define EVENTSETUP_DATAKEYTAGS_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     DataKeyTags
// 
/**\class DataKeyTags DataKeyTags.h Core/CoreFramework/interface/DataKeyTags.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 31 14:13:07 EST 2005
// $Id: DataKeyTags.h,v 1.1 2005/05/29 02:29:53 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/CoreFramework/interface/HCTypeTag.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      
      class DataKey;
      
      typedef heterocontainer::HCTypeTag<DataKey> TypeTag;
      
      class SimpleStringTag {
        public:
         SimpleStringTag(const char* iString) : tag_(iString) {}
         SimpleStringTag() : tag_("") {}
         bool operator==(const SimpleStringTag& iRHS) const ;
         bool operator<(const SimpleStringTag& iRHS) const ;
         
         const char* value() const { return tag_; }
         
        private:
         const char* tag_;
      };

      class NameTag : public SimpleStringTag {
       public:
         NameTag(const char* iUsage) : SimpleStringTag(iUsage) {}
         NameTag() : SimpleStringTag() {}
      };
      
      typedef NameTag IdTags;
   }
}
#endif /* EVENTSETUP_DATAKEYTAGS_H */
