#ifndef FWCore_Utilities_ESInputTag_h
#define FWCore_Utilities_ESInputTag_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     ESInputTag
// 
/**\class ESInputTag ESInputTag.h FWCore/Utilities/interface/ESInputTag.h

 Description: Parameter type used to denote how data should be obtained from the EventSetup

 Usage:
    The ESInputTag can be used in conjunction with an EventSetup Record to retrieve a particular data
 item from the Record.  The EventSetup uses two pieces of information to find data in a Record
    1) the C++ class type of the data
    2) an optional string which we will refer to as the 'dataLabel' 
 The dataLabel is used to differentiate objects of the same type placed in the same Record.
 
 In addition, every piece of data in the EventSetup comes from either an ESSource or an ESProducer.  Every 
 ESSource and ESProducer is assigned a label (referred to here as the moduleLabel) in the configuration of
 the job.  This label may be explicitly set or it may just be the C++ class type of the ESSource/ESProducer.
 
 The ESInputTag allows one to specify both the dataLabel and moduleLabel.  The dataLabel is used to find the data
 being requested.  The moduleLabel is only used to determine if the data that was found comes from the specified
 module.  If the data does not come from the module then an error has occurred.  If the moduleLabel is set to the
 empty string then the data is allowed to come from any module.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 19 13:01:38 CST 2009
//

// system include files
#include <string>

// user include files

// forward declarations
namespace edm {
   class ESInputTag {
      
   public:
      ESInputTag();
      ESInputTag(const std::string& iModuleLabel, const std::string& iDataLabel);
      ESInputTag(const std::string& iEncodedValue);
      
      //virtual ~ESInputTag();
      
      // ---------- const member functions ---------------------
      
      /**Returns the label assigned to the module for the data to be retrieved.
       If the value matches the defaultModule value (which is the empty string)
       Then no match is attempted with the module label.
       */
      const std::string& module() const { return module_;}
      
      /**Returns the label used to access the data from the EventSetup.
       The empty string is an allowed (and default) value.
       */
      const std::string& data() const {return data_;}
      
      bool operator==(const edm::ESInputTag& iRHS) const;
      
      std::string encode() const;
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      
   private:
      //ESInputTag(const ESInputTag&); // allow default
      
      //const ESInputTag& operator=(const ESInputTag&); // allow default
      
      // ---------- member data --------------------------------
      std::string module_;
      std::string data_;
   
   };
   
}


#endif
