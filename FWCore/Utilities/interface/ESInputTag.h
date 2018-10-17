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
 For example, say there is an ESProducer of C++ class type FooESProd. If the python configuration has
       process.FooESProd = cms.ESProducer("FooESProd", ...)
 then the module label is 'FooESProd'.
 If the python configuration has
       process.add_( cms.ESProducer("FooESProd", ...)
 then the module label is also 'FooESProd'.
 However, if the python configuration has
       process.foos = cms.ESProducer("FooESProd",...)
 then the module label is 'foos'.

 The ESInputTag allows one to specify both the dataLabel and moduleLabel.  The dataLabel is used to find the data
 being requested.  The moduleLabel is only used to determine if the data that was found comes from the specified
 module.  If the data does not come from the module then an error has occurred.  If the moduleLabel is set to the
 empty string then the data is allowed to come from any module.

 Example 1:
 FooESProd makes Foos with the default dataLabel ("")
 The module is declared in the python configuration as
     process.FooESProd = cms.ESProducer("FooESProd",...)
 The following python configurations for ESInputTag can be used to get its data
     cms.ESInputTag("")
     cms.ESInputTag(":")
     cms.ESInputTag("","")
     cms.ESInputTag("FooESProd")
     cms.ESInputTag("FooESProd:")
     cms.ESInputTag("FooESProd","")

 Example 2:
 FooESProd makes Foos with the dataLabel "bar"
 The module is declared in the python configuration as
     process.FooESProd = cms.ESProducer("FooESProd",...)
 The following python configurations for ESInputTag can be used to get its data
     cms.ESInputTag(":bar")
     cms.ESInputTag("","bar")
     cms.ESInputTag("FooESProd:bar")
     cms.ESInputTag("FooESProd","bar")

 Example 3:
 FooESProd makes Foos with the default dataLabel ("")
     process.FooESProd = cms.ESProducer("FooESProd",...)
 Foo2ESProd also makes Foos with the default dataLabel ("")
     process.Foo2ESProd = cms.ESProducer("Foo2ESProd",...)
 The jobs has an ESPrefer which states Foos should come from FooESProd
     process.perferedFoo = cms.ESPrefer("FooESProd")
 Then to get the data, one can specify the ESInputTag identical to Example 1.
 However, the following ESInputTags will lead to an exception begin thrown
     cms.ESInputTag("Foo2ESProd")
     cms.ESInputTag("Foo2ESProd:")
     cms.ESInputTag("Foo2ESProd","")
 The exception happens because FooESProd is the only allowed source of Foos
 but the ESInputTag says you really wanted them to come from Foo2ESProd.
*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 19 13:01:38 CST 2009
//

// system include files
#include <iosfwd>
#include <string>

// user include files

// forward declarations
namespace edm {
   class ESInputTag {
   public:
      ESInputTag();
      ESInputTag(const std::string& iModuleLabel, const std::string& iDataLabel);
      ESInputTag(const std::string& iEncodedValue);

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

   private:

      // ---------- member data --------------------------------
      std::string module_;
      std::string data_;
   };

   std::ostream& operator<<(std::ostream&, ESInputTag const&);
}

#endif
