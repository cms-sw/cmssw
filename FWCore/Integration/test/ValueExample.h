#ifndef Integration_test_ValueExample_h
#define Integration_test_ValueExample_h
// -*- C++ -*-
//
// Package:     Integration
// Class  :     ValueExample
// 
/**\class ValueExample ValueExample.h FWCore/Integration/test/stubs/ValueExample.h

 Description: An example of a trivial Service

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:51:59 EDT 2005
//

// system include files

// user include files
namespace edm {
  class ParameterSet;
}

// forward declarations
class ValueExample
{
   
public:
   ValueExample(const edm::ParameterSet&);
   virtual ~ValueExample();
   
   // ---------- const member functions ---------------------
   int value() const { return value_; }
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   
private:
      ValueExample(const ValueExample&); // stop default
   
   const ValueExample& operator=(const ValueExample&); // stop default
   
   // ---------- member data --------------------------------
   int value_;
};

#endif
