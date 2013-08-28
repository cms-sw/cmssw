// -*- C++ -*-
//
// Package:     Common
// Class  :     IntValues_reflex
// 
// Implementation:
//     Hand create the reflex information for some simple tests
//
// Original Author:  Chris Jones
//         Created:  Wed Oct 31 16:23:47 EDT 2007
//

// system include files
#include "Reflex/Builder/ReflexBuilder.h"

// user include files
#include "DataFormats/Common/test/IntValues.h"

namespace {
  using namespace test_with_reflex;
  //Need Reflex dicctionaries for the conversion
  Reflex::Type type_intvalue = Reflex::TypeBuilder("IntValue");
  Reflex::Type type_intvalue2 = Reflex::TypeBuilder("IntValue2");
  
  void type_intvalue_d() {
    Reflex::ClassBuilder("IntValue", typeid(IntValue), sizeof(IntValue), Reflex::PUBLIC, Reflex::STRUCT);
  }
  void type_intvalue2_d() {
    Reflex::ClassBuilder("IntValue2",typeid(IntValue2),sizeof(IntValue2), Reflex::PUBLIC, Reflex::STRUCT).
    AddBase(type_intvalue, Reflex::BaseOffset<IntValue,IntValue2>::Get(), Reflex::PUBLIC);
  }
  
  struct Dictionaries {
    Dictionaries() {
      type_intvalue_d();
      type_intvalue2_d();
    }
    ~Dictionaries() {
      type_intvalue.Unload();
      type_intvalue2.Unload();
    }
  };
  static Dictionaries instance;
}
