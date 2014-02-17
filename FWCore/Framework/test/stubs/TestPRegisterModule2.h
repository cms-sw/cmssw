#ifndef test_TestPRegisterModule2_h
#define test_TestPRegisterModule2_h
// -*- C++ -*-
//
// Package:     test
// Class  :     TestPRegisterModule2
// 
/**\class TestPRegisterModule2 TestPRegisterModule2.h Framework/test/interface/TestPRegisterModule2.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Sep 24 10:57:51 CEST 2005
// $Id: TestPRegisterModule2.h,v 1.2 2005/12/28 00:51:29 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/EDProducer.h"

// forward declarations

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}

class TestPRegisterModule2 : public edm::EDProducer
{
public:
   explicit TestPRegisterModule2(edm::ParameterSet const& p);
   
   void produce(edm::Event& e, edm::EventSetup const&);
   
private:
};

#endif
