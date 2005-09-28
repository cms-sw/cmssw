#ifndef test_TestPRegisterModule1_h
#define test_TestPRegisterModule1_h
// -*- C++ -*-
//
// Package:     test
// Class  :     TestPRegisterModule1
// 
/**\class TestPRegisterModule1 TestPRegisterModule1.h Framework/test/interface/TestPRegisterModule1.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Sep 24 10:57:48 CEST 2005
// $Id$
//

// system include files

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations

class TestPRegisterModule1 : public edm::EDProducer
{
public:
   explicit TestPRegisterModule1(edm::ParameterSet const& p);   
   void produce(edm::Event& e, edm::EventSetup const&);
   
private:
   edm::ParameterSet pset_;
};


#endif
