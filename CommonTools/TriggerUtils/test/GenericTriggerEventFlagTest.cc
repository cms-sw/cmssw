// -*- C++ -*-
//
// Package:    TriggerUtils
// Class:      GenericTriggerEventFlagTest
//
// $Id: GenericTriggerEventFlagTest.cc,v 1.1 2012/01/19 17:18:35 vadler Exp $
//
/**
  \class    GenericTriggerEventFlagTest
  \brief    Tests functionality of GenericTriggerEventFlag

   This unit test ...

  \author   Volker Adler
  \version  $Id: GenericTriggerEventFlagTest.cc,v 1.1 2012/01/19 17:18:35 vadler Exp $
*/


#include "FWCore/Framework/interface/EDFilter.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


class GenericTriggerEventFlagTest : public edm::EDFilter {

    GenericTriggerEventFlag * genericTriggerEventFlag_;

  public:

    explicit GenericTriggerEventFlagTest( const edm::ParameterSet & iConfig );
    virtual ~GenericTriggerEventFlagTest();

  private:

    virtual bool beginRun( edm::Run & iRun, const edm::EventSetup & iSetup );
    virtual bool filter( edm::Event & iEvent, const edm::EventSetup & iSetup );

};


GenericTriggerEventFlagTest::GenericTriggerEventFlagTest( const edm::ParameterSet & iConfig )
: genericTriggerEventFlag_( new GenericTriggerEventFlag( iConfig ) )
{
}


GenericTriggerEventFlagTest::~GenericTriggerEventFlagTest()
{

  delete genericTriggerEventFlag_;

}


bool GenericTriggerEventFlagTest::beginRun( edm::Run & iRun, const edm::EventSetup & iSetup )
{

  if ( genericTriggerEventFlag_->on() ) genericTriggerEventFlag_->initRun( iRun, iSetup );

  return true;

}


bool GenericTriggerEventFlagTest::filter( edm::Event & iEvent, const edm::EventSetup & iSetup )
{

  if ( genericTriggerEventFlag_->on() && ! genericTriggerEventFlag_->accept( iEvent, iSetup ) ) return false;

  return true;

}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( GenericTriggerEventFlagTest );
