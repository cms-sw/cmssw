// -*- C++ -*-
//
// Package:    TriggerUtils
// Class:      GenericTriggerEventFlagTest
//
// $Id:$
//
/**
  \class    GenericTriggerEventFlagTest
  \brief    Tests functionality of GenericTriggerEventFlag

   This unit test ...

  \author   Volker Adler
  \version  $Id:$
*/


#include "FWCore/Framework/interface/EDFilter.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


class GenericTriggerEventFlagTest : public edm::EDFilter {

  public:

    explicit GenericTriggerEventFlagTest( const edm::ParameterSet & iConfig );
    virtual ~GenericTriggerEventFlagTest();

  private:

    virtual bool beginRun( edm::Run & iRun, const edm::EventSetup & iSetup );
    virtual bool filter( edm::Event & iEvent, const edm::EventSetup & iSetup );

    GenericTriggerEventFlag * genericTriggerEventFlag_;

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
