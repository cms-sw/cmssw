// -*- C++ -*-
//
// Package:    TriggerUtils
// Class:      GenericTriggerEventFlagTest
//
// $Id: GenericTriggerEventFlagTest.cc,v 1.3 2013/02/28 00:22:50 wmtan Exp $
//
/**
  \class    GenericTriggerEventFlagTest
  \brief    Tests functionality of GenericTriggerEventFlag

   This unit test ...

  \author   Volker Adler
  \version  $Id: GenericTriggerEventFlagTest.cc,v 1.3 2013/02/28 00:22:50 wmtan Exp $
*/


#include "FWCore/Framework/interface/EDFilter.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


class GenericTriggerEventFlagTest : public edm::EDFilter {

    GenericTriggerEventFlag * genericTriggerEventFlag_;

  public:

    explicit GenericTriggerEventFlagTest( const edm::ParameterSet & iConfig );
    virtual ~GenericTriggerEventFlagTest();

  private:

    virtual void beginRun(const edm::Run & iRun, const edm::EventSetup& iSetup) override;
    virtual bool filter( edm::Event & iEvent, const edm::EventSetup& iSetup) override;

};


GenericTriggerEventFlagTest::GenericTriggerEventFlagTest( const edm::ParameterSet & iConfig )
: genericTriggerEventFlag_( new GenericTriggerEventFlag( iConfig ) )
{
}


GenericTriggerEventFlagTest::~GenericTriggerEventFlagTest()
{

  delete genericTriggerEventFlag_;

}


void GenericTriggerEventFlagTest::beginRun(const edm::Run & iRun, const edm::EventSetup& iSetup)
{

  if ( genericTriggerEventFlag_->on() ) genericTriggerEventFlag_->initRun( iRun, iSetup );

}


bool GenericTriggerEventFlagTest::filter( edm::Event & iEvent, const edm::EventSetup& iSetup)
{

  if ( genericTriggerEventFlag_->on() && ! genericTriggerEventFlag_->accept( iEvent, iSetup ) ) return false;

  return true;

}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( GenericTriggerEventFlagTest );
