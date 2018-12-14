/****************************************************************************
 *
 *  CondFormats/CTPPSReadoutObjects/plugins/CTPPSBeamParametersESSource.cc
 *
 *  Description :  - Loads CTPPSBeamParameters from the CTPPSBeamParametersESSource_cfi.py
 *                   config file.
 *                 - Currently, one single set of beam parameters is provided. Just to be
 *                   ready in time for 10_4_0 and allow simple tests.
 *                 - To be further developed to provide multiple sets of parameters
 *                 - Needed while CTPPSBeamParameters is not available in database
 *
 *
 * Author:
 * Wagner Carvalho wcarvalh@cern.ch
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"

#include <memory>

//----------------------------------------------------------------------------------------------------

using namespace std;

/**
 * \brief Loads CTPPSBeamParameters from a config file.
 **/

class CTPPSBeamParametersESSource: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder
{
public:

  CTPPSBeamParametersESSource(const edm::ParameterSet&);
  ~CTPPSBeamParametersESSource() override = default;

  std::unique_ptr<CTPPSBeamParameters> produce(const CTPPSBeamParametersRcd&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:

  bool setBeamPars_;

  /// Set BP to their values from config
  void setBeamParameters(const edm::ParameterSet &);
  ///  Fill CTPPSBeamParameters object with BP
  std::unique_ptr<CTPPSBeamParameters> fillBeamParameters();

  // Beam parameters
  double beamMom45_, beamMom56_;
  double betaStarX45_, betaStarY45_, betaStarX56_, betaStarY56_;
  double beamDivX45_, beamDivY45_, beamDivX56_, beamDivY56_;
  double halfXangleX45_, halfXangleY45_;
  double halfXangleX56_, halfXangleY56_;
  double vtxOffsetX45_, vtxOffsetY45_, vtxOffsetZ45_;
  double vtxOffsetX56_, vtxOffsetY56_, vtxOffsetZ56_;
  double vtxStddevX_, vtxStddevY_, vtxStddevZ_;

protected:

/// sets infinite validity of this data
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&) override;
};

//----------------------------------------------------------------------------------------------------

CTPPSBeamParametersESSource::CTPPSBeamParametersESSource(const edm::ParameterSet& iConfig) :
  setBeamPars_( iConfig.getParameter<bool>( "setBeamPars" ) ),
  beamMom45_( 0. ), beamMom56_( 0. ),
  betaStarX45_( 0. ), betaStarY45_( 0. ), betaStarX56_( 0. ), betaStarY56_( 0. ),
  beamDivX45_( 0. ), beamDivY45_( 0. ), beamDivX56_( 0. ), beamDivY56_( 0. ),
  halfXangleX45_( 0. ), halfXangleY45_( 0. ),
  halfXangleX56_( 0. ), halfXangleY56_( 0. ),
  vtxOffsetX45_( 0. ), vtxOffsetY45_( 0. ), vtxOffsetZ45_( 0. ),
  vtxOffsetX56_( 0. ), vtxOffsetY56_( 0. ), vtxOffsetZ56_( 0. ),
  vtxStddevX_( 0. ), vtxStddevY_( 0. ), vtxStddevZ_( 0. )
{
  if ( setBeamPars_ )
    setBeamParameters( iConfig );

  setWhatProduced(this);
  findingRecord<CTPPSBeamParametersRcd>();
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSBeamParameters>
CTPPSBeamParametersESSource::produce( const CTPPSBeamParametersRcd & )
{
  // If beam parameters are available from the config file, fill their values into CTPPSBeamParameters object
  auto bp = ( setBeamPars_ )
    ? fillBeamParameters()
    : std::make_unique<CTPPSBeamParameters>();

  edm::LogInfo("CTPPSBeamParametersESSource::produce") << "\n" << *bp;

  return bp;
}

//----------------------------------------------------------------------------------------------------

void
CTPPSBeamParametersESSource::setBeamParameters( const edm::ParameterSet& iConfig )
{
  beamMom45_ = iConfig.getParameter<double>( "beamMom45" );
  beamMom56_ = iConfig.getParameter<double>( "beamMom56" );
  betaStarX45_ = iConfig.getParameter<double>( "betaStarX45" );
  betaStarX56_ = iConfig.getParameter<double>( "betaStarX56" );
  betaStarY45_ = iConfig.getParameter<double>( "betaStarY45" );
  betaStarY56_ = iConfig.getParameter<double>( "betaStarY56" );
  beamDivX45_ = iConfig.getParameter<double>( "beamDivX45" );
  beamDivX56_ = iConfig.getParameter<double>( "beamDivX56" );
  beamDivY45_ = iConfig.getParameter<double>( "beamDivY45" );
  beamDivY56_ = iConfig.getParameter<double>( "beamDivY56" );
  halfXangleX45_ = iConfig.getParameter<double>( "halfXangleX45" );
  halfXangleX56_ = iConfig.getParameter<double>( "halfXangleX56" );
  halfXangleY45_ = iConfig.getParameter<double>( "halfXangleY45" );
  halfXangleY56_ = iConfig.getParameter<double>( "halfXangleY56" );
  vtxOffsetX45_ = iConfig.getParameter<double>( "vtxOffsetX45" );
  vtxOffsetY45_ = iConfig.getParameter<double>( "vtxOffsetY45" );
  vtxOffsetZ45_ = iConfig.getParameter<double>( "vtxOffsetZ45" );
  vtxOffsetX56_ = iConfig.getParameter<double>( "vtxOffsetX56" );
  vtxOffsetY56_ = iConfig.getParameter<double>( "vtxOffsetY56" );
  vtxOffsetZ56_ = iConfig.getParameter<double>( "vtxOffsetZ56" );
  vtxStddevX_ = iConfig.getParameter<double>( "vtxStddevX" );
  vtxStddevY_ = iConfig.getParameter<double>( "vtxStddevY" );
  vtxStddevZ_ = iConfig.getParameter<double>( "vtxStddevZ" );
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSBeamParameters>
CTPPSBeamParametersESSource::fillBeamParameters()
{
  auto p = std::make_unique<CTPPSBeamParameters>();

  p->setBeamMom45( beamMom45_ );
  p->setBeamMom56( beamMom56_ );

  p->setBetaStarX45( betaStarX45_ );
  p->setBetaStarY45( betaStarY45_ );
  p->setBetaStarX56( betaStarX56_ );
  p->setBetaStarY56( betaStarY56_ );

  p->setBeamDivergenceX45( beamDivX45_ );
  p->setBeamDivergenceY45( beamDivY45_ );
  p->setBeamDivergenceX56( beamDivX56_ );
  p->setBeamDivergenceY56( beamDivY56_ );

  p->setHalfXangleX45( halfXangleX45_ );
  p->setHalfXangleY45( halfXangleY45_ );
  p->setHalfXangleX56( halfXangleX56_ );
  p->setHalfXangleY56( halfXangleY56_ );

  p->setVtxOffsetX45( vtxOffsetX45_ );
  p->setVtxOffsetY45( vtxOffsetY45_ );
  p->setVtxOffsetZ45( vtxOffsetZ45_ );
  p->setVtxOffsetX56( vtxOffsetX56_ );
  p->setVtxOffsetY56( vtxOffsetY56_ );
  p->setVtxOffsetZ56( vtxOffsetZ56_ );

  p->setVtxStddevX( vtxStddevX_ );
  p->setVtxStddevY( vtxStddevY_ );
  p->setVtxStddevZ( vtxStddevZ_ );

  return p;
}

//----------------------------------------------------------------------------------------------------

void
CTPPSBeamParametersESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& key,
                                             const edm::IOVSyncValue& iosv, edm::ValidityInterval& oValidity )
{
  edm::LogInfo("CTPPSBeamParametersESSource")
    << ">> CTPPSBeamParametersESSource::setIntervalFor(" << key.name() << ")\n"
    << "    run=" << iosv.eventID().run() << ", event=" << iosv.eventID().event();

  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;
}

//----------------------------------------------------------------------------------------------------

void
CTPPSBeamParametersESSource::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.add<bool>( "setBeamPars", true );
  // beam momentum (GeV)
  desc.add<double>( "beamMom45", 6500. );
  desc.add<double>( "beamMom56", 6500. );
  // beta* (cm)
  desc.add<double>( "betaStarX45", 30. );
  desc.add<double>( "betaStarY45", 30. );
  desc.add<double>( "betaStarX56", 30. );
  desc.add<double>( "betaStarY56", 30. );
  // beam divergence (rad)
  desc.add<double>( "beamDivX45", 0.1 );
  desc.add<double>( "beamDivY45", 0.1 );
  desc.add<double>( "beamDivX56", 0.1 );
  desc.add<double>( "beamDivY56", 0.1 );
  // half crossing angle (rad)
  desc.add<double>( "halfXangleX45", 80.e-6 );
  desc.add<double>( "halfXangleY45", 80.e-6 );
  desc.add<double>( "halfXangleX56", 80.e-6 );
  desc.add<double>( "halfXangleY56", 80.e-6 );
  // vertex offset (cm)
  desc.add<double>( "vtxOffsetX45", 1.e-2 );
  desc.add<double>( "vtxOffsetY45", 1.e-2 );
  desc.add<double>( "vtxOffsetZ45", 1.e-2 );
  desc.add<double>( "vtxOffsetX56", 1.e-2 );
  desc.add<double>( "vtxOffsetY56", 1.e-2 );
  desc.add<double>( "vtxOffsetZ56", 1.e-2 );
  // vertex sigma (cm)
  desc.add<double>( "vtxStddevX", 2.e-2 );
  desc.add<double>( "vtxStddevY", 2.e-2 );
  desc.add<double>( "vtxStddevZ", 2.e-2 );

  descriptions.add( "ctppsBeamParametersESSource", desc );
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE( CTPPSBeamParametersESSource );

