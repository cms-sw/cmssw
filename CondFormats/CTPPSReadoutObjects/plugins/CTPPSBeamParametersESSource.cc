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

  CTPPSBeamParametersESSource(const edm::ParameterSet &);
  ~CTPPSBeamParametersESSource() override;

  std::unique_ptr<CTPPSBeamParameters> produce(const CTPPSBeamParametersRcd &);

private:
  
  bool setBeamPars;
  
  void initializeBeamParameters();    //  Initialize beam parameters (BP) to zero 
  void setBeamParameters(const edm::ParameterSet &);    //  Set BP to their values from config
  CTPPSBeamParameters* fillBeamParameters();   //  Fill CTPPSBeamParameters object with BP 
  
  // Beam parameters 
  double beamMom45, beamMom56, betaStarX45, betaStarX56, betaStarY45, betaStarY56, 
         beamDivX45, beamDivX56, beamDivY45, beamDivY56, halfXangleX45, halfXangleX56, 
         halfXangleY45, halfXangleY56, vtxOffsetX45, vtxOffsetX56, vtxOffsetY45, vtxOffsetY56, 
         vtxOffsetZ45, vtxOffsetZ56, vtxStddevX, vtxStddevY, vtxStddevZ ; 

protected:

/// sets infinite validity of this data
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&) override;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

CTPPSBeamParametersESSource::CTPPSBeamParametersESSource(const edm::ParameterSet& conf) :
  setBeamPars(conf.getUntrackedParameter<bool>("setBeamPars","False"))
  
{

  initializeBeamParameters();
  if(setBeamPars) setBeamParameters(conf);
  
  setWhatProduced(this);
  findingRecord<CTPPSBeamParametersRcd>();

}

//----------------------------------------------------------------------------------------------------

CTPPSBeamParametersESSource::~CTPPSBeamParametersESSource()
{
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSBeamParameters> CTPPSBeamParametersESSource::produce( const CTPPSBeamParametersRcd & )
{
  
  CTPPSBeamParameters *bp;
  bp = new CTPPSBeamParameters();
  
  // If beam parameters are available from the config file, fill their values into CTPPSBeamParameters object
  if(setBeamPars) bp = fillBeamParameters();
  
  edm::LogInfo("CTPPSBeamParametersESSource::produce") << "\n" << *bp << "\n" ;

  return std::unique_ptr<CTPPSBeamParameters>(bp);
  
}

//----------------------------------------------------------------------------------------------------

void CTPPSBeamParametersESSource::initializeBeamParameters() 
{
  beamMom45 = beamMom56 = betaStarX45 = betaStarX56 = betaStarY45 = betaStarY56 = 
  beamDivX45 = beamDivX56 = beamDivY45 = beamDivY56 = halfXangleX45 = halfXangleX56 = 
  halfXangleY45 = halfXangleY56 = vtxOffsetX45 = vtxOffsetX56 = vtxOffsetY45 = vtxOffsetY56 = 
  vtxOffsetZ45 = vtxOffsetZ56 = vtxStddevX = vtxStddevY = vtxStddevZ = 0.0 ;
}

//----------------------------------------------------------------------------------------------------

void CTPPSBeamParametersESSource::setBeamParameters(const edm::ParameterSet& conf)
{
    beamMom45 = conf.getParameter<double>("beamMom45");
    beamMom56 = conf.getParameter<double>("beamMom56");
    betaStarX45 = conf.getParameter<double>("betaStarX45");
    betaStarX56 = conf.getParameter<double>("betaStarX56");
    betaStarY45 = conf.getParameter<double>("betaStarY45");
    betaStarY56 = conf.getParameter<double>("betaStarY56");
    beamDivX45 = conf.getParameter<double>("beamDivX45");
    beamDivX56 = conf.getParameter<double>("beamDivX56");
    beamDivY45 = conf.getParameter<double>("beamDivY45");
    beamDivY56 = conf.getParameter<double>("beamDivY56");
    halfXangleX45 = conf.getParameter<double>("halfXangleX45");
    halfXangleX56 = conf.getParameter<double>("halfXangleX56");
    halfXangleY45 = conf.getParameter<double>("halfXangleY45");
    halfXangleY56 = conf.getParameter<double>("halfXangleY56");
    vtxOffsetX45 = conf.getParameter<double>("vtxOffsetX45");
    vtxOffsetY45 = conf.getParameter<double>("vtxOffsetY45");
    vtxOffsetZ45 = conf.getParameter<double>("vtxOffsetZ45");
    vtxOffsetX56 = conf.getParameter<double>("vtxOffsetX56");
    vtxOffsetY56 = conf.getParameter<double>("vtxOffsetY56");
    vtxOffsetZ56 = conf.getParameter<double>("vtxOffsetZ56");
    vtxStddevX = conf.getParameter<double>("vtxStddevX");
    vtxStddevY = conf.getParameter<double>("vtxStddevY");
    vtxStddevZ = conf.getParameter<double>("vtxStddevZ");
}

//----------------------------------------------------------------------------------------------------

CTPPSBeamParameters* CTPPSBeamParametersESSource::fillBeamParameters() 
{
  CTPPSBeamParameters* p = new CTPPSBeamParameters();
  
  p->setBeamMom45( beamMom45 );
  p->setBeamMom56( beamMom56 );
  
  p->setBetaStarX45( betaStarX45 );
  p->setBetaStarY45( betaStarY45 );
  p->setBetaStarX56( betaStarX56 );
  p->setBetaStarY56( betaStarY56 );
  
  p->setBeamDivergenceX45( beamDivX45 );
  p->setBeamDivergenceY45( beamDivY45 );
  p->setBeamDivergenceX56( beamDivX56 );
  p->setBeamDivergenceY56( beamDivY56 );
  
  p->setHalfXangleX45( halfXangleX45 );
  p->setHalfXangleY45( halfXangleY45 );
  p->setHalfXangleX56( halfXangleX56 );
  p->setHalfXangleY56( halfXangleY56 );
  
  p->setVtxOffsetX45( vtxOffsetX45 );
  p->setVtxOffsetY45( vtxOffsetY45 );
  p->setVtxOffsetZ45( vtxOffsetZ45 );
  p->setVtxOffsetX56( vtxOffsetX56 );
  p->setVtxOffsetY56( vtxOffsetY56 );
  p->setVtxOffsetZ56( vtxOffsetZ56 );
  
  p->setVtxStddevX( vtxStddevX );
  p->setVtxStddevY( vtxStddevY );
  p->setVtxStddevZ( vtxStddevZ );
  
  // edm::LogInfo("CTPPSBeamParametersESSource::fillBeamParameters") << "\n" ;
  
  return p ;
}

//----------------------------------------------------------------------------------------------------

void CTPPSBeamParametersESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
						     const edm::IOVSyncValue& iosv, edm::ValidityInterval& oValidity)
{
  /*
  LogVerbatim("CTPPSBeamParametersESSource")
    << ">> CTPPSBeamParametersESSource::setIntervalFor(" << key.name() << ")";

  LogVerbatim("CTPPSBeamParametersESSource")
    << "    run=" << iosv.eventID().run() << ", event=" << iosv.eventID().event();
  */
  
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;
  
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSBeamParametersESSource);
