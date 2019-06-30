/*! \class DTTPG
 *  \author Nicola Pozzobon
 *  \brief EDProducer of L1 DT based on the Hough Transform
 *  \date 2018, Sep 12
 */

#define IS_THIS_CMSSW

#include "L1Trigger/DTHoughTPG/plugins/DTTPG.h"

DTTPG::DTTPG( const edm::ParameterSet& aConfig )
{
  std::cerr << "************************************************************" << std::endl
            << "*      D    I    S    C    L    A    I    M    E    R      *" << std::endl
            << "************************************************************" << std::endl
            << "* This simulation of the Phase 2 DT Trigger Primitives     *" << std::endl
            << "* based on Majority-Mean Timer and Compact Hough Transform *" << std::endl
            << "* is meant to run within CMSSW and does not reproduce two  *" << std::endl
            << "* features of the design version that is used for hardware *" << std::endl
            << "* development and that requires a dedicated setup to have  *" << std::endl
            << "* it running in a CMSSW-based working area:                *" << std::endl
            << "*                                                          *" << std::endl
            << "* 1) this simulation employs standard integer data types   *" << std::endl
            << "*    instead of arbitrary size data types, which may cause *" << std::endl
            << "*    unoptimal handling of extremely rare bit overflow;    *" << std::endl
            << "*                                                          *" << std::endl
            << "* 2) this simulation does not account for the actual       *" << std::endl
            << "*    stream of data and long integration windows of DTs    *" << std::endl
            << "*    due to the maximum drift time of ~16 BX durations,    *" << std::endl
            << "*    therefore processing is done event-by-event and each  *" << std::endl
            << "*    event takes place at BX = 20, as according to the     *" << std::endl
            << "*    usual CMSSW conventions.                              *" << std::endl
            << "*                                                          *" << std::endl
            << "* Also, the output of this simulation is in the form of    *" << std::endl
            << "* DTTF-friendly data formats, i.e. L1MuDTChambPhDigi and   *" << std::endl
            << "* L1MuDTChambThDigi, fixed in the number of 2 per chamber, *" << std::endl
            << "* which is extremely unlikely for Phase 2.                 *" << std::endl
            << "*                                                          *" << std::endl
            << "* 12th Sep 2018, NP, PLZ, FM                               *" << std::endl             
            << "************************************************************" << std::endl;

  produces< std::vector< DTHough< RefDTDigi_t > > >( "FromDTDigis" );
  produces< std::vector< DTHough< RefDTDigi_t > > >( "FromDTDigisSingleSL" );
  produces< std::vector< DTHough< RefDTDigi_t > > >( "FromDTDigisMMTOnly" );
  produces< L1MuDTChambPhContainer >( "MMTCHT" );
  //produces< L1MuDTChambThContainer >( "MMTCHT" );
  produces< L1Phase2MuDTPhContainer >( "MMTCHT" );
  produces< L1Phase2MuDTPhContainer >( "MMTCHTslRF" );

  std::fill( LUTfindWireZeroIdx, LUTfindWireZeroIdx + 43, 0xFF );
  LUTfindWireZeroIdx[0] = LUTfindWireZeroIdx[4] = 0;
  LUTfindWireZeroIdx[2] = LUTfindWireZeroIdx[6] = 1;
  LUTfindWireZeroIdx[18] = LUTfindWireZeroIdx[22] = 2;
  LUTfindWireZeroIdx[20] = LUTfindWireZeroIdx[24] = 3;
  LUTfindWireZeroIdx[36] = LUTfindWireZeroIdx[40] = 4;
  LUTfindWireZeroIdx[38] = LUTfindWireZeroIdx[42] = 5;

  edm::InputTag dtDigiTag( "simMuonDTDigis", "" );
  dtDigisToken = consumes< DTDigiCollection >( dtDigiTag );
}

DTTPG::~DTTPG(){}

void DTTPG::beginJob(){}

void DTTPG::beginRun( const edm::Run& aRun, const edm::EventSetup& anEventSetup )
{
  anEventSetup.get< MuonGeometryRecord >().get( DTGeometryHandle );
  const std::vector< const DTChamber* > vecChambers = DTGeometryHandle->chambers();
  for ( unsigned int iCha = 0; iCha < vecChambers.size(); ++iCha )
  {
    const DTChamber* thisChamber = vecChambers.at(iCha);
    DTChamberId thisChambId = thisChamber->id();
    unsigned int idxWheel = static_cast< unsigned int >( thisChambId.wheel() + 2 );
    unsigned int idxSector = static_cast< unsigned int >( thisChambId.sector() - 1 );
    unsigned int idxStation = static_cast< unsigned int >( thisChambId.station() - 1 );
    unsigned int idxChamber = idxWheel * 14 * 4 + idxSector * 4 + idxStation;
    assert( idxChamber < 280 );
    const DTLayer* layer1SL1 = DTGeometryHandle->layer( DTLayerId( thisChambId, 1, 1 ) );
    const DTLayer* layer1SL2 = DTGeometryHandle->layer( DTLayerId( thisChambId, 2, 1 ) );
    const DTLayer* layer1SL3 = DTGeometryHandle->layer( DTLayerId( thisChambId, 3, 1 ) );
    LocalPoint locWirePosSL1 = LocalPoint( layer1SL1->specificTopology().wirePosition(2), 0, 0 );
    LocalPoint locWirePosSL2 = ( layer1SL2 == 0x0 ) ? LocalPoint(0, 0, 0) : LocalPoint( layer1SL2->specificTopology().wirePosition(2), 0, 0 );
    LocalPoint locWirePosSL3 = LocalPoint( layer1SL3->specificTopology().wirePosition(2), 0, 0 );
    GlobalPoint globWirePosSL1 = layer1SL1->toGlobal( locWirePosSL1 );
    GlobalPoint globWirePosSL2 = ( layer1SL2 == 0x0 ) ? GlobalPoint(0, 0, 0) : layer1SL2->toGlobal( locWirePosSL2 );
    GlobalPoint globWirePosSL3 = layer1SL3->toGlobal( locWirePosSL3 );
    LocalPoint locWirePosInChambSL1 = thisChamber->toLocal( globWirePosSL1 );
    LocalPoint locWirePosInChambSL2 = ( layer1SL2 == 0x0 ) ? LocalPoint(0, 0, 0) : thisChamber->toLocal( globWirePosSL2 );
    LocalPoint locWirePosInChambSL3 = thisChamber->toLocal( globWirePosSL3 );
    double relativeShift = ( locWirePosInChambSL3.x() - locWirePosInChambSL1.x() );
    WireIdx_t numWireShift = static_cast< WireIdx_t >( std::round( 10 * relativeShift / defDTCellWidth ) / 10 );
    double centralCoordMCellZero = ( numWireShift > 0 ) ? locWirePosInChambSL1.x() : locWirePosInChambSL3.x();
    double longCoordMCellZero = locWirePosInChambSL2.y();
    vecCentralRPhiZero[ idxChamber ] = centralCoordMCellZero;
    vecCentralLongZero[ idxChamber ] = longCoordMCellZero;
    vecNumWireShift[ idxChamber ] = numWireShift;
  }
}

void DTTPG::endRun( const edm::Run& aRun, const edm::EventSetup& anEventSetup ){}

void DTTPG::produce( edm::Event& anEvent, const edm::EventSetup& anEventSetup )
{
  int32_t evtBx = anEvent.eventAuxiliary().bunchCrossing();
  if ( evtBx == -1 )
    evtBx = 20;
  int32_t localZeroTime = evtBx << 5;

  anEvent.getByToken( dtDigisToken, DTDigiHandle );
  anEventSetup.get< MuonGeometryRecord >().get( DTGeometryHandle );

  auto outputHoughTrig = std::make_unique< std::vector< DTHough< RefDTDigi_t > > >();
  auto outputHoughTrigSingleSL = std::make_unique< std::vector< DTHough< RefDTDigi_t > > >();
  auto outputHoughTrigMMTOnly = std::make_unique< std::vector< DTHough< RefDTDigi_t > > >();
  std::vector< L1MuDTChambPhDigi > *outputPhiTrigger = new std::vector< L1MuDTChambPhDigi >();
  std::vector< L1Phase2MuDTPhDigi > *outputPhiTrigger2 = new std::vector< L1Phase2MuDTPhDigi >();
  std::vector< L1Phase2MuDTPhDigi > *outputPhiTrigger2slRF = new std::vector< L1Phase2MuDTPhDigi >();

  //std::vector< L1MuDTChambThDigi > outputThetaTrigger;

  std::map< DTChamberId, std::vector< std::pair< uint32_t, RefDTDigi_t > > > mapDigisByChamber = this->RetrieveDigis( anEvent );
  this->RunAlgorithm( localZeroTime, anEvent.isRealData(), mapDigisByChamber, &(*outputHoughTrigMMTOnly), &(*outputHoughTrigSingleSL), &(*outputHoughTrig), &(*outputPhiTrigger), &(*outputPhiTrigger2), &(*outputPhiTrigger2slRF) );

  anEvent.put( std::move( outputHoughTrig ), "FromDTDigis" );
  anEvent.put( std::move( outputHoughTrigSingleSL ), "FromDTDigisSingleSL" );
  anEvent.put( std::move( outputHoughTrigMMTOnly ), "FromDTDigisMMTOnly" );
  std::unique_ptr< L1MuDTChambPhContainer > resultPhiTrig ( new L1MuDTChambPhContainer );
  resultPhiTrig->setContainer( *outputPhiTrigger );
  anEvent.put( std::move( resultPhiTrig ), "MMTCHT" );
  //std::unique_ptr< L1MuDTChambThContainer > resultThetaTrig ( new L1MuDTChambThContainer );
  //resultThetaTrig->setContainer( outputThetaTrigger );
  //anEvent.put( std::move( resultThetaTrig ), "MMTCHT" );

  std::unique_ptr< L1Phase2MuDTPhContainer > resultPhiTrigPh2 ( new L1Phase2MuDTPhContainer );
  resultPhiTrigPh2->setContainer( *outputPhiTrigger2 );
  anEvent.put( std::move( resultPhiTrigPh2 ), "MMTCHT" );
  std::unique_ptr< L1Phase2MuDTPhContainer > resultPhiTrigPh2slRF ( new L1Phase2MuDTPhContainer );
  resultPhiTrigPh2slRF->setContainer( *outputPhiTrigger2slRF );
  anEvent.put( std::move( resultPhiTrigPh2slRF ), "MMTCHTslRF" );

  return;
}

void DTTPG::endJob(){}

DEFINE_FWK_MODULE(DTTPG);
