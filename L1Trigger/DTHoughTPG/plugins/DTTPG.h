#ifndef DTTPG_h
#define DTTPG_h

/*! \class DTTPG
 *  \author Nicola Pozzobon
 *  \brief EDProducer of L1 DT based on the Hough Transform
 *  \date 2018, Sep 12
 */

#include <memory>
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "L1Trigger/DTUtilities/interface/DTTrigGeom.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "L1Trigger/DTHoughTPG/interface/Constants.h"
#include "L1Trigger/DTHoughTPG/interface/DTHough.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
//#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"

class DTTPG : public edm::EDProducer
{
  public :
    explicit DTTPG( const edm::ParameterSet& iPSet );
    ~DTTPG();
  private :
    virtual void beginJob();
    virtual void beginRun( const edm::Run& iRun, const edm::EventSetup& iSetup );
    virtual void endRun( const edm::Run& iRun, const edm::EventSetup& iSetup );
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );
    virtual void endJob();

    std::map< DTChamberId, std::vector< std::pair< uint32_t, RefDTDigi_t > > > RetrieveDigis( const edm::Event& iEvent ) const;
    static bool TimeSortDigis( const std::pair< uint32_t, RefDTDigi_t > firstDigiRef,
                               const std::pair< uint32_t, RefDTDigi_t > secondDigiRef );
    InputBMask_t ConvertHit( int32_t localZeroTime, const std::pair< uint32_t, RefDTDigi_t > aDigiRef ) const;
    static bool PhiBQualSortDTHough( const DTHough< RefDTDigi_t > firstDTHough,
                                     const DTHough< RefDTDigi_t > secondDTHough );
    void StoreTempOutput( DTChamberId aChambId,
                          uint32_t superLayer,
                          int32_t localZeroTime,
                          WiBits_t wireBitsMCell[ MAX_MACROCELLS ],
                          unsigned int vecMCellCMSSWHits[ MAX_MACROCELLS ][ NUM_MACROCELLWIRES ],    
                          TimeMMT_t timeVec[ MAX_MACROCELLS ],
                          double centralMCellCoord[ MAX_MACROCELLS ], 
                          std::vector< std::pair< uint32_t, RefDTDigi_t > > thisChamberHits,
                          BOOL_t qualityFlag[ MAX_MACROCELLS ],
                          std::vector< DTHough< RefDTDigi_t > > *tempOutVector ) const;
    void StoreTempOutput( DTChamberId aChambId,
                          uint32_t superLayer,
                          int32_t localZeroTime,
                          WiBits_t wireBitsMCell[ MAX_MACROCELLS ],
                          unsigned int vecMCellCMSSWHits[ MAX_MACROCELLS ][ NUM_MACROCELLWIRES ],
                          TimeMMT_t timeVec[ MAX_MACROCELLS ],
                          double centralMCellCoord[ MAX_MACROCELLS ],
                          std::vector< std::pair< uint32_t, RefDTDigi_t > > thisChamberHits,
                          TanPhi_t twoTanPhiMCell[ MAX_MACROCELLS ][ MAX_TANPHI_CLUSTERS ],
                          MCellPos_t xZeroMCell[ MAX_MACROCELLS ][ MAX_TANPHI_CLUSTERS ],
                          Qual_t qualCHTMCell[ MAX_MACROCELLS ],
                          std::vector< DTHough< RefDTDigi_t > > *tempOutVector ) const;
    void StoreTempOutput( DTChamberId aChambId,
                          int32_t localZeroTime,
                          WiBits_t wireBitsMCell1[ MAX_MACROCELLS ],
                          WiBits_t wireBitsMCell3[ MAX_MACROCELLS ],
                          unsigned int vecMCellCMSSWHits1[ MAX_MACROCELLS ][ NUM_MACROCELLWIRES ],
                          unsigned int vecMCellCMSSWHits3[ MAX_MACROCELLS ][ NUM_MACROCELLWIRES ],
                          TimeMMT_t timeVec1[ MAX_MACROCELLS ],
                          TimeMMT_t timeVec3[ MAX_MACROCELLS ],
                          double centralMCellCoord[ MAX_MACROCELLS ],
                          std::vector< std::pair< uint32_t, RefDTDigi_t > > thisChamberHits,
                          TanPhi_t twoTanPhiMCell[ MAX_MACROCELLS ][ MAX_TANPHI_CLUSTERS * 10 ],
                          MCellPos_t xZeroMCell1[ MAX_MACROCELLS ][ MAX_TANPHI_CLUSTERS * 10 ],
                          MCellPos_t xZeroMCell3[ MAX_MACROCELLS ][ MAX_TANPHI_CLUSTERS * 10 ],
                          Qual_t qualCHTMCell[ MAX_MACROCELLS ][ 10 ],
                          std::vector< DTHough< RefDTDigi_t > > *tempOutVector ) const;
    void RemoveDuplicates( std::vector< DTHough< RefDTDigi_t > > *tempDTHoughStorage,
                           std::vector< DTHough< RefDTDigi_t > > *outputHoughTrig ) const;
    void RemoveDuplicates( std::vector< DTHough< RefDTDigi_t > > *tempDTHoughStorage,
                           std::vector< DTHough< RefDTDigi_t > > *outputHoughTrig,
                           std::vector< L1MuDTChambPhDigi > *outputPhiTrigger,
                           std::vector< L1Phase2MuDTPhDigi > *outputPhiTrigger2,
                           std::vector< L1Phase2MuDTPhDigi > *outputPhiTrigger2slRF ) const;
    void RunAlgorithm( int32_t localZeroTime, bool isData,
                       std::map< DTChamberId, std::vector< std::pair< uint32_t, RefDTDigi_t > > > aMapDigisByChamber,
                       std::vector< DTHough< RefDTDigi_t > > *anOutputHoughTrigMMTOnly,
                       std::vector< DTHough< RefDTDigi_t > > *anOutputHoughTrigCHT1SL,
                       std::vector< DTHough< RefDTDigi_t > > *anOutputHoughTrigCHT2SL,
                       std::vector< L1MuDTChambPhDigi > *anOutputPhiTrigger,
                       std::vector< L1Phase2MuDTPhDigi > *outputPhiTrigger2,
                       std::vector< L1Phase2MuDTPhDigi > *outputPhiTrigger2slRF );
    void DriveHits( b2_Idx_t aSuperLayer,
                    NHitPerChmb_t aHitCounter,
                    WireShift_t numWireShift,
                    InputBMask_t vecEncodedHits[ MAX_CHAMBERHITS ],
                    TimeMMT_t vecMCellHits[ MAX_MACROCELLS ][ NUM_MACROCELLWIRES ],
                    uint32_t vecMCellCMSSWHits[ MAX_MACROCELLS ][ NUM_MACROCELLWIRES ],
                    uint32_t vecIdxCMSSWHits[ MAX_CHAMBERHITS ] ) const;
    void RunMMTOneSL( TimeMMT_t vecMCellHits[ MAX_MACROCELLS ][ NUM_MACROCELLWIRES ],
                      WiBits_t wireBits[ MAX_MACROCELLS ],
                      BOOL_t qualityFlag[ MAX_MACROCELLS ],
                      TimeMMT_t timePMCell[ MAX_MACROCELLS ] ) const;
    TimeMMT_t RunMMTOneMC( TimeMMT_t vecHits[ NUM_MACROCELLWIRES ],
                           WiBits_t *tempWireBit,
                           BOOL_t *qualityFlag ) const;
    TimeMMT_t RunMMTEquations( TimeMMT_t lowerBound,
                               TimeMMT_t upperBound,
                               TimeMMT_t vecHits[ NUM_MACROCELLWIRES ],
                               BOOL_t *qualityFlag ) const;
    TimeMMT_t ComputeEquation( b4_Idx_t eqNumber,
                               CompMMT_t hitTimeA,
                               CompMMT_t hitTimeB,
                               CompMMT_t hitTimeC ) const;
    TimeMMT_t ComputeCorrection( b2_Idx_t corrNumber,
                                 CompMMT_t hitTime1,
                                 CompMMT_t hitTime2 ) const;
    MMTBin_t FindMMTBin( TimeMMT_t aTimeP,
                         TimeMMT_t aLowerBound,
                         TimeMMT_t anUpperBound ) const;
    void RunCHTOneSL( WireShift_t corrFirstWire,
                      TimeMMT_t vecHits[ MAX_MACROCELLS ][ NUM_MACROCELLWIRES ],
                      Qual_t vecQuality[ MAX_MACROCELLS ],
                      BOOL_t isMB1,
                      BOOL_t isPositiveWheel,
                      TimeMMT_t timePMCell[ MAX_MACROCELLS ],
                      MCellPos_t centralMCellCoord[ MAX_MACROCELLS ],
                      b9_Idx_t aChamberIdx,
                      TanPhi_t tanPhiMCell[ MAX_MACROCELLS ][ MAX_TANPHI_CLUSTERS ],
                      MCellPos_t x0MCell[ MAX_MACROCELLS ][ MAX_TANPHI_CLUSTERS ] ) const;
    void RunCHTOneSL( WireShift_t corrFirstWire,
                      TimeMMT_t vecHits[ MAX_MACROCELLS ][ NUM_MACROCELLWIRES ],
                      Qual_t vecQuality[ MAX_MACROCELLS ],
                      BOOL_t isMB1,
                      BOOL_t isPositiveWheel,
                      CHTBitset_t vecBitset[ MAX_MACROCELLS ],
                      TimeMMT_t timePMCell[ MAX_MACROCELLS ],
                      MCellPos_t centralMCellCoord[ MAX_MACROCELLS ], b9_Idx_t aChamberIdx,
                      TanPhi_t tanPhiMCell[ MAX_MACROCELLS ][ MAX_TANPHI_CLUSTERS ],
                      MCellPos_t x0MCell[ MAX_MACROCELLS ][ MAX_TANPHI_CLUSTERS ] ) const;
    void RunCHTTwoSL( WireShift_t corrFirstWire1,
                      WireShift_t corrFirstWire3,
                      TimeMMT_t vecHits1[ MAX_MACROCELLS ][ NUM_MACROCELLWIRES ],
                      TimeMMT_t vecHits3[ MAX_MACROCELLS ][ NUM_MACROCELLWIRES ], 
                      Qual_t vecQuality1[ MAX_MACROCELLS ],
                      Qual_t vecQuality3[ MAX_MACROCELLS ],
                      Qual_t vecQualityComb[ MAX_MACROCELLS ][ 10 ],
                      BOOL_t isMB1,
                      BOOL_t isPositiveWheel,
                      CHTBitset_t vecBitset1[ MAX_MACROCELLS ],
                      CHTBitset_t vecBitset3[ MAX_MACROCELLS ],
                      TimeMMT_t timePMCell1[ MAX_MACROCELLS ],
                      TimeMMT_t timePMCell3[ MAX_MACROCELLS ],
                      MCellPos_t centralMCellCoord[ MAX_MACROCELLS ],
                      b9_Idx_t aChamberIdx,
                      TanPhi_t twoTanPhiMCell[ MAX_MACROCELLS ][ MAX_TANPHI_CLUSTERS * 10 ],
                      MCellPos_t x0MCell1[ MAX_MACROCELLS ][ MAX_TANPHI_CLUSTERS * 10 ],
                      MCellPos_t x0MCell3[ MAX_MACROCELLS ][ MAX_TANPHI_CLUSTERS * 10 ] ) const;
    CHTBitset_t RunCHTOneMC( TimeMMT_t vecHits[ NUM_MACROCELLWIRES ],
                             Qual_t *aQuality,
                             TimeMMT_t timeP ) const;
    CHTBitset_t RunCHTTwoMC( MCellPos_t aMCellXDiff,
                             TimeMMT_t vecHits1[ NUM_MACROCELLWIRES ],
                             TimeMMT_t vecHits3[ NUM_MACROCELLWIRES ],
                             TimeMMT_t aTimeP ) const;
    void UnpackCHTOneMC( TimeMMT_t vecHits[ NUM_MACROCELLWIRES ],
                         TimeMMT_t aTimeP,
                         MCellPos_t aCentralMCellCoord,
                         CHTBitset_t aBitSet,
                         b9_Idx_t aChamberIdx,
                         BOOL_t isMB1,
                         BOOL_t isPositiveWheel,
                         TanPhi_t resTanPhi[ MAX_TANPHI_CLUSTERS ],
                         MCellPos_t resX0[ MAX_TANPHI_CLUSTERS ] ) const;
    void FillCHTHist( b3_Idx_t anInnerLayer,
                      b3_Idx_t anOuterLayer,
                      CompMMT_t aTimeListByPlace[ NUM_MACROCELLWIRES ],
                      CHTBin_t aHalfFillWidth,
                      CHTBin_t aHalfFillWidth2,
                      CompBin_t aNumZLayerDiff,
                      CHTBin_t aDenZLayerDiff,
                      CompBin_t aNumCorrZLayerDiff,
                      CHTBin_t aDenCorrZLayerDiff,
                      CHTHist_t aCHTHist[ BINNUM_TANPHI ],
                      TimeMMT_t aTimeP ) const;
    void FillCHTHist( b3_Idx_t anInnerLayer,
                      b3_Idx_t anOuterLayer,
                      CompMMT_t aTimeListByPlace1[ NUM_MACROCELLWIRES ],
                      CompMMT_t aTimeListByPlace3[ NUM_MACROCELLWIRES ],
                      MCellPos_t aMCellXDiff,
                      CHTBin_t aHalfFillWidth,
                      CHTBin_t aHalfFillWidth2,
                      CompBin_t aNumZLayerDiff,
                      CHTBin_t aDenZLayerDiff,
                      CompBin_t aNumCorrZLayerDiff,
                      CHTBin_t aDenCorrZLayerDiff,
                      CHTHist_t aCHTHist[ BINNUM_TANPHI ],
                      TimeMMT_t aTimeP ) const;
    void FindTanPhiClusters( TanPhiClu_t clustCentroids2TanPhi128[ MAX_TANPHI_CLUSTERS ],
                             b3_Idx_t *cntStoredClusters,
                             CHTBitset_t tanPhiBitSet ) const;
    CHTBin_t CorrectCHTWindow( CHTBin_t *aBin,
                               CHTBin_t aHalfWidth,
                               CHTBin_t aHalfWidth2,
                               b3_Idx_t aPair,
                               CHTBin_t aNumCorrZLayerDiff,
                               CHTBin_t aDenCorrZLayerDiff ) const;
    TimeTDC_t CorrectDriftTime( TanPhi_t a2TanPhi128 ) const;
    BOOL_t CheckPositiveWheel( DTChamberId aChambId ) const;
    unsigned int BuildFiredSuperLayers( Qual_t aQuality, MCellPos_t aX1, MCellPos_t aX3 ) const;

    b7_Idx_t LUTfindWireZeroIdx[43] = {0};
    WireShift_t vecNumWireShift[280] = {0};
    double vecCentralRPhiZero[280] = {0};
    double vecCentralLongZero[280] = {0};

    edm::EDGetTokenT< DTDigiCollection > dtDigisToken;
    edm::Handle< DTDigiCollection > DTDigiHandle;
    edm::ESHandle< DTGeometry > DTGeometryHandle;

    TimeMMT_t defMaxDriftTime;
    CompTDC_t defVDrift;
    TanPhi_t defSlopeToTime;
    TimeMMT_t *mmtNonLinCorr;

  protected :
};

#include "./DTTPG_Algo.icc"
#include "./DTTPG_Algo_MMT.icc"
#include "./DTTPG_Algo_CHT1.icc"
#include "./DTTPG_Algo_CHT2.icc"
#include "./DTTPG_Prod.icc"

#endif
