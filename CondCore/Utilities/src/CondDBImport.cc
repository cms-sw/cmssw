//

//TO DO: add the comparison between stored object and input object - bool cond::serialization::equal( obj,read_back_copy )
#define IMPORT_PAYLOAD_CASE( TYPENAME )  \
  if( inputTypeName == #TYPENAME ){ \
    match = true; \
    const TYPENAME& obj = *static_cast<const TYPENAME*>( inputPtr ); \
    payloadId = destination.storePayload( obj, boost::posix_time::microsec_clock::universal_time() ); \
  } 

#define IGNORE_FOR_IMPORT_CASE( TYPENAME ) \
  if( inputTypeName == #TYPENAME ){ \
    match = true; \
    payloadId = 0; \
    newInsert = false; \
    std::cout <<"WARNING: typename "<<inputTypeName<<" will be skipped in the import."<<std::endl; \
  }

#define FETCH_PAYLOAD_CASE( TYPENAME ) \
  if( payloadTypeName == #TYPENAME ){ \
    auto payload = deserialize<TYPENAME>( payloadTypeName, data, streamerInfo, isOra ); \
    payloadPtr = payload; \
    match = true; \
  }

#include "CondCore/Utilities/interface/CondDBImport.h"
#include "CondCore/CondDB/interface/Serialization.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondFormats.h"
//
#include <sstream>

namespace cond {

  namespace persistency {

    cond::Hash import( const std::string& inputTypeName, const void* inputPtr, Session& destination ){
      cond::Hash payloadId("");
      bool newInsert = false;
      bool match = false;
      if( inputPtr ){
      IMPORT_PAYLOAD_CASE( std::string ) 
      IMPORT_PAYLOAD_CASE( std::vector<unsigned long long> )
      IMPORT_PAYLOAD_CASE( AlCaRecoTriggerBits )
      IMPORT_PAYLOAD_CASE( AlignmentErrors )
      IMPORT_PAYLOAD_CASE( AlignmentSurfaceDeformations )
      IMPORT_PAYLOAD_CASE( Alignments )
      IMPORT_PAYLOAD_CASE( BeamSpotObjects )
      IMPORT_PAYLOAD_CASE( CSCBadChambers )
      IMPORT_PAYLOAD_CASE( CSCBadStrips )
      IMPORT_PAYLOAD_CASE( CSCBadWires )
      IMPORT_PAYLOAD_CASE( CSCChamberIndex )
      IMPORT_PAYLOAD_CASE( CSCChamberMap )
      IMPORT_PAYLOAD_CASE( CSCChamberTimeCorrections )
      IMPORT_PAYLOAD_CASE( CSCCrateMap )
      IMPORT_PAYLOAD_CASE( CSCDBChipSpeedCorrection )
      IMPORT_PAYLOAD_CASE( CSCDBCrosstalk )
      IMPORT_PAYLOAD_CASE( CSCDBGains )
      IMPORT_PAYLOAD_CASE( CSCDBGasGainCorrection )
      IMPORT_PAYLOAD_CASE( CSCDBL1TPParameters )
      IMPORT_PAYLOAD_CASE( CSCDBNoiseMatrix )
      IMPORT_PAYLOAD_CASE( CSCDBPedestals )
      IMPORT_PAYLOAD_CASE( CSCDDUMap )
      IMPORT_PAYLOAD_CASE( CSCL1TPParameters )
      IMPORT_PAYLOAD_CASE( CSCRecoDigiParameters )
      IMPORT_PAYLOAD_CASE( CastorChannelQuality )
      IMPORT_PAYLOAD_CASE( CastorElectronicsMap )
      IMPORT_PAYLOAD_CASE( CastorGainWidths )
      IMPORT_PAYLOAD_CASE( CastorGains )
      IMPORT_PAYLOAD_CASE( CastorPedestalWidths )
      IMPORT_PAYLOAD_CASE( CastorPedestals )
      IMPORT_PAYLOAD_CASE( CastorQIEData )
      IMPORT_PAYLOAD_CASE( CastorRecoParams )
      IMPORT_PAYLOAD_CASE( CastorSaturationCorrs )
      IMPORT_PAYLOAD_CASE( CentralityTable )
      IMPORT_PAYLOAD_CASE( DTCCBConfig )
      IMPORT_PAYLOAD_CASE( DTDeadFlag )
      IMPORT_PAYLOAD_CASE( DTHVStatus )
      IMPORT_PAYLOAD_CASE( DTKeyedConfig )
      IMPORT_PAYLOAD_CASE( DTLVStatus )
      IMPORT_PAYLOAD_CASE( DTMtime )
      IMPORT_PAYLOAD_CASE( DTReadOutMapping )
      IMPORT_PAYLOAD_CASE( DTStatusFlag )
      IMPORT_PAYLOAD_CASE( DTT0 )
      IMPORT_PAYLOAD_CASE( DTTPGParameters )
      IMPORT_PAYLOAD_CASE( DTTtrig )
      IMPORT_PAYLOAD_CASE( DropBoxMetadata )
      IMPORT_PAYLOAD_CASE( ESChannelStatus )
      IMPORT_PAYLOAD_CASE( ESEEIntercalibConstants )
      IMPORT_PAYLOAD_CASE( ESFloatCondObjectContainer )
      IMPORT_PAYLOAD_CASE( ESGain )
      IMPORT_PAYLOAD_CASE( ESMIPToGeVConstant )
      IMPORT_PAYLOAD_CASE( ESMissingEnergyCalibration )
      IMPORT_PAYLOAD_CASE( ESPedestals )
      IMPORT_PAYLOAD_CASE( ESRecHitRatioCuts )
      IMPORT_PAYLOAD_CASE( ESThresholds )
      IMPORT_PAYLOAD_CASE( ESTimeSampleWeights )
      IMPORT_PAYLOAD_CASE( EcalADCToGeVConstant )
      IMPORT_PAYLOAD_CASE( EcalChannelStatus )
      IMPORT_PAYLOAD_CASE( EcalClusterEnergyCorrectionObjectSpecificParameters )
      IMPORT_PAYLOAD_CASE( EcalDAQTowerStatus )
      IMPORT_PAYLOAD_CASE( EcalDCSTowerStatus )
      IMPORT_PAYLOAD_CASE( EcalDQMChannelStatus )
      IMPORT_PAYLOAD_CASE( EcalDQMTowerStatus )
      IMPORT_PAYLOAD_CASE( EcalFloatCondObjectContainer )
      IMPORT_PAYLOAD_CASE( EcalFunParams )
      IMPORT_PAYLOAD_CASE( EcalGainRatios )
      IMPORT_PAYLOAD_CASE( EcalLaserAPDPNRatios )
      IMPORT_PAYLOAD_CASE( EcalMappingElectronics )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<EcalMappingElement> )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<EcalPedestal> )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<EcalTPGLinearizationConstant> )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<EcalDQMStatusCode> )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<EcalTPGCrystalStatusCode> )
      IMPORT_PAYLOAD_CASE( EcalCondTowerObjectContainer<EcalDAQStatusCode> )
      IMPORT_PAYLOAD_CASE( EcalCondTowerObjectContainer<EcalChannelStatusCode> )
      IMPORT_PAYLOAD_CASE( EcalCondTowerObjectContainer<EcalDQMStatusCode> )
      IMPORT_PAYLOAD_CASE( EcalPedestals )
      IMPORT_PAYLOAD_CASE( EcalSRSettings )
      IMPORT_PAYLOAD_CASE( EcalSampleMask )
      IMPORT_PAYLOAD_CASE( EcalTBWeights )
      IMPORT_PAYLOAD_CASE( EcalTPGCrystalStatus )
      IMPORT_PAYLOAD_CASE( EcalTPGFineGrainEBGroup )
      IMPORT_PAYLOAD_CASE( EcalTPGFineGrainEBIdMap )
      IMPORT_PAYLOAD_CASE( EcalTPGFineGrainStripEE )
      IMPORT_PAYLOAD_CASE( EcalTPGFineGrainTowerEE )
      IMPORT_PAYLOAD_CASE( EcalTPGLinearizationConst )
      IMPORT_PAYLOAD_CASE( EcalTPGLutGroup )
      IMPORT_PAYLOAD_CASE( EcalTPGLutIdMap )
      IMPORT_PAYLOAD_CASE( EcalTPGPedestals )
      IMPORT_PAYLOAD_CASE( EcalTPGPhysicsConst )
      IMPORT_PAYLOAD_CASE( EcalTPGSlidingWindow )
      IMPORT_PAYLOAD_CASE( EcalTPGSpike )
      IMPORT_PAYLOAD_CASE( EcalTPGStripStatus )
      IMPORT_PAYLOAD_CASE( EcalTPGTowerStatus )
      IMPORT_PAYLOAD_CASE( EcalTPGWeightGroup )
      IMPORT_PAYLOAD_CASE( EcalTPGWeightIdMap )
      IMPORT_PAYLOAD_CASE( EcalTimeOffsetConstant )
      IMPORT_PAYLOAD_CASE( EcalTimeDependentCorrections )
      IMPORT_PAYLOAD_CASE( EcalWeightXtalGroups )
      IMPORT_PAYLOAD_CASE( FileBlob )
      IMPORT_PAYLOAD_CASE( GBRForest )
      IMPORT_PAYLOAD_CASE( HcalChannelQuality )
      IMPORT_PAYLOAD_CASE( HcalCholeskyMatrices )
      IMPORT_PAYLOAD_CASE( HcalDcsValues )
      IMPORT_PAYLOAD_CASE( HcalElectronicsMap )
      IMPORT_PAYLOAD_CASE( HcalFlagHFDigiTimeParams )
      IMPORT_PAYLOAD_CASE( HcalGains )
      IMPORT_PAYLOAD_CASE( HcalGainWidths )
      IMPORT_PAYLOAD_CASE( HcalL1TriggerObjects )
      IMPORT_PAYLOAD_CASE( HcalLUTCorrs )
      IMPORT_PAYLOAD_CASE( HcalLongRecoParams )
      IMPORT_PAYLOAD_CASE( HcalLutMetadata )
      IMPORT_PAYLOAD_CASE( HcalMCParams )
      IMPORT_PAYLOAD_CASE( HcalPFCorrs )
      IMPORT_PAYLOAD_CASE( HcalPedestalWidths )
      IMPORT_PAYLOAD_CASE( HcalPedestals )
      IMPORT_PAYLOAD_CASE( HcalQIEData )
      IMPORT_PAYLOAD_CASE( HcalRecoParams )
      IMPORT_PAYLOAD_CASE( HcalRespCorrs )
      IMPORT_PAYLOAD_CASE( HcalTimeCorrs )
      IMPORT_PAYLOAD_CASE( HcalZSThresholds )
      IMPORT_PAYLOAD_CASE( JetCorrectorParametersCollection )
      IMPORT_PAYLOAD_CASE( L1CaloEcalScale )
      IMPORT_PAYLOAD_CASE( L1CaloEtScale )
      IMPORT_PAYLOAD_CASE( L1CaloGeometry )
      IMPORT_PAYLOAD_CASE( L1CaloHcalScale )
      IMPORT_PAYLOAD_CASE( L1GctChannelMask )
      IMPORT_PAYLOAD_CASE( L1GctJetFinderParams )
      IMPORT_PAYLOAD_CASE( L1GtBoardMaps )
      IMPORT_PAYLOAD_CASE( L1GtParameters )
      IMPORT_PAYLOAD_CASE( L1GtPrescaleFactors )
      IMPORT_PAYLOAD_CASE( L1GtPsbSetup )
      IMPORT_PAYLOAD_CASE( L1GtStableParameters )
      IMPORT_PAYLOAD_CASE( L1GtTriggerMask )
      IMPORT_PAYLOAD_CASE( L1GtTriggerMenu )
      IMPORT_PAYLOAD_CASE( L1MuCSCPtLut )
      IMPORT_PAYLOAD_CASE( L1MuCSCTFAlignment )
      IMPORT_PAYLOAD_CASE( L1MuCSCTFConfiguration )
      IMPORT_PAYLOAD_CASE( L1MuDTEtaPatternLut )
      IMPORT_PAYLOAD_CASE( L1MuDTExtLut )
      IMPORT_PAYLOAD_CASE( L1MuDTPhiLut )
      IMPORT_PAYLOAD_CASE( L1MuDTPtaLut )
      IMPORT_PAYLOAD_CASE( L1MuDTQualPatternLut )
      IMPORT_PAYLOAD_CASE( L1MuDTTFMasks )
      IMPORT_PAYLOAD_CASE( L1MuDTTFParameters )
      IMPORT_PAYLOAD_CASE( L1MuGMTChannelMask )
      IMPORT_PAYLOAD_CASE( L1MuGMTParameters )
      IMPORT_PAYLOAD_CASE( L1MuGMTScales )
      IMPORT_PAYLOAD_CASE( L1MuTriggerPtScale )
      IMPORT_PAYLOAD_CASE( L1MuTriggerScales )
      IMPORT_PAYLOAD_CASE( L1RCTChannelMask )
      IMPORT_PAYLOAD_CASE( L1RCTNoisyChannelMask )
      IMPORT_PAYLOAD_CASE( L1RCTParameters )
      IMPORT_PAYLOAD_CASE( L1RPCBxOrConfig )
      IMPORT_PAYLOAD_CASE( L1RPCConeDefinition )
      IMPORT_PAYLOAD_CASE( L1RPCConfig )
      IMPORT_PAYLOAD_CASE( L1RPCHsbConfig ) 
      IMPORT_PAYLOAD_CASE( L1RPCHwConfig )
      IMPORT_PAYLOAD_CASE( L1TriggerKey )
      IMPORT_PAYLOAD_CASE( L1TriggerKeyList )
      IMPORT_PAYLOAD_CASE( lumi::LumiSectionData )
      IMPORT_PAYLOAD_CASE( MixingModuleConfig )
      IMPORT_PAYLOAD_CASE( MuScleFitDBobject )
      /**
      if( inputTypeName == "PhysicsTools::Calibration::MVAComputerContainer" ){ \
      std::cout <<"@@@@@ MVAComputer!"<<std::endl;\
      match = true;\
      const PhysicsTools::Calibration::MVAComputerContainer& obj = *static_cast<const PhysicsTools::Calibration::MVAComputerContainer*>( inputPtr ); \
      PhysicsTools::Calibration::MVAComputerContainer tmp;		\
      for( auto entry : obj.entries ) {					\
      std::cout <<"#Adding new entry label="<<entry.first<<std::endl;	\
      PhysicsTools::Calibration::MVAComputer& c = tmp.add( entry.first ); \
      c.inputSet = entry.second.inputSet;				\
      c.output =  entry.second.output;					\
      auto ps = entry.second.getProcessors();				\
      for( size_t i=0;i<ps.size();i++ ){				\
      std::cout <<"PRocess type="<<demangledName( typeid(*ps[i] ) )<<std::endl;	\
      c.addProcessor( ps[i] );						\
      }									\
      }									\
      std::pair<std::string,bool> st = destination.storePayload( tmp, boost::posix_time::microsec_clock::universal_time() ); \
      payloadId = st.first;						\
      newInsert = st.second;						\
      } 
      **/

      IMPORT_PAYLOAD_CASE( PhysicsTools::Calibration::MVAComputerContainer )
      IMPORT_PAYLOAD_CASE( PCaloGeometry )
      IMPORT_PAYLOAD_CASE( PGeometricDet )
      IMPORT_PAYLOAD_CASE( PGeometricDetExtra )
	//IMPORT_PAYLOAD_CASE( PerformancePayload )
      IMPORT_PAYLOAD_CASE( PerformancePayloadFromTable )
      IMPORT_PAYLOAD_CASE( PerformancePayloadFromTFormula )
      IMPORT_PAYLOAD_CASE( PerformancePayloadFromBinnedTFormula )
      IMPORT_PAYLOAD_CASE( PerformanceWorkingPoint )
      IMPORT_PAYLOAD_CASE( PhysicsTools::Calibration::HistogramD3D )
      IMPORT_PAYLOAD_CASE( RPCEMap )
      IMPORT_PAYLOAD_CASE( RPCClusterSize )
      IMPORT_PAYLOAD_CASE( RPCStripNoises )
      IMPORT_PAYLOAD_CASE( RPCObFebmap )
      IMPORT_PAYLOAD_CASE( RPCObGas )
      IMPORT_PAYLOAD_CASE( RPCObImon )
      IMPORT_PAYLOAD_CASE( RPCObGasMix )
      IMPORT_PAYLOAD_CASE( RPCObPVSSmap )
      IMPORT_PAYLOAD_CASE( RPCObStatus )
      IMPORT_PAYLOAD_CASE( RPCObTemp )
      IMPORT_PAYLOAD_CASE( RPCObUXC )
      IMPORT_PAYLOAD_CASE( RPCObVmon )
      IMPORT_PAYLOAD_CASE( RecoIdealGeometry )
      IMPORT_PAYLOAD_CASE( RunInfo )
      IMPORT_PAYLOAD_CASE( SiPixelCalibConfiguration )
      IMPORT_PAYLOAD_CASE( SiPixelCPEGenericErrorParm )
      IMPORT_PAYLOAD_CASE( SiPixelFedCablingMap )
      IMPORT_PAYLOAD_CASE( SiPixelGainCalibrationForHLT )
      IMPORT_PAYLOAD_CASE( SiPixelGainCalibrationOffline )
      IMPORT_PAYLOAD_CASE( SiPixelLorentzAngle )
      IMPORT_PAYLOAD_CASE( SiPixelQuality )
      IMPORT_PAYLOAD_CASE( SiPixelTemplateDBObject )
      IMPORT_PAYLOAD_CASE( SiStripApvGain )
      IMPORT_PAYLOAD_CASE( SiStripBadStrip )
      IMPORT_PAYLOAD_CASE( SiStripBackPlaneCorrection )
      IMPORT_PAYLOAD_CASE( SiStripConfObject )
      IMPORT_PAYLOAD_CASE( SiStripDetVOff )
      IMPORT_PAYLOAD_CASE( SiStripFedCabling )
      IMPORT_PAYLOAD_CASE( SiStripLatency )
      IMPORT_PAYLOAD_CASE( SiStripLorentzAngle )
      IMPORT_PAYLOAD_CASE( SiStripNoises )
      IMPORT_PAYLOAD_CASE( SiStripPedestals )
      IMPORT_PAYLOAD_CASE( SiStripThreshold )
      IMPORT_PAYLOAD_CASE( TrackProbabilityCalibration )
      IMPORT_PAYLOAD_CASE( cond::BaseKeyed )
      IMPORT_PAYLOAD_CASE( ESCondObjectContainer<ESChannelStatusCode> )
      IMPORT_PAYLOAD_CASE( ESCondObjectContainer<ESPedestal> )
      IMPORT_PAYLOAD_CASE( ESCondObjectContainer<float> )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<EcalChannelStatusCode> )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<EcalMGPAGainRatio> )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<EcalTPGPedestal> )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<EcalXtalGroupId> )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<float> )
      if( inputTypeName == "PhysicsTools::Calibration::Histogram3D<double,double,double,double>" ){
	match = true;
	const PhysicsTools::Calibration::Histogram3D<double,double,double,double>& obj = *static_cast<const PhysicsTools::Calibration::Histogram3D<double,double,double,double>*>( inputPtr ); 
	payloadId = destination.storePayload( obj, boost::posix_time::microsec_clock::universal_time() ); 
      } 
      if( inputTypeName == "PhysicsTools::Calibration::Histogram2D<double,double,double>" ){
	match = true;
	const PhysicsTools::Calibration::Histogram2D<double,double,double>& obj = *static_cast<const PhysicsTools::Calibration::Histogram2D<double,double,double>*>( inputPtr ); 
	payloadId = destination.storePayload( obj, boost::posix_time::microsec_clock::universal_time() ); 
      } 

      
      if( ! match ) throwException( "Payload type \""+inputTypeName+"\" is unknown.","import" );
      }
      return payloadId;
    }

    std::pair<std::string,boost::shared_ptr<void> > fetch( const cond::Hash& payloadId, Session& session ){
      boost::shared_ptr<void> payloadPtr;
      cond::Binary data;
      cond::Binary streamerInfo;
      std::string payloadTypeName;
      bool found = session.fetchPayloadData( payloadId, payloadTypeName, data, streamerInfo );
      if( !found ) throwException( "Payload with id "+boost::lexical_cast<std::string>(payloadId)+" has not been found in the database.","fetchAndCompare" );
      //std::cout <<"--> payload type "<<payloadTypeName<<" has blob size "<<data.size()<<std::endl;
      bool match = false;
      bool isOra = session.isOraSession();
    FETCH_PAYLOAD_CASE( std::string ) 
    FETCH_PAYLOAD_CASE( std::vector<unsigned long long> )
    FETCH_PAYLOAD_CASE( AlCaRecoTriggerBits )
    FETCH_PAYLOAD_CASE( AlignmentErrors )
    FETCH_PAYLOAD_CASE( AlignmentSurfaceDeformations )
    FETCH_PAYLOAD_CASE( Alignments )
    FETCH_PAYLOAD_CASE( BeamSpotObjects )
    FETCH_PAYLOAD_CASE( CSCBadChambers )
    FETCH_PAYLOAD_CASE( CSCBadStrips )
    FETCH_PAYLOAD_CASE( CSCBadWires )
    FETCH_PAYLOAD_CASE( CSCChamberIndex )
    FETCH_PAYLOAD_CASE( CSCChamberMap )
    FETCH_PAYLOAD_CASE( CSCChamberTimeCorrections )
    FETCH_PAYLOAD_CASE( CSCCrateMap )
    FETCH_PAYLOAD_CASE( CSCDBChipSpeedCorrection )
    FETCH_PAYLOAD_CASE( CSCDBCrosstalk )
    FETCH_PAYLOAD_CASE( CSCDBGains )
    FETCH_PAYLOAD_CASE( CSCDBGasGainCorrection )
    FETCH_PAYLOAD_CASE( CSCDBL1TPParameters )
    FETCH_PAYLOAD_CASE( CSCDBNoiseMatrix )
    FETCH_PAYLOAD_CASE( CSCDBPedestals )
    FETCH_PAYLOAD_CASE( CSCDDUMap )
    FETCH_PAYLOAD_CASE( CSCL1TPParameters )
    FETCH_PAYLOAD_CASE( CSCRecoDigiParameters )
    FETCH_PAYLOAD_CASE( CastorChannelQuality )
    FETCH_PAYLOAD_CASE( CastorElectronicsMap )
    FETCH_PAYLOAD_CASE( CastorGainWidths )
    FETCH_PAYLOAD_CASE( CastorGains )
    FETCH_PAYLOAD_CASE( CastorPedestalWidths )
    FETCH_PAYLOAD_CASE( CastorPedestals )
    FETCH_PAYLOAD_CASE( CastorQIEData )
    FETCH_PAYLOAD_CASE( CastorRecoParams )
    FETCH_PAYLOAD_CASE( CastorSaturationCorrs )
    FETCH_PAYLOAD_CASE( CentralityTable )
    FETCH_PAYLOAD_CASE( DTCCBConfig )
    FETCH_PAYLOAD_CASE( DTDeadFlag )
    FETCH_PAYLOAD_CASE( DTHVStatus )
    FETCH_PAYLOAD_CASE( DTKeyedConfig )
    FETCH_PAYLOAD_CASE( DTLVStatus )
    FETCH_PAYLOAD_CASE( DTMtime )
    FETCH_PAYLOAD_CASE( DTReadOutMapping )
    FETCH_PAYLOAD_CASE( DTStatusFlag )
    FETCH_PAYLOAD_CASE( DTT0 )
    FETCH_PAYLOAD_CASE( DTTPGParameters )
    FETCH_PAYLOAD_CASE( DTTtrig )
    FETCH_PAYLOAD_CASE( DropBoxMetadata )
    FETCH_PAYLOAD_CASE( ESChannelStatus )
    FETCH_PAYLOAD_CASE( ESEEIntercalibConstants )
    FETCH_PAYLOAD_CASE( ESFloatCondObjectContainer )
    FETCH_PAYLOAD_CASE( ESGain )
    FETCH_PAYLOAD_CASE( ESMIPToGeVConstant )
    FETCH_PAYLOAD_CASE( ESMissingEnergyCalibration )
    FETCH_PAYLOAD_CASE( ESPedestals )
    FETCH_PAYLOAD_CASE( ESRecHitRatioCuts )
    FETCH_PAYLOAD_CASE( ESThresholds )
    FETCH_PAYLOAD_CASE( ESTimeSampleWeights )
    FETCH_PAYLOAD_CASE( EcalADCToGeVConstant )
    FETCH_PAYLOAD_CASE( EcalChannelStatus )
    FETCH_PAYLOAD_CASE( EcalClusterEnergyCorrectionObjectSpecificParameters )
    FETCH_PAYLOAD_CASE( EcalDAQTowerStatus )
    FETCH_PAYLOAD_CASE( EcalDCSTowerStatus )
    FETCH_PAYLOAD_CASE( EcalDQMChannelStatus )
    FETCH_PAYLOAD_CASE( EcalDQMTowerStatus )
    FETCH_PAYLOAD_CASE( EcalFloatCondObjectContainer )
    FETCH_PAYLOAD_CASE( EcalFunParams )
    FETCH_PAYLOAD_CASE( EcalGainRatios )
    FETCH_PAYLOAD_CASE( EcalLaserAPDPNRatios )
    FETCH_PAYLOAD_CASE( EcalMappingElectronics )
    FETCH_PAYLOAD_CASE( EcalCondObjectContainer<EcalMappingElement> )
    FETCH_PAYLOAD_CASE( EcalCondObjectContainer<EcalPedestal> )
    FETCH_PAYLOAD_CASE( EcalCondObjectContainer<EcalTPGLinearizationConstant> )
    FETCH_PAYLOAD_CASE( EcalCondObjectContainer<EcalDQMStatusCode> )
    FETCH_PAYLOAD_CASE( EcalCondObjectContainer<EcalTPGCrystalStatusCode> )
    FETCH_PAYLOAD_CASE( EcalCondTowerObjectContainer<EcalDAQStatusCode> )
    FETCH_PAYLOAD_CASE( EcalCondTowerObjectContainer<EcalChannelStatusCode> )
    FETCH_PAYLOAD_CASE( EcalCondTowerObjectContainer<EcalDQMStatusCode> )
    FETCH_PAYLOAD_CASE( EcalPedestals )
    FETCH_PAYLOAD_CASE( EcalSRSettings )
    FETCH_PAYLOAD_CASE( EcalSampleMask )
    FETCH_PAYLOAD_CASE( EcalTBWeights )
    FETCH_PAYLOAD_CASE( EcalTimeDependentCorrections )
    FETCH_PAYLOAD_CASE( EcalTPGCrystalStatus )
    FETCH_PAYLOAD_CASE( EcalTPGFineGrainEBGroup )
    FETCH_PAYLOAD_CASE( EcalTPGFineGrainEBIdMap )
    FETCH_PAYLOAD_CASE( EcalTPGFineGrainStripEE )
    FETCH_PAYLOAD_CASE( EcalTPGFineGrainTowerEE )
    FETCH_PAYLOAD_CASE( EcalTPGLinearizationConst )
    FETCH_PAYLOAD_CASE( EcalTPGLutGroup )
    FETCH_PAYLOAD_CASE( EcalTPGLutIdMap )
    FETCH_PAYLOAD_CASE( EcalTPGPedestals )
    FETCH_PAYLOAD_CASE( EcalTPGPhysicsConst )
    FETCH_PAYLOAD_CASE( EcalTPGSlidingWindow )
    FETCH_PAYLOAD_CASE( EcalTPGSpike )
    FETCH_PAYLOAD_CASE( EcalTPGStripStatus )
    FETCH_PAYLOAD_CASE( EcalTPGTowerStatus )
    FETCH_PAYLOAD_CASE( EcalTPGWeightGroup )
    FETCH_PAYLOAD_CASE( EcalTPGWeightIdMap )
    FETCH_PAYLOAD_CASE( EcalTimeOffsetConstant )
    FETCH_PAYLOAD_CASE( EcalWeightXtalGroups )
    FETCH_PAYLOAD_CASE( FileBlob )
    FETCH_PAYLOAD_CASE( GBRForest )
    FETCH_PAYLOAD_CASE( HcalChannelQuality )
    FETCH_PAYLOAD_CASE( HcalCholeskyMatrices )
    FETCH_PAYLOAD_CASE( HcalElectronicsMap )
    FETCH_PAYLOAD_CASE( HcalFlagHFDigiTimeParams )
    FETCH_PAYLOAD_CASE( HcalDcsValues )
    FETCH_PAYLOAD_CASE( HcalGains )
    FETCH_PAYLOAD_CASE( HcalGainWidths )
    FETCH_PAYLOAD_CASE( HcalL1TriggerObjects )
    FETCH_PAYLOAD_CASE( HcalLUTCorrs )
    FETCH_PAYLOAD_CASE( HcalLongRecoParams )
    FETCH_PAYLOAD_CASE( HcalLutMetadata )
    FETCH_PAYLOAD_CASE( HcalMCParams )
    FETCH_PAYLOAD_CASE( HcalPFCorrs )
    FETCH_PAYLOAD_CASE( HcalPedestalWidths )
    FETCH_PAYLOAD_CASE( HcalPedestals )
    FETCH_PAYLOAD_CASE( HcalQIEData )
    FETCH_PAYLOAD_CASE( HcalRecoParams )
    FETCH_PAYLOAD_CASE( HcalRespCorrs )
    FETCH_PAYLOAD_CASE( HcalTimeCorrs )
    FETCH_PAYLOAD_CASE( HcalZSThresholds )
    FETCH_PAYLOAD_CASE( JetCorrectorParametersCollection )
    FETCH_PAYLOAD_CASE( L1CaloEcalScale )
    FETCH_PAYLOAD_CASE( L1CaloEtScale )
    FETCH_PAYLOAD_CASE( L1CaloGeometry )
    FETCH_PAYLOAD_CASE( L1CaloHcalScale )
    FETCH_PAYLOAD_CASE( L1GctChannelMask )
    FETCH_PAYLOAD_CASE( L1GctJetFinderParams )
    FETCH_PAYLOAD_CASE( L1GtBoardMaps )
    FETCH_PAYLOAD_CASE( L1GtParameters )
    FETCH_PAYLOAD_CASE( L1GtPrescaleFactors )
    FETCH_PAYLOAD_CASE( L1GtPsbSetup )
    FETCH_PAYLOAD_CASE( L1GtStableParameters )
    FETCH_PAYLOAD_CASE( L1GtTriggerMask )
    FETCH_PAYLOAD_CASE( L1GtTriggerMenu )
    FETCH_PAYLOAD_CASE( L1MuCSCPtLut )
    FETCH_PAYLOAD_CASE( L1MuCSCTFAlignment )
    FETCH_PAYLOAD_CASE( L1MuCSCTFConfiguration )
    FETCH_PAYLOAD_CASE( L1MuDTEtaPatternLut )
    FETCH_PAYLOAD_CASE( L1MuDTExtLut )
    FETCH_PAYLOAD_CASE( L1MuDTPhiLut )
    FETCH_PAYLOAD_CASE( L1MuDTPtaLut )
    FETCH_PAYLOAD_CASE( L1MuDTQualPatternLut )
    FETCH_PAYLOAD_CASE( L1MuDTTFMasks )
    FETCH_PAYLOAD_CASE( L1MuDTTFParameters )
    FETCH_PAYLOAD_CASE( L1MuGMTChannelMask )
    FETCH_PAYLOAD_CASE( L1MuGMTParameters )
    FETCH_PAYLOAD_CASE( L1MuGMTScales )
    FETCH_PAYLOAD_CASE( L1MuTriggerPtScale )
    FETCH_PAYLOAD_CASE( L1MuTriggerScales )
    FETCH_PAYLOAD_CASE( L1RCTChannelMask )
    FETCH_PAYLOAD_CASE( L1RCTNoisyChannelMask )
    FETCH_PAYLOAD_CASE( L1RCTParameters )
    FETCH_PAYLOAD_CASE( L1RPCBxOrConfig )
    FETCH_PAYLOAD_CASE( L1RPCConeDefinition )
    FETCH_PAYLOAD_CASE( L1RPCConfig )
    FETCH_PAYLOAD_CASE( L1RPCHsbConfig ) 
    FETCH_PAYLOAD_CASE( L1RPCHwConfig )
    FETCH_PAYLOAD_CASE( L1TriggerKey )
    FETCH_PAYLOAD_CASE( L1TriggerKeyList )
    FETCH_PAYLOAD_CASE( lumi::LumiSectionData )
    FETCH_PAYLOAD_CASE( MixingModuleConfig )
    FETCH_PAYLOAD_CASE( MuScleFitDBobject )
    FETCH_PAYLOAD_CASE( PhysicsTools::Calibration::MVAComputerContainer )
    FETCH_PAYLOAD_CASE( PCaloGeometry )
    FETCH_PAYLOAD_CASE( PGeometricDet )
    FETCH_PAYLOAD_CASE( PGeometricDetExtra )
      //FETCH_PAYLOAD_CASE( PerformancePayload )
    FETCH_PAYLOAD_CASE( PerformancePayloadFromTable )
    FETCH_PAYLOAD_CASE( PerformancePayloadFromTFormula )
    FETCH_PAYLOAD_CASE( PerformancePayloadFromBinnedTFormula )
    FETCH_PAYLOAD_CASE( PerformanceWorkingPoint )
    FETCH_PAYLOAD_CASE( PhysicsTools::Calibration::HistogramD3D )
    FETCH_PAYLOAD_CASE( RPCEMap )
    FETCH_PAYLOAD_CASE( RPCClusterSize )
    FETCH_PAYLOAD_CASE( RPCStripNoises )
    FETCH_PAYLOAD_CASE( RPCObFebmap )
    FETCH_PAYLOAD_CASE( RPCObGas )
    FETCH_PAYLOAD_CASE( RPCObImon )
    FETCH_PAYLOAD_CASE( RPCObGasMix )
    FETCH_PAYLOAD_CASE( RPCObPVSSmap )
    FETCH_PAYLOAD_CASE( RPCObStatus )
    FETCH_PAYLOAD_CASE( RPCObTemp )
    FETCH_PAYLOAD_CASE( RPCObUXC )
    FETCH_PAYLOAD_CASE( RPCObVmon )
    FETCH_PAYLOAD_CASE( RecoIdealGeometry )
    FETCH_PAYLOAD_CASE( RunInfo )
    FETCH_PAYLOAD_CASE( SiPixelCalibConfiguration )
    FETCH_PAYLOAD_CASE( SiPixelCPEGenericErrorParm )
    FETCH_PAYLOAD_CASE( SiPixelFedCablingMap )
    FETCH_PAYLOAD_CASE( SiPixelGainCalibrationForHLT )
    FETCH_PAYLOAD_CASE( SiPixelGainCalibrationOffline )
    FETCH_PAYLOAD_CASE( SiPixelLorentzAngle )
    FETCH_PAYLOAD_CASE( SiPixelQuality )
    FETCH_PAYLOAD_CASE( SiPixelTemplateDBObject )
    FETCH_PAYLOAD_CASE( SiStripApvGain )
    FETCH_PAYLOAD_CASE( SiStripBackPlaneCorrection )
    FETCH_PAYLOAD_CASE( SiStripBadStrip )
    FETCH_PAYLOAD_CASE( SiStripConfObject )
    FETCH_PAYLOAD_CASE( SiStripDetVOff )
    FETCH_PAYLOAD_CASE( SiStripFedCabling )
    FETCH_PAYLOAD_CASE( SiStripLatency )
    FETCH_PAYLOAD_CASE( SiStripLorentzAngle )
    FETCH_PAYLOAD_CASE( SiStripNoises )
    FETCH_PAYLOAD_CASE( SiStripPedestals )
    FETCH_PAYLOAD_CASE( SiStripThreshold )
    FETCH_PAYLOAD_CASE( TrackProbabilityCalibration )
    FETCH_PAYLOAD_CASE( cond::BaseKeyed )
    FETCH_PAYLOAD_CASE( ESCondObjectContainer<ESChannelStatusCode> )
    FETCH_PAYLOAD_CASE( ESCondObjectContainer<ESPedestal> )
    FETCH_PAYLOAD_CASE( ESCondObjectContainer<float> )
    FETCH_PAYLOAD_CASE( EcalCondObjectContainer<EcalChannelStatusCode> )
    FETCH_PAYLOAD_CASE( EcalCondObjectContainer<EcalMGPAGainRatio> )
    FETCH_PAYLOAD_CASE( EcalCondObjectContainer<EcalTPGPedestal> )
    FETCH_PAYLOAD_CASE( EcalCondObjectContainer<EcalXtalGroupId> )
    FETCH_PAYLOAD_CASE( EcalCondObjectContainer<float> )

    //   
    if( payloadTypeName == "PhysicsTools::Calibration::Histogram3D<double,double,double,double>" ){    
      auto payload = deserialize<PhysicsTools::Calibration::Histogram3D<double,double,double,double> >(payloadTypeName, data, streamerInfo );
      payloadPtr = payload;
      match = true;
    }
    if( payloadTypeName == "PhysicsTools::Calibration::Histogram2D<double,double,double>" ){    
      auto payload = deserialize<PhysicsTools::Calibration::Histogram2D<double,double,double> >(payloadTypeName, data, streamerInfo );
      payloadPtr = payload;
      match = true;
    }
  

    if( ! match ) throwException( "Payload type \""+payloadTypeName+"\" is unknown.","fetch" );
    return std::make_pair( payloadTypeName, payloadPtr );
  }

 }
}

