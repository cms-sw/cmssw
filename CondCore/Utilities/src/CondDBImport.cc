//

//TO DO: add the comparison between stored object and input object - bool cond::serialization::equal( obj,read_back_copy )
#define IMPORT_PAYLOAD_CASE( TYPENAME )  \
  if( inputTypeName == #TYPENAME ){ \
    match = true; \
    const TYPENAME& obj = *static_cast<const TYPENAME*>( inputPtr ); \
    payloadId = destination.storePayload( obj, boost::posix_time::microsec_clock::universal_time() ); \
  } 

#include "CondCore/CondDB/interface/Serialization.h"

#include "CondCore/Utilities/interface/CondDBImport.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondFormats.h"

//
#include <sstream>

namespace cond {

  namespace persistency {

    std::pair<std::string,boost::shared_ptr<void> > fetchIfExists( const cond::Hash& payloadId, Session& session, bool& exists ){
      boost::shared_ptr<void> payloadPtr;
      cond::Binary data;
      cond::Binary streamerInfo;
      std::string payloadTypeName;
      exists = session.fetchPayloadData( payloadId, payloadTypeName, data, streamerInfo );
      if( exists ) {
	bool isOra = session.isOraSession();
	return fetchOne(payloadTypeName, data, streamerInfo, payloadPtr, isOra);
      } else return std::make_pair( std::string(""), boost::shared_ptr<void>() );
    }

    cond::Hash import( Session& source, const cond::Hash& sourcePayloadId, const std::string& inputTypeName, const void* inputPtr, Session& destination ){
      cond::Hash payloadId("");
      bool newInsert = false;
      bool match = false;
      if( inputPtr ){
      IMPORT_PAYLOAD_CASE( std::string ) 
      IMPORT_PAYLOAD_CASE( std::vector<unsigned long long> )
      IMPORT_PAYLOAD_CASE( AlCaRecoTriggerBits )
      IMPORT_PAYLOAD_CASE( AlignmentErrors )
      IMPORT_PAYLOAD_CASE( AlignmentErrorsExtended )
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
      IMPORT_PAYLOAD_CASE( DTRecoConditions )
      IMPORT_PAYLOAD_CASE( DTRecoUncertainties )
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
      IMPORT_PAYLOAD_CASE( EcalTimeBiasCorrections )
      IMPORT_PAYLOAD_CASE( EcalTimeOffsetConstant )
      IMPORT_PAYLOAD_CASE( EcalTimeDependentCorrections )
      IMPORT_PAYLOAD_CASE( EcalWeightXtalGroups )
      IMPORT_PAYLOAD_CASE( EcalSamplesCorrelation )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<EcalPulseShape> )
      IMPORT_PAYLOAD_CASE( EcalPulseShape )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<EcalPulseCovariance> )
      IMPORT_PAYLOAD_CASE( EcalPulseCovariance )
      IMPORT_PAYLOAD_CASE( EcalCondObjectContainer<EcalPulseSymmCovariance> )
      IMPORT_PAYLOAD_CASE( EcalPulseSymmCovariance )
      IMPORT_PAYLOAD_CASE( FileBlob )
      IMPORT_PAYLOAD_CASE( GBRForest )
      IMPORT_PAYLOAD_CASE( GBRForestD )
      IMPORT_PAYLOAD_CASE( HBHENegativeEFilter )	
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
      IMPORT_PAYLOAD_CASE( HcalZDCLowGainFractions )
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
      IMPORT_PAYLOAD_CASE( HcalInterpolatedPulseColl )
      IMPORT_PAYLOAD_CASE( JetCorrectorParametersCollection )
      IMPORT_PAYLOAD_CASE( JME::JetResolutionObject )
      IMPORT_PAYLOAD_CASE( METCorrectorParametersCollection )
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
      IMPORT_PAYLOAD_CASE( l1t::CaloParams )
      IMPORT_PAYLOAD_CASE( l1t::CaloConfig )
      IMPORT_PAYLOAD_CASE( L1TriggerKey )
      IMPORT_PAYLOAD_CASE( MagFieldConfig )
      if( inputTypeName == "L1TriggerKeyList" ){ 
	match = true;
	const L1TriggerKeyList& obj = *static_cast<const L1TriggerKeyList*>( inputPtr );
        L1TriggerKeyList converted;
	for( const auto& kitem : obj.tscKeyToTokenMap() ){
	  std::string pid("0");
	  std::string sourcePid = source.parsePoolToken( kitem.second );
	  if( !destination.lookupMigratedPayload( source.connectionString(), sourcePid, pid ) ){
	    std::cout <<"WARNING: L1Trigger key stored on "<<sourcePid<<" has not been migrated (yet?). Attemping to do the export..."<<std::endl;
	    bool exists = false;
            std::pair<std::string,boost::shared_ptr<void> > missingPayload = fetchIfExists( sourcePid, source, exists );
	    if( exists ) pid = import( source, sourcePid, missingPayload.first, missingPayload.second.get(), destination );
	    std::cout <<"WARNING: OID "<<sourcePid<<" will be mapped to HASH "<<pid<<std::endl;
	    if( pid != "0" ) destination.addMigratedPayload( source.connectionString(), sourcePid, pid );
	  }
          converted.addKey( kitem.first, pid );
	}
	for( const auto& ritem : obj.recordTypeToKeyToTokenMap() ){
	  for( const auto& kitem : ritem.second ){
	    std::string pid("0");
	    std::string sourcePid = source.parsePoolToken( kitem.second );
	    if( !destination.lookupMigratedPayload( source.connectionString(), sourcePid, pid ) ){
	      std::cout <<"WARNING: L1Trigger key stored on "<<sourcePid<<" has not been migrated (yet?). Attemping to do the export..."<<std::endl;
	      bool exists = false;
	      std::pair<std::string,boost::shared_ptr<void> > missingPayload = fetchIfExists( sourcePid, source, exists );
	      if( exists ) pid = import( source, sourcePid, missingPayload.first, missingPayload.second.get(), destination );
	      std::cout <<"WARNING: OID "<<sourcePid<<" will be mapped to HASH "<<pid<<std::endl;
	      if( pid != "0" ) destination.addMigratedPayload( source.connectionString(), sourcePid, pid );
	    }
	    converted.addKey( ritem.first, kitem.first, pid );
	  }
	}
	payloadId = destination.storePayload( converted, boost::posix_time::microsec_clock::universal_time() );
      }
      //IMPORT_PAYLOAD_CASE( L1TriggerKeyList )
      IMPORT_PAYLOAD_CASE( lumi::LumiSectionData )
      IMPORT_PAYLOAD_CASE( MixingModuleConfig )
      IMPORT_PAYLOAD_CASE( MuScleFitDBobject )
      IMPORT_PAYLOAD_CASE( DYTThrObject )
      IMPORT_PAYLOAD_CASE( DYTParamObject )
      IMPORT_PAYLOAD_CASE( OOTPileupCorrectionBuffer )
      IMPORT_PAYLOAD_CASE( StorableDoubleMap<AbsOOTPileupCorrection> )
      IMPORT_PAYLOAD_CASE( PhysicsTools::Calibration::MVAComputerContainer )
      IMPORT_PAYLOAD_CASE( PCaloGeometry )
      IMPORT_PAYLOAD_CASE( PHcalParameters )
      IMPORT_PAYLOAD_CASE( HcalParameters )
      IMPORT_PAYLOAD_CASE( PGeometricDet )
      IMPORT_PAYLOAD_CASE( PGeometricDetExtra )
      IMPORT_PAYLOAD_CASE( PTrackerParameters )
	//IMPORT_PAYLOAD_CASE( PerformancePayload )
      IMPORT_PAYLOAD_CASE( PerformancePayloadFromTable )
      IMPORT_PAYLOAD_CASE( PerformancePayloadFromTFormula )
      IMPORT_PAYLOAD_CASE( PerformancePayloadFromBinnedTFormula )
      IMPORT_PAYLOAD_CASE( PerformanceWorkingPoint )
      IMPORT_PAYLOAD_CASE( PhysicsTGraphPayload )
      IMPORT_PAYLOAD_CASE( PhysicsTFormulaPayload )
      IMPORT_PAYLOAD_CASE( PhysicsTools::Calibration::HistogramD3D )
      IMPORT_PAYLOAD_CASE( QGLikelihoodCategory                     )
      IMPORT_PAYLOAD_CASE( QGLikelihoodObject               )
      IMPORT_PAYLOAD_CASE( QGLikelihoodSystematicsObject     )
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
      IMPORT_PAYLOAD_CASE( RPFlatParams )
      IMPORT_PAYLOAD_CASE( RecoIdealGeometry )
      IMPORT_PAYLOAD_CASE( RunInfo )
      IMPORT_PAYLOAD_CASE( SiPixelCalibConfiguration )
      IMPORT_PAYLOAD_CASE( SiPixelCPEGenericErrorParm )
      IMPORT_PAYLOAD_CASE( SiPixelFedCablingMap )
      IMPORT_PAYLOAD_CASE( SiPixelGainCalibrationForHLT )
      IMPORT_PAYLOAD_CASE( SiPixelGainCalibrationOffline )
      IMPORT_PAYLOAD_CASE( SiPixelGenErrorDBObject )
      IMPORT_PAYLOAD_CASE( SiPixelLorentzAngle )
      IMPORT_PAYLOAD_CASE( SiPixelDynamicInefficiency )
      IMPORT_PAYLOAD_CASE( SiPixelQuality )
      IMPORT_PAYLOAD_CASE( SiPixelTemplateDBObject )
      IMPORT_PAYLOAD_CASE( SiPixel2DTemplateDBObject )
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
      if( inputTypeName == "std::vector<unsignedlonglong,std::allocator<unsignedlonglong>>" ){
	match = true;
	const std::vector<unsigned long long>& obj = *static_cast<const std::vector<unsigned long long>*>( inputPtr );
	payloadId = destination.storePayload( obj, boost::posix_time::microsec_clock::universal_time() );
      }
      
      if( ! match ) throwException( "Payload type \""+inputTypeName+"\" is unknown.","import" );
      }
      return payloadId;
    }

 }
}

