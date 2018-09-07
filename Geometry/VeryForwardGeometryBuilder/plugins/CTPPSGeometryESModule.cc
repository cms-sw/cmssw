/****************************************************************************
*
* Authors:
*  Jan Kaspar (jan.kaspar@gmail.com)
*  Dominik Mierzejewski <dmierzej@cern.ch>
*
****************************************************************************/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSDDDNames.h"

#include <regex>

/**
 * \brief Builds ideal, real and misaligned geometries.
 *
 * First, it creates a tree of DetGeomDesc from DDCompView. For real and misaligned geometries,
 * it applies alignment corrections (RPAlignmentCorrections) found in corresponding ...GeometryRecord.
 *
 * Second, it creates CTPPSGeometry from DetGeoDesc tree.
 **/
class CTPPSGeometryESModule : public edm::ESProducer
{
  public:
    CTPPSGeometryESModule( const edm::ParameterSet& );
    ~CTPPSGeometryESModule() override {}

    std::unique_ptr<DetGeomDesc> produceIdealGD( const IdealGeometryRecord& );

    std::unique_ptr<DetGeomDesc> produceRealGD( const VeryForwardRealGeometryRecord& );
    std::unique_ptr<CTPPSGeometry> produceRealTG( const VeryForwardRealGeometryRecord& );

    std::unique_ptr<DetGeomDesc> produceMisalignedGD( const VeryForwardMisalignedGeometryRecord& );
    std::unique_ptr<CTPPSGeometry> produceMisalignedTG( const VeryForwardMisalignedGeometryRecord& );

  protected:
    static void applyAlignments( const edm::ESHandle<DetGeomDesc>&, const edm::ESHandle<RPAlignmentCorrectionsData>&, DetGeomDesc*& );
    static void buildDetGeomDesc( DDFilteredView* fv, DetGeomDesc* gd );

    unsigned int verbosity_;
    std::string compactViewTag_;
};


//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSGeometryESModule::CTPPSGeometryESModule(const edm::ParameterSet& iConfig) :
  verbosity_( iConfig.getUntrackedParameter<unsigned int>( "verbosity", 1 ) ),
  compactViewTag_( iConfig.getParameter<std::string>( "compactViewTag" ) )
{
  setWhatProduced( this, &CTPPSGeometryESModule::produceIdealGD );

  setWhatProduced( this, &CTPPSGeometryESModule::produceRealGD );
  setWhatProduced( this, &CTPPSGeometryESModule::produceRealTG );

  setWhatProduced( this, &CTPPSGeometryESModule::produceMisalignedGD );
  setWhatProduced( this, &CTPPSGeometryESModule::produceMisalignedTG );
}

//----------------------------------------------------------------------------------------------------

void
CTPPSGeometryESModule::applyAlignments( const edm::ESHandle<DetGeomDesc>& idealGD,
                                        const edm::ESHandle<RPAlignmentCorrectionsData>& alignments,
                                        DetGeomDesc*& newGD )
{
  newGD = new DetGeomDesc( *(idealGD.product()) );
  std::deque<const DetGeomDesc*> buffer;
  std::deque<DetGeomDesc*> bufferNew;
  buffer.emplace_back( idealGD.product() );
  bufferNew.emplace_back( newGD );

  while ( !buffer.empty() ) {
    const DetGeomDesc* sD = buffer.front();
    DetGeomDesc* pD = bufferNew.front();
    buffer.pop_front();
    bufferNew.pop_front();

    const std::string name = pD->name().name();

    // Is it sensor? If yes, apply full sensor alignments
    if ( name == DDD_TOTEM_RP_SENSOR_NAME
      || name == DDD_CTPPS_DIAMONDS_SEGMENT_NAME
      || name == DDD_CTPPS_UFSD_SEGMENT_NAME
      || name == DDD_CTPPS_PIXELS_SENSOR_NAME
      || std::regex_match( name, std::regex( DDD_TOTEM_TIMING_SENSOR_TMPL ) )) {
      unsigned int plId = pD->geographicalID();

      if ( alignments.isValid() ) {
        const RPAlignmentCorrectionData& ac = alignments->getFullSensorCorrection( plId );
        pD->ApplyAlignment( ac );
      }
    }

    // Is it RP box? If yes, apply RP alignments
    if ( name == DDD_TOTEM_RP_RP_NAME
      || name == DDD_CTPPS_DIAMONDS_RP_NAME
      || name == DDD_CTPPS_PIXELS_RP_NAME
      || name == DDD_TOTEM_TIMING_RP_NAME ) {
      unsigned int rpId = pD->geographicalID();

      if ( alignments.isValid() ) {
        const RPAlignmentCorrectionData& ac = alignments->getRPCorrection( rpId );
        pD->ApplyAlignment( ac );
      }
    }

    // create and add children
    for ( unsigned int i = 0; i < sD->components().size(); i++ ) {
      const DetGeomDesc* sDC = sD->components()[i];
      buffer.emplace_back( sDC );

      // create new node with the same information as in sDC and add it as a child of pD
      DetGeomDesc* cD = new DetGeomDesc( *sDC );
      pD->addComponent( cD );

      bufferNew.emplace_back( cD );
    }
  }
}

//----------------------------------------------------------------------------------------------------

void
CTPPSGeometryESModule::buildDetGeomDesc( DDFilteredView* fv, DetGeomDesc* gd )
{
  // try to dive into next level
  if ( !fv->firstChild() ) return;

  // loop over siblings in the level
  do {
    // create new DetGeomDesc node and add it to the parent's (gd) list
    DetGeomDesc* newGD = new DetGeomDesc( fv );

    const std::string name = fv->logicalPart().name().name();

    // strip sensors
    if ( name == DDD_TOTEM_RP_SENSOR_NAME ) {
      const std::vector<int>& copy_num = fv->copyNumbers();
      // check size of copy numubers array
      if ( copy_num.size() < 3 )
        throw cms::Exception("DDDTotemRPContruction") << "size of copyNumbers for strip sensor is "
          << copy_num.size() << ". It must be >= 3.";

      // extract information
      const unsigned int decRPId = copy_num[copy_num.size() - 3];
      const unsigned int arm = decRPId / 100;
      const unsigned int station = ( decRPId % 100 ) / 10;
      const unsigned int rp = decRPId % 10;
      const unsigned int detector = copy_num[copy_num.size() - 1];
      newGD->setGeographicalID( TotemRPDetId( arm, station, rp, detector ) );
    }

    // strip and pixels RPs
    else if ( name == DDD_TOTEM_RP_RP_NAME || name == DDD_CTPPS_PIXELS_RP_NAME ) {
      unsigned int decRPId = fv->copyno();

      // check if it is a pixel RP
      if ( decRPId >= 10000 ){
        decRPId = decRPId % 10000;
        const unsigned int armIdx = ( decRPId / 100 ) % 10;
        const unsigned int stIdx = ( decRPId / 10 ) % 10;
        const unsigned int rpIdx = decRPId % 10;
        newGD->setGeographicalID( CTPPSPixelDetId( armIdx, stIdx, rpIdx ) );
      }
      else {
        const unsigned int armIdx = ( decRPId / 100 ) % 10;
        const unsigned int stIdx = ( decRPId / 10 ) % 10;
        const unsigned int rpIdx = decRPId % 10;
        newGD->setGeographicalID( TotemRPDetId( armIdx, stIdx, rpIdx ) );
      }
    }

    else if ( std::regex_match( name, std::regex( DDD_TOTEM_TIMING_SENSOR_TMPL ) ) ) {
      const std::vector<int>& copy_num = fv->copyNumbers();
      // check size of copy numbers array
      if ( copy_num.size() < 4 )
        throw cms::Exception("DDDTotemRPContruction") << "size of copyNumbers for TOTEM timing sensor is "
          << copy_num.size() << ". It must be >= 4.";

      const unsigned int decRPId = copy_num[copy_num.size()-4];
      const unsigned int arm = decRPId / 100, station = ( decRPId % 100 )/10, rp = decRPId % 10;
      const unsigned int plane = copy_num[copy_num.size()-2], channel = copy_num[copy_num.size()-1];
      newGD->setGeographicalID( TotemTimingDetId( arm, station, rp, plane, channel ) );
    }

    else if ( name == DDD_TOTEM_TIMING_RP_NAME ) {
      const unsigned int arm = fv->copyno() / 100, station = ( fv->copyno() % 100 )/10, rp = fv->copyno() % 10;
      newGD->setGeographicalID( TotemTimingDetId( arm, station, rp ) );
    }

    // pixel sensors
    else if ( name == DDD_CTPPS_PIXELS_SENSOR_NAME ) {
      const std::vector<int>& copy_num = fv->copyNumbers();
      // check size of copy numubers array
      if ( copy_num.size() < 4 )
        throw cms::Exception("DDDTotemRPContruction") << "size of copyNumbers for pixel sensor is "
          << copy_num.size() << ". It must be >= 4.";

      // extract information
      const unsigned int decRPId = copy_num[copy_num.size() - 4] % 10000;
      const unsigned int arm = decRPId / 100;
      const unsigned int station = ( decRPId % 100 ) / 10;
      const unsigned int rp = decRPId % 10;
      const unsigned int detector = copy_num[copy_num.size() - 2] - 1;
      newGD->setGeographicalID( CTPPSPixelDetId( arm, station, rp, detector ) );
    }

    // diamond/UFSD sensors
    else if ( name == DDD_CTPPS_DIAMONDS_SEGMENT_NAME || name == DDD_CTPPS_UFSD_SEGMENT_NAME ) {
      const std::vector<int>& copy_num = fv->copyNumbers();

      const unsigned int id = copy_num[copy_num.size()-1];
      const unsigned int arm = copy_num[1]-1;
      const unsigned int station = 1;
      const unsigned int rp = 6;
      const unsigned int plane = ( id / 100 );
      const unsigned int channel = id % 100;

      newGD->setGeographicalID( CTPPSDiamondDetId( arm, station, rp, plane, channel ) );
    }

    // diamond/UFSD RPs
    else if ( name == DDD_CTPPS_DIAMONDS_RP_NAME ) {
      const std::vector<int>& copy_num = fv->copyNumbers();

      // check size of copy numubers array
      if ( copy_num.size() < 2 )
        throw cms::Exception("DDDTotemRPContruction") << "size of copyNumbers for diamond RP is "
          << copy_num.size() << ". It must be >= 2.";

      const unsigned int arm = copy_num[1] - 1;
      const unsigned int station = 1;
      const unsigned int rp = 6;

      newGD->setGeographicalID( CTPPSDiamondDetId( arm, station, rp ) );
    }

    // add component
    gd->addComponent( newGD );

    // recursion
    buildDetGeomDesc( fv, newGD );
  } while ( fv->nextSibling() );

  // go a level up
  fv->parent();
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc>
CTPPSGeometryESModule::produceIdealGD( const IdealGeometryRecord& iRecord )
{
  // get the DDCompactView from EventSetup
  edm::ESHandle<DDCompactView> cpv;
  iRecord.get( compactViewTag_, cpv );

  // create DDFilteredView and apply the filter
  DDPassAllFilter filter;
  DDFilteredView fv( *( cpv.product() ), filter );

  // conversion to DetGeomDesc structure
  DetGeomDesc* root = new DetGeomDesc( &fv );
  buildDetGeomDesc( &fv, root );

  // construct the tree of DetGeomDesc
  return std::unique_ptr<DetGeomDesc>( const_cast<DetGeomDesc*>( root ) );
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc>
CTPPSGeometryESModule::produceRealGD( const VeryForwardRealGeometryRecord& iRecord )
{
  // get the input GeometricalDet
  edm::ESHandle<DetGeomDesc> idealGD;
  iRecord.getRecord<IdealGeometryRecord>().get( idealGD );

  // load alignments
  edm::ESHandle<RPAlignmentCorrectionsData> alignments;
  try { iRecord.getRecord<RPRealAlignmentRecord>().get( alignments ); }
  catch ( cms::Exception& ) {}

  if ( alignments.isValid() ) {
    if ( verbosity_ )
      edm::LogVerbatim("CTPPSGeometryESModule::produceRealGD")
        << ">> CTPPSGeometryESModule::produceRealGD > Real geometry: "
        << alignments->getRPMap().size() << " RP and "
        << alignments->getSensorMap().size() << " sensor alignments applied.";
  }
  else {
    if ( verbosity_ )
      edm::LogVerbatim("CTPPSGeometryESModule::produceRealGD")
        << ">> CTPPSGeometryESModule::produceRealGD > Real geometry: No alignments applied.";
  }

  DetGeomDesc* newGD = nullptr;
  applyAlignments( idealGD, alignments, newGD );
  return std::unique_ptr<DetGeomDesc>( newGD );
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc>
CTPPSGeometryESModule::produceMisalignedGD( const VeryForwardMisalignedGeometryRecord& iRecord )
{
  // get the input GeometricalDet
  edm::ESHandle<DetGeomDesc> idealGD;
  iRecord.getRecord<IdealGeometryRecord>().get( idealGD );

  // load alignments
  edm::ESHandle<RPAlignmentCorrectionsData> alignments;
  try { iRecord.getRecord<RPMisalignedAlignmentRecord>().get( alignments ); }
  catch ( cms::Exception& ) {}

  if ( alignments.isValid() ) {
    if ( verbosity_ )
      edm::LogVerbatim("CTPPSGeometryESModule::produceMisalignedGD")
        << ">> CTPPSGeometryESModule::produceMisalignedGD > Misaligned geometry: "
        << alignments->getRPMap().size() << " RP and "
        << alignments->getSensorMap().size() << " sensor alignments applied.";
  } else {
    if ( verbosity_ )
      edm::LogVerbatim("CTPPSGeometryESModule::produceMisalignedGD")
        << ">> CTPPSGeometryESModule::produceMisalignedGD > Misaligned geometry: No alignments applied.";
  }

  DetGeomDesc* newGD = nullptr;
  applyAlignments( idealGD, alignments, newGD );
  return std::unique_ptr<DetGeomDesc>( newGD );
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSGeometry>
CTPPSGeometryESModule::produceRealTG( const VeryForwardRealGeometryRecord& iRecord )
{
  edm::ESHandle<DetGeomDesc> gD;
  iRecord.get( gD );

  return std::make_unique<CTPPSGeometry>( gD.product() );
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSGeometry>
CTPPSGeometryESModule::produceMisalignedTG( const VeryForwardMisalignedGeometryRecord& iRecord )
{
  edm::ESHandle<DetGeomDesc> gD;
  iRecord.get( gD );

  return std::make_unique<CTPPSGeometry>( gD.product() );
}

DEFINE_FWK_EVENTSETUP_MODULE( CTPPSGeometryESModule );
