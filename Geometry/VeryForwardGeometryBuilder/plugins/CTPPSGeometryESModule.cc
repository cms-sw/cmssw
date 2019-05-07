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

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
//#include "CondFormats/AlignmentRecord/interface/CTPPSRPAlignmentCorrectionsDataRcd.h"

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

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::unique_ptr<DetGeomDesc> produceIdealGD( const IdealGeometryRecord& );

    template<typename ALIGNMENT_REC> struct GDTokens {
      explicit GDTokens( edm::ESConsumesCollector&& iCC):
        idealGDToken_{iCC.consumesFrom<DetGeomDesc, IdealGeometryRecord>(edm::ESInputTag())},
        alignmentToken_{iCC.consumesFrom<CTPPSRPAlignmentCorrectionsData, ALIGNMENT_REC>(edm::ESInputTag())}
      {}
      const edm::ESGetToken<DetGeomDesc, IdealGeometryRecord> idealGDToken_;
      const edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, ALIGNMENT_REC> alignmentToken_;
    };

    std::unique_ptr<DetGeomDesc> produceRealGD( const VeryForwardRealGeometryRecord& );
    std::unique_ptr<CTPPSGeometry> produceRealTG( const VeryForwardRealGeometryRecord& );

    std::unique_ptr<DetGeomDesc> produceMisalignedGD( const VeryForwardMisalignedGeometryRecord& );
    std::unique_ptr<CTPPSGeometry> produceMisalignedTG( const VeryForwardMisalignedGeometryRecord& );

    template<typename REC>
    std::unique_ptr<DetGeomDesc> produceGD( IdealGeometryRecord const&, const std::optional<REC>&, GDTokens<REC> const&, const char* name);
    

    static void applyAlignments( const DetGeomDesc&, const CTPPSRPAlignmentCorrectionsData*, DetGeomDesc*& );
    static void buildDetGeomDesc( DDFilteredView* fv, DetGeomDesc* gd );

    const unsigned int verbosity_;
    const edm::ESGetToken<DDCompactView, IdealGeometryRecord> compactViewToken_;

    const GDTokens<RPRealAlignmentRecord> gdRealTokens_;
    const GDTokens<RPMisalignedAlignmentRecord> gdMisTokens_;

    const edm::ESGetToken<DetGeomDesc, VeryForwardRealGeometryRecord> dgdRealToken_;
    const edm::ESGetToken<DetGeomDesc, VeryForwardMisalignedGeometryRecord> dgdMisToken_;

};


//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSGeometryESModule::CTPPSGeometryESModule(const edm::ParameterSet& iConfig) :
  verbosity_( iConfig.getUntrackedParameter<unsigned int>( "verbosity") ),
  compactViewToken_{ setWhatProduced( this, &CTPPSGeometryESModule::produceIdealGD )
    .consumes<DDCompactView>(edm::ESInputTag(""/*optional module label */,
                                             iConfig.getParameter<std::string>( "compactViewTag" ) )) },
  gdRealTokens_{ setWhatProduced( this, &CTPPSGeometryESModule::produceRealGD ) },
  gdMisTokens_{ setWhatProduced( this, &CTPPSGeometryESModule::produceMisalignedGD ) },
  dgdRealToken_{ setWhatProduced( this, &CTPPSGeometryESModule::produceRealTG ).consumes<DetGeomDesc>(edm::ESInputTag()) },
  dgdMisToken_{ setWhatProduced( this, &CTPPSGeometryESModule::produceMisalignedTG ).consumes<DetGeomDesc>(edm::ESInputTag()) }
{
}

void 
CTPPSGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<unsigned int>("verbosity", 1);
  desc.add<std::string>("compactViewTag", std::string());
  descriptions.add("DoodadESSource", desc);
}


//----------------------------------------------------------------------------------------------------

void
CTPPSGeometryESModule::applyAlignments( const DetGeomDesc& idealGD,
                                        const CTPPSRPAlignmentCorrectionsData* alignments,
                                        DetGeomDesc*& newGD )
{
  newGD = new DetGeomDesc( idealGD );
  std::deque<const DetGeomDesc*> buffer;
  std::deque<DetGeomDesc*> bufferNew;
  buffer.emplace_back( &idealGD );
  bufferNew.emplace_back( newGD );

  while ( !buffer.empty() ) {
    const DetGeomDesc* sD = buffer.front();
    DetGeomDesc* pD = bufferNew.front();
    buffer.pop_front();
    bufferNew.pop_front();

    const std::string name = pD->name();

    // Is it sensor? If yes, apply full sensor alignments
    if ( name == DDD_TOTEM_RP_SENSOR_NAME
      || name == DDD_CTPPS_DIAMONDS_SEGMENT_NAME
      || name == DDD_CTPPS_UFSD_SEGMENT_NAME
      || name == DDD_CTPPS_PIXELS_SENSOR_NAME
      || std::regex_match( name, std::regex( DDD_TOTEM_TIMING_SENSOR_TMPL ) )) {
      unsigned int plId = pD->geographicalID();

      if ( alignments ) {
        const auto& ac = alignments->getFullSensorCorrection( plId );
        pD->applyAlignment( ac );
      }
    }

    // Is it RP box? If yes, apply RP alignments
    if ( name == DDD_TOTEM_RP_RP_NAME
      || name == DDD_CTPPS_DIAMONDS_RP_NAME
      || name == DDD_CTPPS_PIXELS_RP_NAME
      || name == DDD_TOTEM_TIMING_RP_NAME ) {
      unsigned int rpId = pD->geographicalID();

      if ( alignments ) {
        const auto& ac = alignments->getRPCorrection( rpId );
        pD->applyAlignment( ac );
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
  auto const& cpv = iRecord.get( compactViewToken_);

  // create DDFilteredView and apply the filter
  DDPassAllFilter filter;
  DDFilteredView fv( cpv, filter );

  // conversion to DetGeomDesc structure
  auto root = std::make_unique<DetGeomDesc>( &fv );
  buildDetGeomDesc( &fv, root.get() );

  // construct the tree of DetGeomDesc
  return root;
}

//----------------------------------------------------------------------------------------------------

template<typename REC>
std::unique_ptr<DetGeomDesc>
CTPPSGeometryESModule::produceGD(IdealGeometryRecord const& iIdealRec, 
                                 std::optional<REC> const& iAlignRec,
                                 GDTokens<REC> const& iTokens,
                                 const char* name) {
  // get the input GeometricalDet
  auto const& idealGD = iIdealRec.get( iTokens.idealGDToken_ );

  // load alignments
  edm::ESHandle<CTPPSRPAlignmentCorrectionsData> alignments;
  if(iAlignRec) {
    alignments = iAlignRec->getHandle(iTokens.alignmentToken_);
  }

  if ( alignments.isValid() ) {
    if ( verbosity_ )
      edm::LogVerbatim(name)
        << ">> "<<name<<" > Real geometry: "
        << alignments->getRPMap().size() << " RP and "
        << alignments->getSensorMap().size() << " sensor alignments applied.";
  }
  else {
    if ( verbosity_ )
      edm::LogVerbatim(name)
        << ">> "<<name<< " > Real geometry: No alignments applied.";
  }

  DetGeomDesc* newGD = nullptr;
  applyAlignments( idealGD, alignments.product(), newGD );
  return std::unique_ptr<DetGeomDesc>( newGD );
}

std::unique_ptr<DetGeomDesc>
CTPPSGeometryESModule::produceRealGD( const VeryForwardRealGeometryRecord& iRecord )
{
  return produceGD(iRecord.getRecord<IdealGeometryRecord>(), iRecord.tryToGetRecord<RPRealAlignmentRecord>(), gdRealTokens_,
                   "CTPPSGeometryESModule::produceRealGD");
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<DetGeomDesc>
CTPPSGeometryESModule::produceMisalignedGD( const VeryForwardMisalignedGeometryRecord& iRecord )
{
  return produceGD(iRecord.getRecord<IdealGeometryRecord>(), iRecord.tryToGetRecord<RPMisalignedAlignmentRecord>(), gdMisTokens_,
                   "CTPPSGeometryESModule::produceMisalignedGD");
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSGeometry>
CTPPSGeometryESModule::produceRealTG( const VeryForwardRealGeometryRecord& iRecord )
{
  auto const& gD = iRecord.get( dgdRealToken_ );

  return std::make_unique<CTPPSGeometry>( &gD );
}

//----------------------------------------------------------------------------------------------------

std::unique_ptr<CTPPSGeometry>
CTPPSGeometryESModule::produceMisalignedTG( const VeryForwardMisalignedGeometryRecord& iRecord )
{
  auto const& gD = iRecord.get( dgdMisToken_ );

  return std::make_unique<CTPPSGeometry>( &gD );
}

DEFINE_FWK_EVENTSETUP_MODULE( CTPPSGeometryESModule );
