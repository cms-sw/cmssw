/****************************************************************************
*
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Class to print out information on current geometry.
 **/
class CTPPSGeometryInfo : public edm::one::EDAnalyzer<>
{
  public:
    explicit CTPPSGeometryInfo( const edm::ParameterSet& );

  private: 
    std::string geometryType_;

    bool printRPInfo_, printSensorInfo_;

    edm::ESWatcher<IdealGeometryRecord> watcherIdealGeometry_;
    edm::ESWatcher<VeryForwardRealGeometryRecord> watcherRealGeometry_;
    edm::ESWatcher<VeryForwardMisalignedGeometryRecord> watcherMisalignedGeometry_;

    void analyze( const edm::Event&, const edm::EventSetup& ) override;

    static void PrintDetId( const CTPPSDetId &id, bool printDetails = true );

    void PrintGeometry( const CTPPSGeometry &, const edm::Event& );
};

//----------------------------------------------------------------------------------------------------

CTPPSGeometryInfo::CTPPSGeometryInfo( const edm::ParameterSet& iConfig ) :
  geometryType_   ( iConfig.getUntrackedParameter<std::string>( "geometryType", "real" ) ),
  printRPInfo_    ( iConfig.getUntrackedParameter<bool>( "printRPInfo", true ) ),
  printSensorInfo_( iConfig.getUntrackedParameter<bool>( "printSensorInfo", true ) )
{
}

//----------------------------------------------------------------------------------------------------

void
CTPPSGeometryInfo::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  edm::ESHandle<CTPPSGeometry> geometry;

  if ( geometryType_ == "ideal" ) {
    if ( watcherIdealGeometry_.check( iSetup ) ) {
      iSetup.get<IdealGeometryRecord>().get( geometry );
      PrintGeometry( *geometry, iEvent );
    }
    return;
  }

  else if ( geometryType_ == "real" ) {
    if ( watcherRealGeometry_.check( iSetup ) ) {
      iSetup.get<VeryForwardRealGeometryRecord>().get( geometry );
      PrintGeometry( *geometry, iEvent );
    }
    return;
  }

  else if ( geometryType_ == "misaligned" ) {
    if ( watcherMisalignedGeometry_.check( iSetup ) ) {
      iSetup.get<VeryForwardMisalignedGeometryRecord>().get( geometry );
      PrintGeometry( *geometry, iEvent );
    }
    return;
  }

  throw cms::Exception("CTPPSGeometryInfo") << "Unknown geometry type: `" << geometryType_ << "'.";
}

//----------------------------------------------------------------------------------------------------

void
CTPPSGeometryInfo::PrintDetId( const CTPPSDetId &id, bool printDetails )
{
  std::ostringstream oss;
  oss << id.rawId();

  const unsigned int rpDecId = id.arm()*100 + id.station()*10 + id.rp();

  if ( id.subdetId() == CTPPSDetId::sdTrackingStrip ) {
    TotemRPDetId fid( id );
    oss << " (strip RP " << rpDecId;
    if ( printDetails )
      oss <<  ", plane " << fid.plane();
    oss << ")";
  }

  if ( id.subdetId() == CTPPSDetId::sdTrackingPixel ) {
    CTPPSPixelDetId fid( id );
    oss << " (pixel RP " << rpDecId;
    if ( printDetails )
      oss <<  ", plane " << fid.plane();
    oss << ")";
  }

  if (id.subdetId() == CTPPSDetId::sdTimingDiamond) {
    CTPPSDiamondDetId fid( id );
    oss << " (diamd RP " << rpDecId;
    if ( printDetails )
      oss <<  ", plane " << fid.plane() << ", channel " << fid.channel();
    oss << ")";
  }
  edm::LogVerbatim("CTPPSGeometryInfo") << oss.str();
}

//----------------------------------------------------------------------------------------------------

void
CTPPSGeometryInfo::PrintGeometry( const CTPPSGeometry& geometry, const edm::Event& event )
{
  time_t unixTime = event.time().unixTime();
  char timeStr[50];
  strftime( timeStr, 50, "%F %T", localtime( &unixTime ) );

  edm::LogVerbatim("CTPPSGeometryInfo")
    << "new " << geometryType_ << " geometry found in run="
    << event.id().run() << ", event=" << event.id().event() << ", UNIX timestamp=" << unixTime
    << " (" << timeStr << ")";

  // RP geometry
  if ( printRPInfo_ ) {
    std::ostringstream oss;
    oss << "* RPs:\n"
        << "    ce: RP center in global coordinates, in mm\n";
    for ( auto it = geometry.beginRP(); it != geometry.endRP(); ++it ) {
      const DDTranslation &t = it->second->translation();

      PrintDetId( CTPPSDetId( it->first ), false );
      oss << std::fixed << std::setprecision( 3 ) << " | ce=(" << t.x() << ", " << t.y() << ", " << t.z() << ")";
    }
    edm::LogVerbatim("CTPPSGeometryInfo") << oss.str();
  }

  // sensor geometry
  if ( printSensorInfo_ ) {
    edm::LogVerbatim("CTPPSGeometryInfo")
      << "* sensors:\n"
      << "    ce: sensor center in global coordinates, in mm\n"
      << "    a1: local axis (1, 0, 0) in global coordinates\n"
      << "    a2: local axis (0, 1, 0) in global coordinates\n"
      << "    a3: local axis (0, 0, 1) in global coordinates";

    for ( auto it = geometry.beginSensor(); it != geometry.endSensor(); ++it ) {
      CTPPSDetId detId( it->first );

      const CLHEP::Hep3Vector gl_o  = geometry.localToGlobal( detId, CLHEP::Hep3Vector( 0, 0, 0 ) );
      const CLHEP::Hep3Vector gl_a1 = geometry.localToGlobal( detId, CLHEP::Hep3Vector( 1, 0, 0 ) ) - gl_o;
      const CLHEP::Hep3Vector gl_a2 = geometry.localToGlobal( detId, CLHEP::Hep3Vector( 0, 1, 0 ) ) - gl_o;
      const CLHEP::Hep3Vector gl_a3 = geometry.localToGlobal( detId, CLHEP::Hep3Vector( 0, 0, 1 ) ) - gl_o;

      PrintDetId( detId );

      edm::LogVerbatim("CTPPSGeometryInfo")
        << " | ce=(" << gl_o.x() << ", " << gl_o.y() << ", " << gl_o.z() << ")"
        << " | a1=(" << gl_a1.x() << ", " << gl_a1.y() << ", " << gl_a1.z() << ")"
        << " | a2=(" << gl_a2.x() << ", " << gl_a2.y() << ", " << gl_a2.z() << ")"
        << " | a3=(" << gl_a3.x() << ", " << gl_a3.y() << ", " << gl_a3.z() << ")";
    }
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE( CTPPSGeometryInfo );
