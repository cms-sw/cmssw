////////////////////////////
// Geometry Checklist     //
// Maps with PSimHits     //
//                        //
// Nicola Pozzobon - 2012 //
////////////////////////////

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH2D.h>

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class AnalyzerSimHitMaps : public edm::EDAnalyzer
{
  /// Public methods
  public:
    /// Constructor/destructor
    explicit AnalyzerSimHitMaps(const edm::ParameterSet& iConfig);
    virtual ~AnalyzerSimHitMaps();
    // Typical methods used on Loops over events
    virtual void beginJob();
    virtual void endJob();
    virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  /// Private methods and variables
  private:
    /// Global Position of SimHits
    TH2D* hSimHit_Barrel_XY;
    TH2D* hSimHit_Barrel_XY_Zoom;
    TH2D* hSimHit_Endcap_Fw_XY;
    TH2D* hSimHit_Endcap_Bw_XY;
    TH2D* hSimHit_RZ;
    TH2D* hSimHit_Endcap_Fw_RZ_Zoom;
    TH2D* hSimHit_Endcap_Bw_RZ_Zoom;

    std::map< std::string, TH2D* > m_hSimHit_Barrel_XY_Survey;
    std::map< std::string, TH2D* > m_hSimHit_RZ_Survey;
};

//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
AnalyzerSimHitMaps::AnalyzerSimHitMaps(edm::ParameterSet const& iConfig) 
{
  /// Insert here what you need to initialize
}

/////////////
// DESTRUCTOR
AnalyzerSimHitMaps::~AnalyzerSimHitMaps()
{
  /// Insert here what you need to delete
  /// when you close the class instance
}  

//////////
// END JOB
void AnalyzerSimHitMaps::endJob()
{
  /// Things to be done at the exit of the event Loop
  std::cerr << " AnalyzerSimHitMaps::endJob" << std::endl;
  /// End of things to be done at the exit from the event Loop
}

////////////
// BEGIN JOB
void AnalyzerSimHitMaps::beginJob()
{
  /// Initialize all slave variables
  /// mainly histogram ranges and resolution
  std::ostringstream histoName;
  std::ostringstream histoTitle;

  /// Things to be done before entering the event Loop
  std::cerr << " AnalyzerSimHitMaps::beginJob" << std::endl;

  /// Book histograms etc
  edm::Service<TFileService> fs;

  hSimHit_Barrel_XY          = fs->make<TH2D>( "hSimHit_Barrel_XY",         "PSimHit Barrel y vs. x",              960, -120, 120, 960, -120, 120 );
  hSimHit_Barrel_XY_Zoom     = fs->make<TH2D>( "hSimHit_Barrel_XY_Zoom",    "PSimHit Barrel y vs. x",              960, 30, 60, 960, -15, 15 );
  hSimHit_Endcap_Fw_XY       = fs->make<TH2D>( "hSimHit_Endcap_Fw_XY",      "PSimHit Forward Endcap y vs. x",      960, -120, 120, 960, -120, 120 );
  hSimHit_Endcap_Bw_XY       = fs->make<TH2D>( "hSimHit_Endcap_Bw_XY",      "PSimHit Backward Endcap y vs. x",     960, -120, 120, 960, -120, 120 );
  hSimHit_RZ                 = fs->make<TH2D>( "hSimHit_RZ",                "PSimHit #rho vs. z",                  900, -300, 300, 480, 0, 120 );
  hSimHit_Endcap_Fw_RZ_Zoom  = fs->make<TH2D>( "hSimHit_Endcap_Fw_RZ_Zoom", "PSimHit Forward Endcap #rho vs. z",   960, 140, 170, 960, 30, 60 );
  hSimHit_Endcap_Bw_RZ_Zoom  = fs->make<TH2D>( "hSimHit_Endcap_Bw_RZ_Zoom", "PSimHit Backward Endcap #rho vs. z",  960, -170, -140, 960, 70, 100 );

  hSimHit_Barrel_XY->Sumw2();
  hSimHit_Barrel_XY_Zoom->Sumw2();
  hSimHit_Endcap_Fw_XY->Sumw2();
  hSimHit_Endcap_Bw_XY->Sumw2();
  hSimHit_RZ->Sumw2();
  hSimHit_Endcap_Fw_RZ_Zoom->Sumw2();
  hSimHit_Endcap_Bw_RZ_Zoom->Sumw2();

  for ( int ix = 0; ix < 11; ix++ )
  {
    for ( int iy = 10; iy > -1; iy-- )
    {
      histoName.str("");
      histoTitle.str("");
      histoName << "hSimHit_Barrel_XY_Survey_" << -110+ix*20 << "x" << -110+(1+ix)*20 << "_" << -110+iy*20 << "y" << -110+(1+iy)*20;
      histoTitle << "PSimHit Barrel y (" << -110+iy*20 << ", " << -110+(1+iy)*20 << ") cm vs. x (" << -110+ix*20 << ", " << -110+(1+ix)*20 <<") cm";
      TH2D* h = fs->make<TH2D>( histoName.str().c_str(), histoTitle.str().c_str(), 800, -110+ix*20, -110+(1+ix)*20, 800, -110+iy*20, -110+(1+iy)*20 );
      m_hSimHit_Barrel_XY_Survey.insert( std::pair< std::string, TH2D* >( histoName.str(), h ) );
    }
  }

  for ( int iz = 0; iz < 27; iz ++ )
  {
    for ( int ir = 5; ir >= 0; ir-- )
    {
      histoName.str("");
      histoTitle.str("");
      histoName << "hSimHit_RZ_Survey_" << -10+ir*20 << "r" << -10+(1+ir)*20 << "_" << -270+iz*20 << "z" << -270+(1+iz)*20;
      histoTitle << "PSimHit #rho (" << -10+ir*20 << ", " << -10+(1+ir)*20 << ") cm vs. z (" << -270+iz*20 << ", " << -270+(1+iz)*20 <<") cm";
      TH2D* h = fs->make<TH2D>( histoName.str().c_str(), histoTitle.str().c_str(), 800, -270+iz*20, -270+(1+iz)*20, 800, -10+ir*20, -10+(1+ir)*20 );
      m_hSimHit_RZ_Survey.insert( std::pair< std::string, TH2D* >( histoName.str(), h ) );
    }
  }

  /// End of things to be done before entering the event Loop
}

//////////
// ANALYZE
void AnalyzerSimHitMaps::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  /// Geometry handles etc
  edm::ESHandle< TrackerGeometry >         geometryHandle;
  const TrackerGeometry*                   theGeometry;

  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
  theGeometry = &(*geometryHandle);

  //////////////////
  // GET SIM HITS //
  //////////////////
  edm::Handle<edm::PSimHitContainer> simHitHandlePXBH;
  edm::Handle<edm::PSimHitContainer> simHitHandlePXBL;
  edm::Handle<edm::PSimHitContainer> simHitHandlePXFH;
  edm::Handle<edm::PSimHitContainer> simHitHandlePXFL;
  iEvent.getByLabel("g4SimHits", "TrackerHitsPixelBarrelHighTof", simHitHandlePXBH);
  iEvent.getByLabel("g4SimHits", "TrackerHitsPixelBarrelLowTof",  simHitHandlePXBL);
  iEvent.getByLabel("g4SimHits", "TrackerHitsPixelEndcapHighTof", simHitHandlePXFH);
  iEvent.getByLabel("g4SimHits", "TrackerHitsPixelEndcapLowTof",  simHitHandlePXFL);

  /// Loop over Sim Hits collections
  edm::PSimHitContainer::const_iterator iterSimHit;
  for ( iterSimHit = simHitHandlePXBH->begin();
        iterSimHit != simHitHandlePXBH->end();
        iterSimHit++ )
  {
    const GeomDetUnit* gDetUnit = theGeometry->idToDetUnit( iterSimHit->detUnitId() );
    GlobalPoint shPos = gDetUnit->surface().toGlobal( iterSimHit->localPosition() ) ;
    std::ostringstream histoNameXY;
    std::ostringstream histoNameRZ;
    histoNameXY << "hSimHit_Barrel_XY_Survey_" << (20*(floor((shPos.x()+10)/20))-10) << "x" << (20*(floor((shPos.x()+10)/20))+10) <<
                                           "_" << (20*(floor((shPos.y()+10)/20))-10) << "y" << (20*(floor((shPos.y()+10)/20))+10);
    histoNameRZ << "hSimHit_RZ_Survey_" << (20*(floor((shPos.perp()+10)/20))-10) << "r" << (20*(floor((shPos.perp()+10)/20))+10) <<
                                    "_" << (20*(floor((shPos.z()+10)/20))-10) << "z" << (20*(floor((shPos.z()+10)/20))+10);
    hSimHit_RZ->Fill( shPos.z(), shPos.perp() );
    if( m_hSimHit_RZ_Survey.find( histoNameRZ.str() ) != m_hSimHit_RZ_Survey.end() )
      m_hSimHit_RZ_Survey.find( histoNameRZ.str() )->second->Fill( shPos.z(), shPos.perp() );
    if (gDetUnit->type().isBarrel())
    {
      hSimHit_Barrel_XY->Fill( shPos.x(), shPos.y() );
      hSimHit_Barrel_XY_Zoom->Fill( shPos.x(), shPos.y() );
      if ( m_hSimHit_Barrel_XY_Survey.find( histoNameXY.str() ) != m_hSimHit_Barrel_XY_Survey.end() )
        m_hSimHit_Barrel_XY_Survey.find( histoNameXY.str() )->second->Fill( shPos.x(), shPos.y() );
    }
    else if (gDetUnit->type().isEndcap())
    {
      if (shPos.z() > 0)
      {
        hSimHit_Endcap_Fw_XY->Fill( shPos.x(), shPos.y() );
        hSimHit_Endcap_Fw_RZ_Zoom->Fill( shPos.z(), shPos.perp() );
      }
      else
      {
        hSimHit_Endcap_Bw_XY->Fill( shPos.x(), shPos.y() );
        hSimHit_Endcap_Bw_RZ_Zoom->Fill( shPos.z(), shPos.perp() );
      }
    }
  }

  for ( iterSimHit = simHitHandlePXBL->begin();
        iterSimHit != simHitHandlePXBL->end();
        iterSimHit++ )
  {
    const GeomDetUnit* gDetUnit = theGeometry->idToDetUnit( iterSimHit->detUnitId() );
    GlobalPoint shPos = gDetUnit->surface().toGlobal( iterSimHit->localPosition() ) ;
    std::ostringstream histoNameXY;
    std::ostringstream histoNameRZ;
    histoNameXY << "hSimHit_Barrel_XY_Survey_" << (20*(floor((shPos.x()+10)/20))-10) << "x" << (20*(floor((shPos.x()+10)/20))+10) <<
                                           "_" << (20*(floor((shPos.y()+10)/20))-10) << "y" << (20*(floor((shPos.y()+10)/20))+10);
    histoNameRZ << "hSimHit_RZ_Survey_" << (20*(floor((shPos.perp()+10)/20))-10) << "r" << (20*(floor((shPos.perp()+10)/20))+10) <<
                                    "_" << (20*(floor((shPos.z()+10)/20))-10) << "z" << (20*(floor((shPos.z()+10)/20))+10);
    hSimHit_RZ->Fill( shPos.z(), shPos.perp() );
    if( m_hSimHit_RZ_Survey.find( histoNameRZ.str() ) != m_hSimHit_RZ_Survey.end() )
      m_hSimHit_RZ_Survey.find( histoNameRZ.str() )->second->Fill( shPos.z(), shPos.perp() );
    if (gDetUnit->type().isBarrel())
    {
      hSimHit_Barrel_XY->Fill( shPos.x(), shPos.y() );
      hSimHit_Barrel_XY_Zoom->Fill( shPos.x(), shPos.y() );
      if ( m_hSimHit_Barrel_XY_Survey.find( histoNameXY.str() ) != m_hSimHit_Barrel_XY_Survey.end() )
        m_hSimHit_Barrel_XY_Survey.find( histoNameXY.str() )->second->Fill( shPos.x(), shPos.y() );
    }
    else if (gDetUnit->type().isEndcap())
    {
      if (shPos.z() > 0)
      {
        hSimHit_Endcap_Fw_XY->Fill( shPos.x(), shPos.y() );
        hSimHit_Endcap_Fw_RZ_Zoom->Fill( shPos.z(), shPos.perp() );
      }
      else 
      {
        hSimHit_Endcap_Bw_XY->Fill( shPos.x(), shPos.y() );
        hSimHit_Endcap_Bw_RZ_Zoom->Fill( shPos.z(), shPos.perp() );
      }
    }
  }

  for ( iterSimHit = simHitHandlePXFH->begin();
        iterSimHit != simHitHandlePXFH->end();
        iterSimHit++ )
  {
    const GeomDetUnit* gDetUnit = theGeometry->idToDetUnit( iterSimHit->detUnitId() );
    GlobalPoint shPos = gDetUnit->surface().toGlobal( iterSimHit->localPosition() ) ;
    std::ostringstream histoNameXY;
    std::ostringstream histoNameRZ;
    histoNameXY << "hSimHit_Barrel_XY_Survey_" << (20*(floor((shPos.x()+10)/20))-10) << "x" << (20*(floor((shPos.x()+10)/20))+10) <<
                                           "_" << (20*(floor((shPos.y()+10)/20))-10) << "y" << (20*(floor((shPos.y()+10)/20))+10);
    histoNameRZ << "hSimHit_RZ_Survey_" << (20*(floor((shPos.perp()+10)/20))-10) << "r" << (20*(floor((shPos.perp()+10)/20))+10) <<
                                    "_" << (20*(floor((shPos.z()+10)/20))-10) << "z" << (20*(floor((shPos.z()+10)/20))+10);
    hSimHit_RZ->Fill( shPos.z(), shPos.perp() );
    if( m_hSimHit_RZ_Survey.find( histoNameRZ.str() ) != m_hSimHit_RZ_Survey.end() )
      m_hSimHit_RZ_Survey.find( histoNameRZ.str() )->second->Fill( shPos.z(), shPos.perp() );
    if (gDetUnit->type().isBarrel())
    {
      hSimHit_Barrel_XY->Fill( shPos.x(), shPos.y() );
      hSimHit_Barrel_XY_Zoom->Fill( shPos.x(), shPos.y() );
      if ( m_hSimHit_Barrel_XY_Survey.find( histoNameXY.str() ) != m_hSimHit_Barrel_XY_Survey.end() )
        m_hSimHit_Barrel_XY_Survey.find( histoNameXY.str() )->second->Fill( shPos.x(), shPos.y() );
    }
    else if (gDetUnit->type().isEndcap())
    {
      if (shPos.z() > 0)
      {
        hSimHit_Endcap_Fw_XY->Fill( shPos.x(), shPos.y() );
        hSimHit_Endcap_Fw_RZ_Zoom->Fill( shPos.z(), shPos.perp() );
      }
      else 
      {
        hSimHit_Endcap_Bw_XY->Fill( shPos.x(), shPos.y() );
        hSimHit_Endcap_Bw_RZ_Zoom->Fill( shPos.z(), shPos.perp() );
      }
    }
  }

  for ( iterSimHit = simHitHandlePXFL->begin();
        iterSimHit != simHitHandlePXFL->end();
        iterSimHit++ )
  {
    const GeomDetUnit* gDetUnit = theGeometry->idToDetUnit( iterSimHit->detUnitId() );
    GlobalPoint shPos = gDetUnit->surface().toGlobal( iterSimHit->localPosition() ) ;
    std::ostringstream histoNameXY;
    std::ostringstream histoNameRZ;
    histoNameXY << "hSimHit_Barrel_XY_Survey_" << (20*(floor((shPos.x()+10)/20))-10) << "x" << (20*(floor((shPos.x()+10)/20))+10) <<
                                           "_" << (20*(floor((shPos.y()+10)/20))-10) << "y" << (20*(floor((shPos.y()+10)/20))+10);
    histoNameRZ << "hSimHit_RZ_Survey_" << (20*(floor((shPos.perp()+10)/20))-10) << "r" << (20*(floor((shPos.perp()+10)/20))+10) <<
                                    "_" << (20*(floor((shPos.z()+10)/20))-10) << "z" << (20*(floor((shPos.z()+10)/20))+10);
    hSimHit_RZ->Fill( shPos.z(), shPos.perp() );
    if( m_hSimHit_RZ_Survey.find( histoNameRZ.str() ) != m_hSimHit_RZ_Survey.end() )
      m_hSimHit_RZ_Survey.find( histoNameRZ.str() )->second->Fill( shPos.z(), shPos.perp() );
    if (gDetUnit->type().isBarrel())
    {
      hSimHit_Barrel_XY->Fill( shPos.x(), shPos.y() );
      hSimHit_Barrel_XY_Zoom->Fill( shPos.x(), shPos.y() );
      if ( m_hSimHit_Barrel_XY_Survey.find( histoNameXY.str() ) != m_hSimHit_Barrel_XY_Survey.end() )
        m_hSimHit_Barrel_XY_Survey.find( histoNameXY.str() )->second->Fill( shPos.x(), shPos.y() );
    }
    else if (gDetUnit->type().isEndcap())
    {
      if (shPos.z() > 0)
      {
        hSimHit_Endcap_Fw_XY->Fill( shPos.x(), shPos.y() );
        hSimHit_Endcap_Fw_RZ_Zoom->Fill( shPos.z(), shPos.perp() );
      }
      else 
      {
        hSimHit_Endcap_Bw_XY->Fill( shPos.x(), shPos.y() );
        hSimHit_Endcap_Bw_RZ_Zoom->Fill( shPos.z(), shPos.perp() );
      }
    }
  }

} /// End of analyze()

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(AnalyzerSimHitMaps);

