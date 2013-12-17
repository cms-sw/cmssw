//////////////////////////////
// Geometry Checklist       //
// Properties of Pt-Modules //
//                          //
// Nicola Pozzobon - 2011   //
//////////////////////////////

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH2D.h>
#include <TH1D.h>
#include <fstream>

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class AnalyzerPrintGeomInfo : public edm::EDAnalyzer
{
  /// Public methods
  public:
    /// Constructor/destructor
    explicit AnalyzerPrintGeomInfo(const edm::ParameterSet& iConfig);
    virtual ~AnalyzerPrintGeomInfo();
    // Typical methods used on Loops over events
    virtual void beginJob();
    virtual void endJob();
    virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  /// Private methods and variables
  private:

    std::string TextOutput;
    bool DebugMode;
 
    bool testedGeometry;
    bool debugPrintouts;

    /// Output file
    std::ofstream outputFile;

    /// Containers of parameters passed by python
    /// configuration file
    edm::ParameterSet config;    

    /// Histograms
    TH2D* hPXB_Lay_R;
    std::map< unsigned int, TH2D* > mapPXB2Lay_hPXB_Lad_Mod;
    std::map< unsigned int, TH2D* > mapPXB2Lay_hPXB_Lad_Phi;
    std::map< unsigned int, TH2D* > mapPXB2Lay_hPXB_Mod_Z;
    std::map< unsigned int, TH2D* > mapPXB2Lay_hPXB_Lad_R;
    std::map< unsigned int, TH2D* > mapPXB2Lay_hPXB_Mod_R;
    std::map< unsigned int, TH1D* > mapPXB2Lay_hPXB_R;

    TH2D* hPXF_Disk00_Disk;
    TH2D* hPXF_Disk00_Side;
    TH2D* hPXF_Disk00_Z;
    TH2D* hPXF_Disk00_R;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Pan_Mod;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Bla_Mod;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Bla_Pan;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Pan_R;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Mod_R;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Bla_R;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Disk_R;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Pan_Phi;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Mod_Phi;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Bla_Phi;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Pan_Z;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Mod_Z;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Bla_Z;
    std::map< unsigned int, TH2D* > mapPXF2Disk00_hPXF_Disk_Z;

};

//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
AnalyzerPrintGeomInfo::AnalyzerPrintGeomInfo(edm::ParameterSet const& iConfig) : 
  config(iConfig)
{
  /// Insert here what you need to initialize
  TextOutput = iConfig.getParameter< std::string >("TextOutput");
  DebugMode = iConfig.getParameter< bool >("DebugMode");

  /// Open the output file
  outputFile.open(TextOutput, std::ios::out);
}

/////////////
// DESTRUCTOR
AnalyzerPrintGeomInfo::~AnalyzerPrintGeomInfo()
{
  /// Insert here what you need to delete
  /// when you close the class instance
}  

//////////
// END JOB
void AnalyzerPrintGeomInfo::endJob()//edm::Run& run, const edm::EventSetup& iSetup
{
  /// Things to be done at the exit of the event Loop
  outputFile.close();

  std::cerr << " AnalyzerPrintGeomInfo::endJob" << std::endl;
  /// End of things to be done at the exit from the event Loop
}

////////////
// BEGIN JOB
void AnalyzerPrintGeomInfo::beginJob()
{
  std::ostringstream histoName;
  std::ostringstream histoTitle;

  /// Things to be done before entering the event Loop
  std::cerr << " AnalyzerPrintGeomInfo::beginJob" << std::endl;

  edm::Service<TFileService> fs;

  hPXB_Lay_R = fs->make<TH2D>( "hPXB_Lay_R", "PXB Layer vs R (cm)", 250, 0, 125, 18, -0.5, 17.5 );
  hPXB_Lay_R->Sumw2();

  for ( unsigned int layer = 0; layer < 18; layer++ )
  {
    histoName.str("");  histoName << "hPXB_Lad_Mod_" << layer;
    histoTitle.str(""); histoTitle << "PXB Ladder vs Module, Layer " << layer;
    mapPXB2Lay_hPXB_Lad_Mod[ layer ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                       201, -0.5, 200.5, 201, -0.5, 200.5 );
    mapPXB2Lay_hPXB_Lad_Mod[ layer ]->Sumw2();

    histoName.str("");  histoName << "hPXB_Lad_Phi_" << layer;
    histoTitle.str(""); histoTitle << "PXB Ladder vs #phi, Layer " << layer;
    mapPXB2Lay_hPXB_Lad_Phi[ layer ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                       180, -M_PI, M_PI, 201, -0.5, 200.5 );
    mapPXB2Lay_hPXB_Lad_Phi[ layer ]->Sumw2();

    histoName.str("");  histoName << "hPXB_Mod_Z_" << layer;
    histoTitle.str(""); histoTitle << "PXB Module vs z (cm), Layer " << layer;
    mapPXB2Lay_hPXB_Mod_Z[ layer ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                     600, -300, 300, 201, -0.5, 200.5 );
    mapPXB2Lay_hPXB_Mod_Z[ layer ]->Sumw2();

    histoName.str("");  histoName << "hPXB_Lad_R_" << layer;
    histoTitle.str(""); histoTitle << "PXB Ladder vs R (cm), Layer " << layer;
    mapPXB2Lay_hPXB_Lad_R[ layer ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                       250, 0, 125, 201, -0.5, 200.5 );
    mapPXB2Lay_hPXB_Lad_R[ layer ]->Sumw2();

    histoName.str("");  histoName << "hPXB_Mod_R_" << layer;
    histoTitle.str(""); histoTitle << "PXB Module vs R (cm), Layer " << layer;
    mapPXB2Lay_hPXB_Mod_R[ layer ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                     250, 0, 125, 201, -0.5, 200.5 );
    mapPXB2Lay_hPXB_Mod_R[ layer ]->Sumw2();

    histoName.str("");  histoName << "hPXB_R_" << layer;
    histoTitle.str(""); histoTitle << "PXB R (cm), Layer " << layer;
    mapPXB2Lay_hPXB_R[ layer ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                 250, 0, 125 );
    mapPXB2Lay_hPXB_R[ layer ]->Sumw2();
  }

  hPXF_Disk00_Disk = fs->make<TH2D>( "hPXF_Disk00_Disk", "PXF Disk00 vs Disk", 18, -0.5, 17.5, 58, -0.5, 57.5 );
  hPXF_Disk00_Disk->Sumw2();
  hPXF_Disk00_Side = fs->make<TH2D>( "hPXF_Disk00_Side", "PXF Disk00 vs Side", 7, -3.5, 3.5, 58, -0.5, 57.5 );
  hPXF_Disk00_Side->Sumw2();
  hPXF_Disk00_Z = fs->make<TH2D>( "hPXF_Disk00_Z", "PXF Disk00 vs z (cm)", 600, -300, 300, 58, -0.5, 57.5 );
  hPXF_Disk00_Z->Sumw2();
  hPXF_Disk00_R = fs->make<TH2D>( "hPXF_Disk00_R", "PXF Disk00 vs R (cm)", 250, 0, 125, 58, -0.5, 57.5 );
  hPXF_Disk00_R->Sumw2();

  for ( unsigned int disc = 0; disc < 58; disc++ )
  {
    histoName.str("");  histoName << "hPXF_Pan_Mod_" << disc;
    histoTitle.str(""); histoTitle << "PXF Panel vs Module, Disk00 " << disc;
    mapPXF2Disk00_hPXF_Pan_Mod[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                         201, -0.5, 200.5, 11, -0.5, 10.5 );
    mapPXF2Disk00_hPXF_Pan_Mod[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Bla_Mod_" << disc;
    histoTitle.str(""); histoTitle << "PXF Blade vs Module, Disk00 " << disc;
    mapPXF2Disk00_hPXF_Bla_Mod[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                         201, -0.5, 200.5, 85, -0.5, 84.5 );
    mapPXF2Disk00_hPXF_Bla_Mod[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Bla_Pan_" << disc;
    histoTitle.str(""); histoTitle << "PXF Blade vs Panel, Disk00 " << disc;
    mapPXF2Disk00_hPXF_Bla_Pan[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                         11, -0.5, 10.5, 85, -0.5, 84.5 );
    mapPXF2Disk00_hPXF_Bla_Pan[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Pan_R_" << disc;
    histoTitle.str(""); histoTitle << "PXF Panel vs R (cm), Disk00 " << disc;
    mapPXF2Disk00_hPXF_Pan_R[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                       250, 0, 125, 11, -0.5, 10.5 );
    mapPXF2Disk00_hPXF_Pan_R[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Mod_R_" << disc;
    histoTitle.str(""); histoTitle << "PXF Module vs R (cm), Disk00 " << disc;
    mapPXF2Disk00_hPXF_Mod_R[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                       250, 0, 125, 201, -0.5, 200.5);
    mapPXF2Disk00_hPXF_Mod_R[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Bla_R_" << disc;
    histoTitle.str(""); histoTitle << "PXF Blade vs R (cm), Disk00 " << disc;
    mapPXF2Disk00_hPXF_Bla_R[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                       250, 0, 125, 85, -0.5, 84.5 );
    mapPXF2Disk00_hPXF_Bla_R[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Pan_Z_" << disc;
    histoTitle.str(""); histoTitle << "PXF Panel vs z (cm), Disk00 " << disc;
    mapPXF2Disk00_hPXF_Pan_Z[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                       600, -300, 300, 11, -0.5, 10.5 );
    mapPXF2Disk00_hPXF_Pan_Z[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Mod_Z_" << disc;
    histoTitle.str(""); histoTitle << "PXF Module vs z (cm), Disk00 " << disc;
    mapPXF2Disk00_hPXF_Mod_Z[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                       600, -300, 300, 201, -0.5, 200.5);
    mapPXF2Disk00_hPXF_Mod_Z[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Bla_Z_" << disc;
    histoTitle.str(""); histoTitle << "PXF Blade vs z (cm), Disk00 " << disc;
    mapPXF2Disk00_hPXF_Bla_Z[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                       600, -300, 300, 85, -0.5, 84.5 );
    mapPXF2Disk00_hPXF_Bla_Z[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Pan_Phi_" << disc;
    histoTitle.str(""); histoTitle << "PXF Panel vs #phi, Disk00 " << disc;
    mapPXF2Disk00_hPXF_Pan_Phi[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                         180, -M_PI, M_PI, 11, -0.5, 10.5 );
    mapPXF2Disk00_hPXF_Pan_Phi[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Mod_Phi_" << disc;
    histoTitle.str(""); histoTitle << "PXF Module vs #phi, Disk00 " << disc;
    mapPXF2Disk00_hPXF_Mod_Phi[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                         180, -M_PI, M_PI, 201, -0.5, 200.5);
    mapPXF2Disk00_hPXF_Mod_Phi[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Bla_Phi_" << disc;
    histoTitle.str(""); histoTitle << "PXF Blade vs #phi, Disk00 " << disc;
    mapPXF2Disk00_hPXF_Bla_Phi[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                         180, -M_PI, M_PI, 85, -0.5, 84.5 );
    mapPXF2Disk00_hPXF_Bla_Phi[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Disk_R_" << disc;
    histoTitle.str(""); histoTitle << "PXF Disk vs R (cm), Disk00 " << disc;
    mapPXF2Disk00_hPXF_Disk_R[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                        250, 0, 125, 58, -0.5, 57.5 );
    mapPXF2Disk00_hPXF_Disk_R[ disc ]->Sumw2();

    histoName.str("");  histoName << "hPXF_Disk_Z_" << disc;
    histoTitle.str(""); histoTitle << "PXF Disk vs z (cm), Disk00 " << disc;
    mapPXF2Disk00_hPXF_Disk_Z[ disc ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                        600, -300, 300, 58, -0.5, 57.5 );
    mapPXF2Disk00_hPXF_Disk_Z[ disc ]->Sumw2();
  }

  /// End of things to be done before entering the event Loop
}

//////////
// ANALYZE
void AnalyzerPrintGeomInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  /// Geometry handles etc
  edm::ESHandle<TrackerGeometry>                               geometryHandle;
  const TrackerGeometry*                                       theGeometry;
  edm::ESHandle<StackedTrackerGeometry>           stackedGeometryHandle;
  const StackedTrackerGeometry*                   theStackedGeometry;
  StackedTrackerGeometry::StackContainerIterator  StackedTrackerIterator;

  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
  theGeometry = &(*geometryHandle);
  /// Set pointers to Stacked Modules
  iSetup.get<StackedTrackerGeometryRecord>().get(stackedGeometryHandle);
  theStackedGeometry = stackedGeometryHandle.product(); /// Note this is different 
                                                        /// from the "global" geometry

  /// GeometricDet
  /// This is used to get more information on Modules
  const GeometricDet* theTrackerGeometricDet = theGeometry->trackerDet();
  std::vector< const GeometricDet* > theModules = theTrackerGeometricDet->deepComponents();
 
  /// Loop over Stacks
  /// Count how many Stacks there are
  /// divide it by Layer etc...

  /// Maps to store information by layer
  /// Compare to StackedTrackerGeometryBuilder--->Checksums and final summary

  /// PXB
  std::map< uint32_t, uint32_t >                   rodsPerLayer;
  std::map< uint32_t, uint32_t >                   modsPerRodPerLayer;  /// Assumes all Rods are identical within a Layer
  std::map< uint32_t, std::map< uint32_t, bool > > inOutPerRodPerLayer; /// TRUE if innerMod = outerMod + 1

  std::map< uint32_t, double > innerModRadPerLayer;
  std::map< uint32_t, double > outerModRadPerLayer;
  std::map< uint32_t, double > separationPerLayer;
  std::map< uint32_t, double > maxLengthPerLayer;
  std::map< uint32_t, double > activeSurfacePerLayer;

  std::map< uint32_t, float > innerModXPitchPerLayer;
  std::map< uint32_t, float > outerModXPitchPerLayer;
  std::map< uint32_t, float > innerModYPitchPerLayer;
  std::map< uint32_t, float > outerModYPitchPerLayer;

  std::map< uint32_t, uint32_t > innerModChannelsPerLayer;
  std::map< uint32_t, uint32_t > outerModChannelsPerLayer;

  std::map< uint32_t, uint32_t > innerModXROCPerLayer;
  std::map< uint32_t, uint32_t > outerModXROCPerLayer;
  std::map< uint32_t, uint32_t > innerModYROCPerLayer;
  std::map< uint32_t, uint32_t > outerModYROCPerLayer;
  std::map< uint32_t, uint32_t > innerModTotROCPerLayer;
  std::map< uint32_t, uint32_t > outerModTotROCPerLayer;

  std::map< uint32_t, uint32_t > innerModColumnsPerLayer;
  std::map< uint32_t, uint32_t > outerModColumnsPerLayer;
  std::map< uint32_t, uint32_t > innerModRowsPerLayer;
  std::map< uint32_t, uint32_t > outerModRowsPerLayer;

  /// PXF
  std::map< uint32_t, uint32_t >                       ringsPerDisk;
  std::map< uint32_t, std::map< uint32_t, uint32_t > > modsPerRingPerDisk;  /// Assumes all Rings are different within a Disk
  std::map< uint32_t, std::map< uint32_t, bool > >     inOutPerRingPerDisk; /// TRUE if innerMod = outerMod + 1

  std::map< uint32_t, std::map< uint32_t, double > > innerModZPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, double > > outerModZPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, double > > separationPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, double > > maxLengthPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, double > > activeSurfacePerRingPerDisk;

  std::map< uint32_t, std::map< uint32_t, float > > innerModXPitchPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, float > > outerModXPitchPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, float > > innerModYPitchPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, float > > outerModYPitchPerRingPerDisk;

  std::map< uint32_t, std::map< uint32_t, uint32_t > > innerModChannelsPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, uint32_t > > outerModChannelsPerRingPerDisk;

  std::map< uint32_t, std::map< uint32_t, uint32_t > > innerModXROCPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, uint32_t > > outerModXROCPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, uint32_t > > innerModYROCPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, uint32_t > > outerModYROCPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, uint32_t > > innerModTotROCPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, uint32_t > > outerModTotROCPerRingPerDisk;

  std::map< uint32_t, std::map< uint32_t, uint32_t > > innerModColumnsPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, uint32_t > > outerModColumnsPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, uint32_t > > innerModRowsPerRingPerDisk;
  std::map< uint32_t, std::map< uint32_t, uint32_t > > outerModRowsPerRingPerDisk;


  /// Validation maps for Geometry
  /// Loop only on different sets of sub-tracker detectors
  /// PXB
  std::vector<GeomDet*> pxbcont = theGeometry->detsPXB();  
  for ( unsigned int i = 0; i < pxbcont.size(); i++ )
  {
    DetId id0 = pxbcont[i]->geographicalId();
    PXBDetId detId0 = PXBDetId(id0.rawId());

    unsigned int layer0 = detId0.layer();
    unsigned int ladder0 = detId0.ladder();
    unsigned int module0 = detId0.module();

    /// Get the position of each sensor
    double r0 = pxbcont[i]->position().perp();
    double z0 = pxbcont[i]->position().z();
    double phi0 = pxbcont[i]->position().phi();

    hPXB_Lay_R->Fill( r0, layer0 );
    mapPXB2Lay_hPXB_Lad_Mod[ layer0 ]->Fill( module0, ladder0 );
    mapPXB2Lay_hPXB_Lad_Phi[ layer0 ]->Fill( phi0, ladder0 );
    mapPXB2Lay_hPXB_Mod_Z[ layer0 ]->Fill( z0, module0 );
    mapPXB2Lay_hPXB_Lad_R[ layer0 ]->Fill( r0, ladder0 );
    mapPXB2Lay_hPXB_Mod_R[ layer0 ]->Fill( r0, module0 );
    mapPXB2Lay_hPXB_R[ layer0 ]->Fill( r0 );
  }

  /// PXF
  std::vector<GeomDet*> pxfcont = theGeometry->detsPXF();
  for ( unsigned int i = 0; i < pxfcont.size(); i++ )
  {
    DetId id0 = pxfcont[i]->geographicalId();
    PXFDetId detId0 = PXFDetId(id0.rawId());

    unsigned int side0 = detId0.side();
    unsigned int disk0 = detId0.disk();
    unsigned int blade0;
    if (disk0 < 4) blade0 = detId0.blade();
    else blade0 = detId0.ring();
    unsigned int panel0 = detId0.panel();
    unsigned int module0 = detId0.module();

    unsigned int disk00 = 2*disk0 + side0%2;

    /// Get the position of each sensor
    double r0 = pxfcont[i]->position().perp();
    double z0 = pxfcont[i]->position().z();
    double phi0 = pxfcont[i]->position().phi();

    hPXF_Disk00_Disk->Fill( disk0, disk00 );
    hPXF_Disk00_Side->Fill( side0, disk00 );
    hPXF_Disk00_Z->Fill( z0, disk00 );
    hPXF_Disk00_R->Fill( r0, disk00 );

    mapPXF2Disk00_hPXF_Pan_Mod[ disk00 ]->Fill( module0, panel0 );
    mapPXF2Disk00_hPXF_Bla_Mod[ disk00 ]->Fill( module0, blade0 );
    mapPXF2Disk00_hPXF_Bla_Pan[ disk00 ]->Fill( panel0, blade0 );
    mapPXF2Disk00_hPXF_Pan_R[ disk00 ]->Fill( r0, panel0 );
    mapPXF2Disk00_hPXF_Pan_Z[ disk00 ]->Fill( z0, panel0 );
    mapPXF2Disk00_hPXF_Pan_Phi[ disk00 ]->Fill( phi0, panel0 );
    mapPXF2Disk00_hPXF_Mod_R[ disk00 ]->Fill( r0, module0 );
    mapPXF2Disk00_hPXF_Mod_Z[ disk00 ]->Fill( z0, module0 );
    mapPXF2Disk00_hPXF_Mod_Phi[ disk00 ]->Fill( phi0, module0 );
    mapPXF2Disk00_hPXF_Bla_R[ disk00 ]->Fill( r0, blade0 );
    mapPXF2Disk00_hPXF_Bla_Z[ disk00 ]->Fill( z0, blade0 );
    mapPXF2Disk00_hPXF_Bla_Phi[ disk00 ]->Fill( phi0, blade0 );
    mapPXF2Disk00_hPXF_Disk_R[ disk00 ]->Fill( r0, disk0 );
    mapPXF2Disk00_hPXF_Disk_Z[ disk00 ]->Fill( z0, disk0 );
  }

  /// Loop over the detector elements
  /// Information from each sensor in each PtModule must be retrieved!
  for ( StackedTrackerIterator = theStackedGeometry->stacks().begin();
        StackedTrackerIterator != theStackedGeometry->stacks().end();
        ++StackedTrackerIterator )
  {
    StackedTrackerDetUnit* stackDetUnit = *StackedTrackerIterator;
    StackedTrackerDetId stackDetId = stackDetUnit->Id();
    assert(stackDetUnit == theStackedGeometry->idToStack(stackDetId));

    /// GeomDet and GeomDetUnit are needed to access each
    /// DetId and topology and geometric features
    /// Convert to specific DetId
    const GeomDet* det0 = theStackedGeometry->idToDet(stackDetId, 0);
    const GeomDet* det1 = theStackedGeometry->idToDet(stackDetId, 1);
    const GeomDetUnit* detUnit0 = theStackedGeometry->idToDetUnit(stackDetId, 0);
    const GeomDetUnit* detUnit1 = theStackedGeometry->idToDetUnit(stackDetId, 1);
    
    /// Barrel
    if ( stackDetId.isBarrel() )
    { 
      PXBDetId detId0 = PXBDetId(det0->geographicalId().rawId());
      PXBDetId detId1 = PXBDetId(det1->geographicalId().rawId());

      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iStack = stackDetId.iLayer();
      uint32_t iPhi   = stackDetId.iPhi();
      uint32_t iZ     = stackDetId.iZ();

      /// Get index-coordinates of each sensor from DetId
      uint32_t iLayer0  = detId0.layer();
      uint32_t iLayer1  = detId1.layer();
      uint32_t iRod0    = detId0.ladder();
      uint32_t iRod1    = detId1.ladder();
      uint32_t iModule0 = detId0.module();
      uint32_t iModule1 = detId1.module();

      if ( DebugMode )
      {
        std::cerr << "Stack: " << iStack << " from Layers:  " << iLayer0 << " " << iLayer1 << std::endl;
        std::cerr << "Phi:   " << iPhi <<   " from Rods:    " << iRod0 << " " << iRod1 << std::endl;
        std::cerr << "Z:     " << iZ <<     " from Modules: " << iModule0 << " " << iModule1 << " >>> " << (iModule0 == iModule1 + 1) << std::endl;
      }

      /// Store the size of rods and layers
      if ( rodsPerLayer.find(iStack) == rodsPerLayer.end() )
        rodsPerLayer.insert( std::make_pair(iStack, 0) );
      if ( iPhi >= rodsPerLayer[iStack] )
        rodsPerLayer[iStack] = iPhi;

      if ( modsPerRodPerLayer.find(iStack) == modsPerRodPerLayer.end() )
        modsPerRodPerLayer.insert( std::make_pair(iStack, 0) );
      if ( iZ >= modsPerRodPerLayer[iStack] )
        modsPerRodPerLayer[iStack] = iZ;

      /// This is a debug control that checks if
      /// the pairing in Z follows index-based logic
      /// of innerZ = outerZ + 1 or not
      if ( inOutPerRodPerLayer.find(iStack) == inOutPerRodPerLayer.end() )
      {
        /// New Layer, new Rod
        std::map< uint32_t, bool > tempMap;
        tempMap.insert( std::make_pair(iPhi, (iModule1 == (iModule0 + 1))) );
        inOutPerRodPerLayer.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        /// Existing Layer
        std::map< uint32_t, bool > tempMap = inOutPerRodPerLayer.find(iStack)->second;
        if ( tempMap.find(iPhi) == tempMap.end() )
        {
          /// New Rod
          tempMap.insert( std::make_pair(iPhi, (iModule1 == (iModule0 + 1))) );
        }
        else
        {
          /// Existing Rod
          tempMap.find(iPhi)->second = tempMap.find(iPhi)->second && (iModule1 == (iModule0 + 1));
        }
        inOutPerRodPerLayer.find(iStack)->second = tempMap;
      }

      /// Get the radius of each sensor and its z
      double r0 = det0->position().perp();
      double r1 = det1->position().perp();
      double z0 = det0->position().z();
      double z1 = det1->position().z();

      /// Store radii to find average radius of inner and outer sensor
      /// in each Stack layer, and sensor separation in Stacks as well
      if ( innerModRadPerLayer.find(iStack) == innerModRadPerLayer.end() )
        innerModRadPerLayer.insert( std::make_pair(iStack, 0) );
      innerModRadPerLayer[iStack] += r0;
      if ( outerModRadPerLayer.find(iStack) == outerModRadPerLayer.end() )
        outerModRadPerLayer.insert( std::make_pair(iStack, 0) );
      outerModRadPerLayer[iStack] += r1;
      if ( separationPerLayer.find(iStack) == separationPerLayer.end() )
        separationPerLayer.insert( std::make_pair(iStack, 0) );
      separationPerLayer[iStack] += fabs(r1-r0);

      /// Store max length
      if ( maxLengthPerLayer.find(iStack) == maxLengthPerLayer.end() )
        maxLengthPerLayer.insert( std::make_pair(iStack, 0) );
      if ( fabs(z0) > maxLengthPerLayer[iStack] )
        maxLengthPerLayer[iStack] = fabs(z0);
      if ( fabs(z1) > maxLengthPerLayer[iStack] )
        maxLengthPerLayer[iStack] = fabs(z1);

      /// Find pixel pitch and topology related information
      const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( detUnit0 );
      const PixelGeomDetUnit* pix1 = dynamic_cast< const PixelGeomDetUnit* >( detUnit1 );
      const PixelTopology* top0 = dynamic_cast<const PixelTopology*>( & (pix0->specificTopology() ));
      const PixelTopology* top1 = dynamic_cast<const PixelTopology*>( & (pix1->specificTopology() ));
      std::pair< float, float > pitch0 = top0->pitch();
      std::pair< float, float > pitch1 = top1->pitch();

      /// Store pixel pitch etc
      /// NOTE: this assumes that innermost sensor in a stack layer
      /// has ALWAYS the very same pixel pitch
      if ( innerModXPitchPerLayer.find(iStack) == innerModXPitchPerLayer.end() )
        innerModXPitchPerLayer.insert( std::make_pair(iStack, 0) );
      innerModXPitchPerLayer[iStack] += pitch0.first;
      if ( outerModXPitchPerLayer.find(iStack) == outerModXPitchPerLayer.end() )
        outerModXPitchPerLayer.insert( std::make_pair(iStack, 0) );
      outerModXPitchPerLayer[iStack] += pitch1.first;

      if ( innerModYPitchPerLayer.find(iStack) == innerModYPitchPerLayer.end() )
        innerModYPitchPerLayer.insert( std::make_pair(iStack, 0) );
      innerModYPitchPerLayer[iStack] += pitch0.second;
      if ( outerModYPitchPerLayer.find(iStack) == outerModYPitchPerLayer.end() )
        outerModYPitchPerLayer.insert( std::make_pair(iStack, 0) );
      outerModYPitchPerLayer[iStack] += pitch1.second;

      if ( innerModChannelsPerLayer.find(iStack) == innerModChannelsPerLayer.end() )
        innerModChannelsPerLayer.insert( std::make_pair(iStack, 0) );
      innerModChannelsPerLayer[iStack] += top0->nrows()*top0->ncolumns();
      if ( outerModChannelsPerLayer.find(iStack) == outerModChannelsPerLayer.end() )
        outerModChannelsPerLayer.insert( std::make_pair(iStack, 0) );
      outerModChannelsPerLayer[iStack] += top1->nrows()*top1->ncolumns();

      if ( innerModRowsPerLayer.find(iStack) == innerModRowsPerLayer.end() )
        innerModRowsPerLayer.insert( std::make_pair(iStack, 0) );
      innerModRowsPerLayer[iStack] += top0->nrows();
      if ( outerModRowsPerLayer.find(iStack) == outerModRowsPerLayer.end() )
        outerModRowsPerLayer.insert( std::make_pair(iStack, 0) );
      outerModRowsPerLayer[iStack] += top1->nrows();

      if ( innerModColumnsPerLayer.find(iStack) == innerModColumnsPerLayer.end() )
        innerModColumnsPerLayer.insert( std::make_pair(iStack, 0) );
      innerModColumnsPerLayer[iStack] += top0->ncolumns();
      if ( outerModColumnsPerLayer.find(iStack) == outerModColumnsPerLayer.end() )
        outerModColumnsPerLayer.insert( std::make_pair(iStack, 0) );
      outerModColumnsPerLayer[iStack] += top1->ncolumns();

      /// This loops on GeometricDet to get more detailed information
      /// In particular, we are interested in active surface
      if ( activeSurfacePerLayer.find(iStack) == activeSurfacePerLayer.end() )
        activeSurfacePerLayer.insert( std::make_pair( iStack, 0 ) );
      if ( innerModXROCPerLayer.find(iStack) == innerModXROCPerLayer.end() )
        innerModXROCPerLayer.insert( std::make_pair(iStack, 0) );
      if ( innerModYROCPerLayer.find(iStack) == innerModYROCPerLayer.end() )
        innerModYROCPerLayer.insert( std::make_pair(iStack, 0) );
      if ( innerModTotROCPerLayer.find(iStack) == innerModTotROCPerLayer.end() )
        innerModTotROCPerLayer.insert( std::make_pair(iStack, 0) );    
      if ( outerModXROCPerLayer.find(iStack) == outerModXROCPerLayer.end() )
        outerModXROCPerLayer.insert( std::make_pair(iStack, 0) );
      if ( outerModYROCPerLayer.find(iStack) == outerModYROCPerLayer.end() )
        outerModYROCPerLayer.insert( std::make_pair(iStack, 0) );
      if ( outerModTotROCPerLayer.find(iStack) == outerModTotROCPerLayer.end() )
        outerModTotROCPerLayer.insert( std::make_pair(iStack, 0) );

      bool fastExit0 = false;
      bool fastExit1 = false;
      for ( unsigned int iMod = 0; iMod < theModules.size() && !(fastExit0 && fastExit1); iMod++ )
      {
        const GeometricDet* thisModule = theModules.at(iMod);
        if ( thisModule->geographicalId() == det0->geographicalId() ||
             thisModule->geographicalId() == det1->geographicalId() )
        {
          double thisSurface = thisModule->bounds()->length()*thisModule->bounds()->width(); /// cm^2
          activeSurfacePerLayer[iStack] += thisSurface;

          if ( thisModule->geographicalId() == det0->geographicalId() )
          {
            innerModXROCPerLayer[iStack] += thisModule->pixROCx();
            innerModYROCPerLayer[iStack] += thisModule->pixROCy();
            innerModTotROCPerLayer[iStack] += thisModule->pixROCx()*thisModule->pixROCy();
            fastExit0 = true;
          }
          if ( thisModule->geographicalId() == det1->geographicalId() )
          {
            outerModXROCPerLayer[iStack] += thisModule->pixROCx();
            outerModYROCPerLayer[iStack] += thisModule->pixROCy();
            outerModTotROCPerLayer[iStack] += thisModule->pixROCx()*thisModule->pixROCy();
            fastExit1 = true;
          }
        }
      }
    }

    /// Endcap
    if ( stackDetId.isEndcap() )
    {
      PXFDetId detId0 = PXFDetId(det0->geographicalId().rawId());
      PXFDetId detId1 = PXFDetId(det1->geographicalId().rawId());

      /// Get the Stack, iPhi, iZ from StackedTrackerDetId
      uint32_t iSide  = stackDetId.iSide();
      uint32_t iStack = stackDetId.iDisk();
      uint32_t iRing  = stackDetId.iRing();
      uint32_t iPhi   = stackDetId.iPhi();

      /// Get index-coordinates of each sensor from DetId
      uint32_t iDisk0   = detId0.disk();
      uint32_t iRing0   = detId0.ring();
      uint32_t iModule0 = detId0.module();
      uint32_t iDisk1   = detId1.disk();
      uint32_t iRing1   = detId1.ring();
      uint32_t iModule1 = detId1.module();

      if ( DebugMode )
      {
        std::cerr << "Side:  " << iSide << std::endl;
        std::cerr << "Stack: " << iStack << " from Disks:   " << iDisk0 << " " << iDisk1 << std::endl;
        std::cerr << "Ring:  " << iRing <<  " from Rings:   " << iRing0 << " " << iRing1 << std::endl;
        std::cerr << "Phi:   " << iPhi <<   " from Modules: " << iModule0 << " " << iModule1 << " >>> " << (iModule0 == iModule1 + 1) << std::endl;
      }

      /// Store the size of rods and layers
      if ( ringsPerDisk.find(iStack) == ringsPerDisk.end() )
        ringsPerDisk.insert( std::make_pair(iStack, 0) );
      if ( iRing >= ringsPerDisk[iStack] )
        ringsPerDisk[iStack] = iRing;

      if ( modsPerRingPerDisk.find(iStack) == modsPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        modsPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      if ( modsPerRingPerDisk.find(iStack)->second.find(iRing) == modsPerRingPerDisk.find(iStack)->second.end() )
        modsPerRingPerDisk.find(iStack)->second.insert( std::make_pair( iRing, 0 ) );
      if ( iRing >= ringsPerDisk[iStack] )
        ringsPerDisk[iStack] = iRing;
      if ( iPhi >= modsPerRingPerDisk.find(iStack)->second.find(iRing)->second )
        modsPerRingPerDisk.find(iStack)->second.find(iRing)->second = iPhi;

      /// This is a debug control that checks if
      /// the pairing in Phi follows index-based logic
      /// of innerPhi = outerPhi + 1 or not
      if ( inOutPerRingPerDisk.find(iStack) == inOutPerRingPerDisk.end() )
      {
        /// New Disk, new Ring
        std::map< uint32_t, bool > tempMap;
        tempMap.insert( std::make_pair(iRing, (iModule1 == (iModule0 + 1))) );
        inOutPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        /// Existing Disk
        std::map< uint32_t, bool > tempMap = inOutPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          /// New Ring
          tempMap.insert( std::make_pair(iRing, (iModule1 == (iModule0 + 1))) );
        }
        else {
          /// Existing Ring
          tempMap.find(iRing)->second = tempMap.find(iRing)->second && (iModule1 == (iModule0 + 1));
        }
        inOutPerRingPerDisk.find(iStack)->second = tempMap;
      }

      /// Get the z of each sensor
      double z0 = det0->position().z();
      double z1 = det1->position().z();

      /// Store radii to find average radius of inner and outer sensor
      /// in each Stack layer, and sensor separation in Stacks as well
      if ( innerModZPerRingPerDisk.find(iStack) == innerModZPerRingPerDisk.end() )
      {
        std::map< uint32_t, double > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        innerModZPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, double > tempMap = innerModZPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          innerModZPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      innerModZPerRingPerDisk[iStack].find(iRing)->second += fabs(z0);
      if ( outerModZPerRingPerDisk.find(iStack) == outerModZPerRingPerDisk.end() )
      {
        std::map< uint32_t, double > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        outerModZPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, double > tempMap = outerModZPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          outerModZPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      outerModZPerRingPerDisk[iStack].find(iRing)->second += fabs(z1);
      if ( separationPerRingPerDisk.find(iStack) == separationPerRingPerDisk.end() )
      {
        std::map< uint32_t, double > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        separationPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, double > tempMap = separationPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          separationPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      separationPerRingPerDisk[iStack].find(iRing)->second += fabs(z1 - z0);

      /// Find pixel pitch and topology related information
      const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( detUnit0 );
      const PixelGeomDetUnit* pix1 = dynamic_cast< const PixelGeomDetUnit* >( detUnit1 );
      const PixelTopology* top0 = dynamic_cast<const PixelTopology*>( & (pix0->specificTopology() ));
      const PixelTopology* top1 = dynamic_cast<const PixelTopology*>( & (pix1->specificTopology() ));
      std::pair< float, float > pitch0 = top0->pitch();
      std::pair< float, float > pitch1 = top1->pitch();

      /// Store pixel pitch etc
      /// NOTE: this assumes that innermost sensor in a stack layer
      /// has ALWAYS the very same pixel pitch
      if ( innerModXPitchPerRingPerDisk.find(iStack) == innerModXPitchPerRingPerDisk.end() )
      {
        std::map< uint32_t, float > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        innerModXPitchPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, float > tempMap = innerModXPitchPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          innerModXPitchPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      innerModXPitchPerRingPerDisk[iStack].find(iRing)->second += pitch0.first;
      if ( outerModXPitchPerRingPerDisk.find(iStack) == outerModXPitchPerRingPerDisk.end() )
      {
        std::map< uint32_t, float > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        outerModXPitchPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, float > tempMap = outerModXPitchPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          outerModXPitchPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      outerModXPitchPerRingPerDisk[iStack].find(iRing)->second += pitch1.first;

      if ( innerModYPitchPerRingPerDisk.find(iStack) == innerModYPitchPerRingPerDisk.end() )
      {
        std::map< uint32_t, float > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        innerModYPitchPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, float > tempMap = innerModYPitchPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          innerModYPitchPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      innerModYPitchPerRingPerDisk[iStack].find(iRing)->second += pitch0.second;
      if ( outerModYPitchPerRingPerDisk.find(iStack) == outerModYPitchPerRingPerDisk.end() )
      {
        std::map< uint32_t, float > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        outerModYPitchPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, float > tempMap = outerModYPitchPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          outerModYPitchPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      outerModYPitchPerRingPerDisk[iStack].find(iRing)->second += pitch1.second;

      if ( innerModChannelsPerRingPerDisk.find(iStack) == innerModChannelsPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        innerModChannelsPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, uint32_t > tempMap = innerModChannelsPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          innerModChannelsPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      innerModChannelsPerRingPerDisk[iStack].find(iRing)->second += top0->nrows()*top0->ncolumns();
      if ( outerModChannelsPerRingPerDisk.find(iStack) == outerModChannelsPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        outerModChannelsPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, uint32_t > tempMap = outerModChannelsPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          outerModChannelsPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      outerModChannelsPerRingPerDisk[iStack].find(iRing)->second += top1->nrows()*top1->ncolumns();

      if ( innerModRowsPerRingPerDisk.find(iStack) == innerModRowsPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        innerModRowsPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, uint32_t > tempMap = innerModRowsPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          innerModRowsPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      innerModRowsPerRingPerDisk[iStack].find(iRing)->second += top0->nrows();
      if ( outerModRowsPerRingPerDisk.find(iStack) == outerModRowsPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        outerModRowsPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, uint32_t > tempMap = outerModRowsPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          outerModRowsPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      outerModRowsPerRingPerDisk[iStack].find(iRing)->second += top1->nrows();

      if ( innerModColumnsPerRingPerDisk.find(iStack) == innerModColumnsPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        innerModColumnsPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, uint32_t > tempMap = innerModColumnsPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          innerModColumnsPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      innerModColumnsPerRingPerDisk[iStack].find(iRing)->second += top0->ncolumns();
      if ( outerModColumnsPerRingPerDisk.find(iStack) == outerModColumnsPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        outerModColumnsPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, uint32_t > tempMap = outerModColumnsPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          outerModColumnsPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      outerModColumnsPerRingPerDisk[iStack].find(iRing)->second += top1->ncolumns();

      /// This loops on GeometricDet to get more detailed information
      /// In particular, we are interested in active surface
      if ( activeSurfacePerRingPerDisk.find(iStack) == activeSurfacePerRingPerDisk.end() )
      {
        std::map< uint32_t, double > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        activeSurfacePerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, double > tempMap = activeSurfacePerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          activeSurfacePerRingPerDisk.find(iStack)->second = tempMap;
        }
      }

      if ( innerModXROCPerRingPerDisk.find(iStack) == innerModXROCPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        innerModXROCPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, uint32_t > tempMap = innerModXROCPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          innerModXROCPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      if ( outerModXROCPerRingPerDisk.find(iStack) == outerModXROCPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        outerModXROCPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, uint32_t > tempMap = outerModXROCPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          outerModXROCPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }

      if ( innerModYROCPerRingPerDisk.find(iStack) == innerModYROCPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        innerModYROCPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, uint32_t > tempMap = innerModYROCPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          innerModYROCPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      if ( outerModYROCPerRingPerDisk.find(iStack) == outerModYROCPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        outerModYROCPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, uint32_t > tempMap = outerModYROCPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          outerModYROCPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }

      if ( innerModTotROCPerRingPerDisk.find(iStack) == innerModTotROCPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        innerModTotROCPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, uint32_t > tempMap = innerModTotROCPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          innerModTotROCPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }
      if ( outerModTotROCPerRingPerDisk.find(iStack) == outerModTotROCPerRingPerDisk.end() )
      {
        std::map< uint32_t, uint32_t > tempMap;
        tempMap.insert( std::make_pair(iRing, 0) );
        outerModTotROCPerRingPerDisk.insert( std::make_pair(iStack, tempMap) );
      }
      else
      {
        std::map< uint32_t, uint32_t > tempMap = outerModTotROCPerRingPerDisk.find(iStack)->second;
        if ( tempMap.find(iRing) == tempMap.end() )
        {
          tempMap.insert( std::make_pair(iRing, 0) );
          outerModTotROCPerRingPerDisk.find(iStack)->second = tempMap;
        }
      }

      bool fastExit0 = false;
      bool fastExit1 = false;
      for ( unsigned int iMod = 0; iMod < theModules.size() && !fastExit0 && !fastExit1; iMod++ )
      {
        const GeometricDet* thisModule = theModules.at(iMod);
        if ( thisModule->geographicalId() == det0->geographicalId() ||
             thisModule->geographicalId() == det1->geographicalId() )
        {
          double thisSurface = thisModule->bounds()->length()*thisModule->bounds()->width(); /// cm^2
          activeSurfacePerRingPerDisk[iStack].find(iRing)->second += thisSurface;

          if ( thisModule->geographicalId() == det0->geographicalId() )
          {
            innerModXROCPerRingPerDisk[iStack].find(iRing)->second += thisModule->pixROCx();
            innerModYROCPerRingPerDisk[iStack].find(iRing)->second += thisModule->pixROCy();
            innerModTotROCPerRingPerDisk[iStack].find(iRing)->second += thisModule->pixROCx()*thisModule->pixROCy();
            fastExit0 = true;
          }
          if ( thisModule->geographicalId() == det1->geographicalId() )
          {
            outerModXROCPerRingPerDisk[iStack].find(iRing)->second += thisModule->pixROCx();
            outerModYROCPerRingPerDisk[iStack].find(iRing)->second += thisModule->pixROCy();
            outerModTotROCPerRingPerDisk[iStack].find(iRing)->second += thisModule->pixROCx()*thisModule->pixROCy();
            fastExit1 = true;
          }
        }
      }
    }
  }

  outputFile << " *****************" << std::endl;
  outputFile << " * FINAL SUMMARY *" << std::endl;
  outputFile << " *****************" << std::endl;

  for ( std::map< uint32_t, uint32_t >::iterator iterMap = rodsPerLayer.begin();
        iterMap != rodsPerLayer.end();
        ++iterMap )
  {
    /// Index-coordinates and number of modules 
    outputFile << std::endl;
    outputFile << " - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl; 
    outputFile << "  Barrel Stack Layer: " << iterMap->first << std::endl;
    outputFile << "   # Rods:            " << iterMap->second << std::endl;
    outputFile << "   # Modules/Rod:     " << modsPerRodPerLayer[iterMap->first] << std::endl;
    outputFile << "   # Modules          " << modsPerRodPerLayer[iterMap->first] * iterMap->second << std::endl;
    if ( DebugMode )
    {
      outputFile << "     In/Out Correct Order ? ";
      for ( std::map< uint32_t, bool >::iterator iterAux = inOutPerRodPerLayer[iterMap->first].begin();
            iterAux != inOutPerRodPerLayer[iterMap->first].end();
            ++iterAux )
      {
        outputFile << iterAux->second;
      }
      outputFile << std::endl;
    }

    /// Average radii and separation
    outputFile << "  Inner sensor average radius:  " <<
      innerModRadPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << " [cm]" << std::endl;
    outputFile << "    Outer sensor average radius:  " <<
      outerModRadPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << " [cm]" << std::endl;
    outputFile << "      Average sensor separation:    " <<
      separationPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) * 10 << " [mm]" << std::endl;

    /// Stack length and surface
    outputFile << "  Max length of Stack Layer:    " <<
      maxLengthPerLayer[iterMap->first] << " [cm]" << std::endl;
    outputFile << "  Total active surface:         " <<
      activeSurfacePerLayer[iterMap->first] << " [cm^2]" << std::endl;

    /// Topology
    outputFile << "  Inner sensor channels:        " <<
      innerModChannelsPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << std::endl;
    outputFile << "               rows:            " <<
      innerModRowsPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << std::endl;
    outputFile << "               columns:         " <<
      innerModColumnsPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << std::endl;
    outputFile << "  Inner sensor average pitch X: " <<
      double(int(0.5+(10000*innerModXPitchPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) ))) << " [um]" << std::endl;
    outputFile << "                             Y: " <<
      double(int(0.5+(10000*innerModYPitchPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) ))) << " [um]" << std::endl;
    outputFile << "  Inner sensor ROCs X:          " <<
      innerModXROCPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << std::endl;
    outputFile << "                    Y:          " <<
      innerModYROCPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << std::endl;
    outputFile << "                    total:      " <<
      innerModTotROCPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << std::endl;
    outputFile << "  Outer sensor channels:        " <<
      outerModChannelsPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << std::endl;
    outputFile << "               rows:            " <<
      outerModRowsPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << std::endl;
    outputFile << "               columns:         " <<
      outerModColumnsPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << std::endl;
    outputFile << "  Outer sensor average pitch X: " <<
      double(int(0.5+(10000*outerModXPitchPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) ))) << " [um]" << std::endl;
    outputFile << "                             Y: " <<
      double(int(0.5+(10000*outerModYPitchPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) ))) << " [um]" << std::endl;
    outputFile << "  Outer sensor ROCs X:          " <<
      outerModXROCPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << std::endl;
    outputFile << "                    Y:          " <<
      outerModYROCPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << std::endl;
    outputFile << "                    total:      " <<
      outerModTotROCPerLayer[iterMap->first]/(double)(modsPerRodPerLayer[iterMap->first] * iterMap->second) << std::endl;
  }

  for ( std::map< uint32_t, uint32_t >::iterator iterMap = ringsPerDisk.begin();
        iterMap != ringsPerDisk.end();
        ++iterMap )
  {
    /// Index-coordinates and number of modules 
    outputFile << std::endl;
    outputFile << " - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl; 
    outputFile << "  Endcap Stack Layer: " << iterMap->first << std::endl;
    outputFile << "   # Rings:           " << iterMap->second << std::endl;

    uint32_t countMods = 0;

    for ( std::map< uint32_t, uint32_t >::iterator iterOther = modsPerRingPerDisk[iterMap->first].begin();
          iterOther != modsPerRingPerDisk[iterMap->first].end();
          iterOther++ )
    {
      outputFile << "   # Modules/Ring:    " << iterOther->second << " in Ring " << iterOther->first << std::endl;
      countMods += iterOther->second;
    }

    outputFile << "   # Modules          " << countMods << std::endl;
    if ( DebugMode )
    {
      outputFile << "     In/Out Correct Order ? ";
      for ( std::map< uint32_t, bool >::iterator iterAux = inOutPerRingPerDisk[iterMap->first].begin();
            iterAux != inOutPerRingPerDisk[iterMap->first].end();
            ++iterAux )
      {
        outputFile << iterAux->second;
      }
      outputFile << std::endl;
    }

    /// Average radii and separation
    for ( std::map< uint32_t, double >::iterator iterOther = innerModZPerRingPerDisk[iterMap->first].begin();
          iterOther != innerModZPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "  Inner sensor average z:  " <<
        iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " [cm] in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, double >::iterator iterOther = outerModZPerRingPerDisk[iterMap->first].begin();
          iterOther != outerModZPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "    Outer sensor average z:  " <<
        iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " [cm] in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, double >::iterator iterOther = separationPerRingPerDisk[iterMap->first].begin();
          iterOther != separationPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "      Average sensor separation:    " <<
        iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second * 10 / 2.0 <<
        " [mm] in Ring " << iterOther->first << std::endl;
    }

    /// Stack length and surface
    for ( std::map< uint32_t, double >::iterator iterOther = activeSurfacePerRingPerDisk[iterMap->first].begin();
          iterOther != activeSurfacePerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "  Total active surface:         " <<
      iterOther->second << // /(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " [cm^2] in Ring " << iterOther->first << std::endl;
    }

    /// Topology
    for ( std::map< uint32_t, uint32_t >::iterator iterOther = innerModChannelsPerRingPerDisk[iterMap->first].begin();
          iterOther != innerModChannelsPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "  Inner sensor channels:        " <<
      iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, uint32_t >::iterator iterOther = innerModRowsPerRingPerDisk[iterMap->first].begin();
          iterOther != innerModRowsPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "               rows:            " <<
      iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, uint32_t >::iterator iterOther = innerModColumnsPerRingPerDisk[iterMap->first].begin();
          iterOther != innerModColumnsPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "               columns:         " <<
      iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, float >::iterator iterOther = innerModXPitchPerRingPerDisk[iterMap->first].begin();
          iterOther != innerModXPitchPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "  Inner sensor average pitch X: " <<
      double(int(0.5+(10000*iterOther->second)))/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " [um] in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, float >::iterator iterOther = innerModYPitchPerRingPerDisk[iterMap->first].begin();
          iterOther != innerModYPitchPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "                             Y: " <<
      double(int(0.5+(10000*iterOther->second)))/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " [um] in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, uint32_t >::iterator iterOther = innerModXROCPerRingPerDisk[iterMap->first].begin();
          iterOther != innerModXROCPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "  Inner sensor ROCs X:          " <<
      iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, uint32_t >::iterator iterOther = innerModYROCPerRingPerDisk[iterMap->first].begin();
          iterOther != innerModYROCPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "                    Y:          " <<
      iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, uint32_t >::iterator iterOther = innerModTotROCPerRingPerDisk[iterMap->first].begin();
          iterOther != innerModTotROCPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "                    total:      " <<
      iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " in Ring " << iterOther->first << std::endl;
    }

    for ( std::map< uint32_t, uint32_t >::iterator iterOther = outerModChannelsPerRingPerDisk[iterMap->first].begin();
          iterOther != outerModChannelsPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "  Outer sensor channels:        " <<
      iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, uint32_t >::iterator iterOther = outerModRowsPerRingPerDisk[iterMap->first].begin();
          iterOther != outerModRowsPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "               rows:            " <<
      iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, uint32_t >::iterator iterOther = outerModColumnsPerRingPerDisk[iterMap->first].begin();
          iterOther != outerModColumnsPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "               columns:         " <<
      iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, float >::iterator iterOther = outerModXPitchPerRingPerDisk[iterMap->first].begin();
          iterOther != outerModXPitchPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "  Outer sensor average pitch X: " <<
      double(int(0.5+(10000*iterOther->second)))/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " [um] in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, float >::iterator iterOther = outerModYPitchPerRingPerDisk[iterMap->first].begin();
          iterOther != outerModYPitchPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "                             Y: " <<
      double(int(0.5+(10000*iterOther->second)))/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " [um] in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, uint32_t >::iterator iterOther = outerModXROCPerRingPerDisk[iterMap->first].begin();
          iterOther != outerModXROCPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "  Outer sensor ROCs X:          " <<
      iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, uint32_t >::iterator iterOther = outerModYROCPerRingPerDisk[iterMap->first].begin();
          iterOther != outerModYROCPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "                    Y:          " <<
      iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " in Ring " << iterOther->first << std::endl;
    }
    for ( std::map< uint32_t, uint32_t >::iterator iterOther = outerModTotROCPerRingPerDisk[iterMap->first].begin();
          iterOther != outerModTotROCPerRingPerDisk[iterMap->first].end();
          ++iterOther )
    {
      outputFile << "                    total:      " <<
      iterOther->second/(double)modsPerRingPerDisk[iterMap->first].find(iterOther->first)->second/2.0 <<
        " in Ring " << iterOther->first << std::endl;
    }
  }
} /// End of analyze()

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(AnalyzerPrintGeomInfo);

