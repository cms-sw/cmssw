/*! \brief   Checklist
 *  \details TTTracks
 *
 *  \author Nicola Pozzobon
 *  \author Anders Ryd
 *  \author Louise Skinnari
 *  \author David Sweigart
 *  \date   2013, Jul 22
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH1D.h>
#include <TH2D.h>

class AnalyzerL1Track : public edm::EDAnalyzer
{
  /// Public methods
  public:
    /// Constructor/destructor
    explicit AnalyzerL1Track(const edm::ParameterSet& iConfig);
    virtual ~AnalyzerL1Track();
    // Typical methods used on Loops over events
    virtual void beginJob();
    virtual void endJob();
    virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  /// Private methods and variables
  private:
    TH2D* hTrack_NStubs_Sector;
    TH2D* hTrack_NStubs_Wedge;
    TH2D* hTrack_Sector_Phi;
    TH2D* hTrack_Wedge_Eta;

    TH1D* hTrack_3Stubs_Pt;
    TH1D* hTrack_3Stubs_Phi;
    TH1D* hTrack_3Stubs_Eta;

    TH1D* hTrack_2Stubs_Pt;
    TH1D* hTrack_2Stubs_Phi;
    TH1D* hTrack_2Stubs_Eta;

    TH1D* hTPart_Track_3Stubs_Pt;
    TH1D* hTPart_Track_3Stubs_Phi_Pt5;
    TH1D* hTPart_Track_3Stubs_Eta_Pt5;

    TH1D* hTPart_Track_2Stubs_Pt;
    TH1D* hTPart_Track_2Stubs_Phi_Pt5;
    TH1D* hTPart_Track_2Stubs_Eta_Pt5;

    TH1D* hTPart_Cluster_Pt;
    TH1D* hTPart_Cluster_Phi_Pt5;
    TH1D* hTPart_Cluster_Eta_Pt5;

    TH1D* hTPart_Stub_Pt;
    TH1D* hTPart_Stub_Phi_Pt5;
    TH1D* hTPart_Stub_Eta_Pt5;

    TH1D* hTrack_3Stubs_N;
    TH2D* hTrack_3Stubs_Pt_TPart_Pt;
    TH2D* hTrack_3Stubs_PtRes_TPart_Eta;
    TH2D* hTrack_3Stubs_InvPt_TPart_InvPt;
    TH2D* hTrack_3Stubs_InvPtRes_TPart_Eta;
    TH2D* hTrack_3Stubs_Phi_TPart_Phi;
    TH2D* hTrack_3Stubs_PhiRes_TPart_Eta;
    TH2D* hTrack_3Stubs_Eta_TPart_Eta;
    TH2D* hTrack_3Stubs_EtaRes_TPart_Eta;
    TH2D* hTrack_3Stubs_VtxZ0_TPart_VtxZ0;
    TH2D* hTrack_3Stubs_VtxZ0Res_TPart_Eta;
    TH2D* hTrack_3Stubs_Chi2_NStubs;
    TH2D* hTrack_3Stubs_Chi2_TPart_Eta;
    TH2D* hTrack_3Stubs_Chi2Red_NStubs;
    TH2D* hTrack_3Stubs_Chi2Red_TPart_Eta;

    TH1D* hTPart_Eta_Normalization;
    TH1D* hTPart_Eta_NStubs;

    unsigned int maxPtBin;
    unsigned int maxEtaBin;
    std::vector< double > vLimitsPt;
    std::vector< std::string > vStringPt;
    std::vector< double > vLimitsEta;
    std::vector< std::string > vStringEta;

    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrack_3Stubs_Chi2_PtEta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrack_3Stubs_Chi2Red_PtEta;

    unsigned int maxPtBinRes;
    unsigned int maxEtaBinRes;
    double ptBinSize;
    double etaBinSize;

    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrack_3Stubs_PtRes_PtEta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrack_3Stubs_InvPtRes_PtEta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrack_3Stubs_RelPtRes_PtEta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrack_3Stubs_PhiRes_PtEta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrack_3Stubs_EtaRes_PtEta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrack_3Stubs_CotThetaRes_PtEta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrack_3Stubs_VtxZ0Res_PtEta;

    TH1D* hTrack_2Stubs_N;
    TH2D* hTrack_2Stubs_Pt_TPart_Pt;
    TH2D* hTrack_2Stubs_PtRes_TPart_Eta;
    TH2D* hTrack_2Stubs_InvPt_TPart_InvPt;
    TH2D* hTrack_2Stubs_InvPtRes_TPart_Eta;
    TH2D* hTrack_2Stubs_Phi_TPart_Phi;
    TH2D* hTrack_2Stubs_PhiRes_TPart_Eta;
    TH2D* hTrack_2Stubs_Eta_TPart_Eta;
    TH2D* hTrack_2Stubs_EtaRes_TPart_Eta;
    TH2D* hTrack_2Stubs_VtxZ0_TPart_VtxZ0;
    TH2D* hTrack_2Stubs_VtxZ0Res_TPart_Eta;
    TH2D* hTrack_2Stubs_Chi2_NStubs;
    TH2D* hTrack_2Stubs_Chi2_TPart_Eta;
    TH2D* hTrack_2Stubs_Chi2Red_NStubs;
    TH2D* hTrack_2Stubs_Chi2Red_TPart_Eta;

    edm::InputTag tagTTClusters;
    edm::InputTag tagTTClusterMCTruth;
    edm::InputTag tagTTStubs;
    edm::InputTag tagTTStubMCTruth;
    edm::InputTag tagTTTracks;
    edm::InputTag tagTTTrackMCTruth;
};

//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
AnalyzerL1Track::AnalyzerL1Track(edm::ParameterSet const& iConfig) 
{
  /// Insert here what you need to initialize
  vLimitsPt = iConfig.getParameter< std::vector< double > >("vLimitsPt");
  vLimitsEta = iConfig.getParameter< std::vector< double > >("vLimitsEta");
  vStringPt = iConfig.getParameter< std::vector< std::string > >("vStringPt");
  vStringEta = iConfig.getParameter< std::vector< std::string > >("vStringEta");

  if ( vLimitsPt.size() != vStringPt.size() )
  {
    exit(0);
  }
  else
  {
    maxPtBin = vLimitsPt.size();
  }

  if ( vLimitsEta.size() != vStringEta.size() )
  {
    exit(0);
  }
  else
  {
    maxEtaBin = vLimitsEta.size();
  }

  maxPtBinRes = iConfig.getParameter< unsigned int >("maxPtBinRes");
  maxEtaBinRes = iConfig.getParameter< unsigned int >("maxEtaBinRes");
  ptBinSize  = iConfig.getParameter< double >("ptBinSize");
  etaBinSize = iConfig.getParameter< double >("etaBinSize");

  if ( maxPtBinRes == 0 || maxEtaBinRes == 0 )
    exit(0);

  tagTTClusters = iConfig.getParameter< edm::InputTag >("TTClusters");
  tagTTClusterMCTruth = iConfig.getParameter< edm::InputTag >("TTClusterMCTruth");
  tagTTStubs = iConfig.getParameter< edm::InputTag >("TTStubs");
  tagTTStubMCTruth = iConfig.getParameter< edm::InputTag >("TTStubMCTruth");
  tagTTTracks = iConfig.getParameter< edm::InputTag >("TTTracks");
  tagTTTrackMCTruth = iConfig.getParameter< edm::InputTag >("TTTrackMCTruth");
}

/////////////
// DESTRUCTOR
AnalyzerL1Track::~AnalyzerL1Track()
{
  /// Insert here what you need to delete
  /// when you close the class instance
}  

//////////
// END JOB
void AnalyzerL1Track::endJob()
{
  /// Things to be done at the exit of the event Loop
  std::cerr << " AnalyzerL1Track::endJob" << std::endl;
  /// End of things to be done at the exit from the event Loop
}

////////////
// BEGIN JOB
void AnalyzerL1Track::beginJob()
{
  /// Initialize all slave variables
  /// mainly histogram ranges and resolution
  std::ostringstream histoName;
  std::ostringstream histoTitle;

  /// Things to be done before entering the event Loop
  std::cerr << " AnalyzerL1Track::beginJob" << std::endl;

  /// Book histograms etc
  edm::Service<TFileService> fs;

  /// Prepare for LogXY Plots
  int NumBins = 200;
  double MinPt = 0.0;
  double MaxPt = 100.0;

  double* BinVec = new double[NumBins+1];
  for ( int iBin = 0; iBin < NumBins + 1; iBin++ )
  {
    double temp = pow( 10, (- NumBins + iBin)/(MaxPt - MinPt)  );
    BinVec[ iBin ] = temp;
  }

  hTrack_NStubs_Sector      = fs->make<TH2D>( "hTrack_NStubs_Sector",      "TTTrack number of Stubs vs. #phi sector", 35, -0.5, 34.5, 20, -0.5, 19.5 );
  hTrack_NStubs_Wedge       = fs->make<TH2D>( "hTrack_NStubs_Wedge",       "TTTrack number of Stubs vs. #eta wedge",  35, -0.5, 34.5, 20, -0.5, 19.5 );
  hTrack_Sector_Phi         = fs->make<TH2D>( "hTrack_Sector_Phi",         "#phi sector vs. TTTrack #phi",            180, -M_PI, M_PI, 35, -0.5, 34.5 );
  hTrack_Wedge_Eta          = fs->make<TH2D>( "hTrack_Wedge_Eta",          "#eta wedge vs. TTTrack #eta",             180, -M_PI, M_PI, 35, -0.5, 34.5 );

  hTrack_NStubs_Sector->Sumw2();
  hTrack_NStubs_Wedge->Sumw2();
  hTrack_Sector_Phi->Sumw2();
  hTrack_Wedge_Eta->Sumw2();

  hTrack_3Stubs_Pt   = fs->make<TH1D>( "hTrack_3Stubs_Pt",  "TTTrack 3Stubs p_{T}", 100, 0, 50 );
  hTrack_3Stubs_Phi  = fs->make<TH1D>( "hTrack_3Stubs_Phi", "TTTrack 3Stubs #phi",  180, -M_PI, M_PI );
  hTrack_3Stubs_Eta  = fs->make<TH1D>( "hTrack_3Stubs_Eta", "TTTrack 3Stubs #eta",  180, -M_PI, M_PI );
  hTrack_3Stubs_Pt->Sumw2();
  hTrack_3Stubs_Phi->Sumw2();
  hTrack_3Stubs_Eta->Sumw2();

  hTrack_2Stubs_Pt   = fs->make<TH1D>( "hTrack_2Stubs_Pt",  "TTTrack 2Stubs p_{T}", 100, 0, 50 );
  hTrack_2Stubs_Phi  = fs->make<TH1D>( "hTrack_2Stubs_Phi", "TTTrack 2Stubs #phi",  180, -M_PI, M_PI );
  hTrack_2Stubs_Eta  = fs->make<TH1D>( "hTrack_2Stubs_Eta", "TTTrack 2Stubs #eta",  180, -M_PI, M_PI );
  hTrack_2Stubs_Pt->Sumw2();
  hTrack_2Stubs_Phi->Sumw2();
  hTrack_2Stubs_Eta->Sumw2();

  hTPart_Track_3Stubs_Pt       = fs->make<TH1D>( "hTPart_Track_3Stubs_Pt",      "TTTrack 3Stubs TPart p_{T}", 100, 0, 50 );
  hTPart_Track_3Stubs_Phi_Pt5  = fs->make<TH1D>( "hTPart_Track_3Stubs_Phi_Pt5", "TTTrack 3Stubs TPart #phi",  180, -M_PI, M_PI );
  hTPart_Track_3Stubs_Eta_Pt5  = fs->make<TH1D>( "hTPart_Track_3Stubs_Eta_Pt5", "TTTrack 3Stubs TPart  #eta", 180, -M_PI, M_PI );
  hTPart_Track_3Stubs_Pt->Sumw2();
  hTPart_Track_3Stubs_Phi_Pt5->Sumw2();
  hTPart_Track_3Stubs_Eta_Pt5->Sumw2();

  hTPart_Track_2Stubs_Pt       = fs->make<TH1D>( "hTPart_Track_2Stubs_Pt",      "TTTrack 2Stubs TPart p_{T}", 100, 0, 50 );
  hTPart_Track_2Stubs_Phi_Pt5  = fs->make<TH1D>( "hTPart_Track_2Stubs_Phi_Pt5", "TTTrack 2Stubs TPart #phi",  180, -M_PI, M_PI );
  hTPart_Track_2Stubs_Eta_Pt5  = fs->make<TH1D>( "hTPart_Track_2Stubs_Eta_Pt5", "TTTrack 2Stubs TPart  #eta", 180, -M_PI, M_PI );
  hTPart_Track_2Stubs_Pt->Sumw2();
  hTPart_Track_2Stubs_Phi_Pt5->Sumw2();
  hTPart_Track_2Stubs_Eta_Pt5->Sumw2();

  hTPart_Cluster_Pt           = fs->make<TH1D>( "hTPart_Cluster_Pt",      "Cluster TPart p_{T}", 100, 0, 50 );
  hTPart_Cluster_Phi_Pt5      = fs->make<TH1D>( "hTPart_Cluster_Phi_Pt5", "Cluster TPart #phi",  180, -M_PI, M_PI );
  hTPart_Cluster_Eta_Pt5      = fs->make<TH1D>( "hTPart_Cluster_Eta_Pt5", "Cluster TPart #eta",  180, -M_PI, M_PI );
  hTPart_Cluster_Pt->Sumw2();
  hTPart_Cluster_Phi_Pt5->Sumw2();
  hTPart_Cluster_Eta_Pt5->Sumw2();

  hTPart_Stub_Pt           = fs->make<TH1D>( "hTPart_Stub_Pt",      "Stub TPart p_{T}", 100, 0, 50 );
  hTPart_Stub_Phi_Pt5      = fs->make<TH1D>( "hTPart_Stub_Phi_Pt5", "Stub TPart #phi",  180, -M_PI, M_PI );
  hTPart_Stub_Eta_Pt5      = fs->make<TH1D>( "hTPart_Stub_Eta_Pt5", "Stub TPart #eta",  180, -M_PI, M_PI );
  hTPart_Stub_Pt->Sumw2();
  hTPart_Stub_Phi_Pt5->Sumw2();
  hTPart_Stub_Eta_Pt5->Sumw2();

  hTrack_3Stubs_N                  = fs->make<TH1D>( "hTrack_3Stubs_N",                 "Number of TTTrack 3Stubs",                          100, -0.5, 99.5 );
  hTrack_3Stubs_Pt_TPart_Pt        = fs->make<TH2D>( "hTrack_3Stubs_Pt_TPart_Pt",       "TTTrack 3Stubs p_{T} vs. TPart p_{T}",              100, 0, 50, 100, 0, 50 );
  hTrack_3Stubs_PtRes_TPart_Eta    = fs->make<TH2D>( "hTrack_3Stubs_PtRes_TPart_Eta",   "TTTrack 3Stubs p_{T} - TPart p_{T} vs. TPart #eta", 180, -M_PI, M_PI, 200, -4.0, 4.0 );
  hTrack_3Stubs_InvPt_TPart_InvPt  = fs->make<TH2D>( "hTrack_3Stubs_InvPt_TPart_InvPt", "TTTrack 3Stubs p_{T}^{-1} vs. TPart p_{T}^{-1}",    200, 0, 0.8, 200, 0, 0.8 );
  hTrack_3Stubs_InvPt_TPart_InvPt->GetXaxis()->Set( NumBins, BinVec );
  hTrack_3Stubs_InvPt_TPart_InvPt->GetYaxis()->Set( NumBins, BinVec );
  hTrack_3Stubs_InvPtRes_TPart_Eta = fs->make<TH2D>( "hTrack_3Stubs_InvPtRes_TPart_Eta", "TTTrack 3Stubs p_{T}^{-1} - TPart p_{T}^{-1}  vs. TPart #eta", 180, -M_PI, M_PI, 100, -1.0, 1.0 );
  hTrack_3Stubs_Phi_TPart_Phi      = fs->make<TH2D>( "hTrack_3Stubs_Phi_TPart_Phi",      "TTTrack 3Stubs #phi vs. TPart #phi",                           180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hTrack_3Stubs_PhiRes_TPart_Eta   = fs->make<TH2D>( "hTrack_3Stubs_PhiRes_TPart_Eta",   "TTTrack 3Stubs #phi - TPart #phi vs. TPart #eta",              180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hTrack_3Stubs_Eta_TPart_Eta      = fs->make<TH2D>( "hTrack_3Stubs_Eta_TPart_Eta",      "TTTrack 3Stubs #eta vs. TPart #eta",                           180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hTrack_3Stubs_EtaRes_TPart_Eta   = fs->make<TH2D>( "hTrack_3Stubs_EtaRes_TPart_Eta",   "TTTrack 3Stubs #eta - TPart #eta vs. TPart #eta",              180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hTrack_3Stubs_VtxZ0_TPart_VtxZ0  = fs->make<TH2D>( "hTrack_3Stubs_VtxZ0_TPart_VtxZ0",  "TTTrack 3Stubs z_{vtx} vs. TPart z_{vtx}",                     180, -30, 30, 180, -30, 30 );
  hTrack_3Stubs_VtxZ0Res_TPart_Eta = fs->make<TH2D>( "hTrack_3Stubs_VtxZ0Res_TPart_Eta", "TTTrack 3Stubs z_{vtx} - TPart z_{vtx} vs. TPart #eta",        180, -M_PI, M_PI, 100, -5, 5 );
  hTrack_3Stubs_Chi2_NStubs        = fs->make<TH2D>( "hTrack_3Stubs_Chi2_NStubs",        "TTTrack 3Stubs #chi^{2} vs. number of Stubs",                  20, -0.5, 19.5, 200, 0, 50 );
  hTrack_3Stubs_Chi2_TPart_Eta     = fs->make<TH2D>( "hTrack_3Stubs_Chi2_TPart_Eta",     "TTTrack 3Stubs #chi^{2} vs. TPart #eta",                       180, -M_PI, M_PI, 200, 0, 50 );
  hTrack_3Stubs_Chi2Red_NStubs     = fs->make<TH2D>( "hTrack_3Stubs_Chi2Red_NStubs",     "TTTrack 3Stubs #chi^{2}/dof vs. number of Stubs",              20, -0.5, 19.5, 200, 0, 10 );
  hTrack_3Stubs_Chi2Red_TPart_Eta  = fs->make<TH2D>( "hTrack_3Stubs_Chi2Red_TPart_Eta",  "TTTrack 3Stubs #chi^{2}/dof vs. TPart #eta",                   180, -M_PI, M_PI, 200, 0, 10 );
  hTrack_3Stubs_N->Sumw2();
  hTrack_3Stubs_Pt_TPart_Pt->Sumw2();
  hTrack_3Stubs_PtRes_TPart_Eta->Sumw2();
  hTrack_3Stubs_InvPt_TPart_InvPt->Sumw2();
  hTrack_3Stubs_InvPtRes_TPart_Eta->Sumw2();
  hTrack_3Stubs_Phi_TPart_Phi->Sumw2();
  hTrack_3Stubs_PhiRes_TPart_Eta->Sumw2();
  hTrack_3Stubs_Eta_TPart_Eta->Sumw2();
  hTrack_3Stubs_EtaRes_TPart_Eta->Sumw2();
  hTrack_3Stubs_VtxZ0_TPart_VtxZ0->Sumw2();
  hTrack_3Stubs_VtxZ0Res_TPart_Eta->Sumw2();
  hTrack_3Stubs_Chi2_NStubs->Sumw2();
  hTrack_3Stubs_Chi2_TPart_Eta->Sumw2();
  hTrack_3Stubs_Chi2Red_NStubs->Sumw2();
  hTrack_3Stubs_Chi2Red_TPart_Eta->Sumw2();

  hTPart_Eta_Normalization = fs->make<TH1D>("hTPart_Eta_Normalization", "TParticles vs. TPart #eta", 90, 0, M_PI );
  hTPart_Eta_NStubs        = fs->make<TH1D>("hTPart_Eta_NStubs"       , "N Stubs vs. TPart #eta"   , 90, 0, M_PI );
  hTPart_Eta_Normalization->Sumw2();
  hTPart_Eta_NStubs->Sumw2();

  /// Prepare for 1D resolution plots

  for ( unsigned int iPt = 0; iPt < maxPtBin; iPt++ )
  {
    for ( unsigned int iEta = 0; iEta < maxEtaBin; iEta++ )
    {
      std::pair< unsigned int, unsigned int > mapKey = std::make_pair( iPt, iEta );

      histoName.str("");  histoName << "hTrack_3Stubs_Chi2_Pt" << iPt << "_Eta" << iEta;
      histoTitle.str(""); histoTitle << "Track 3Stubs #chi^{2}, " << vStringPt.at(iPt).c_str() << " " << vStringEta.at(iEta).c_str();
      mapTrack_3Stubs_Chi2_PtEta[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(), 50, 0, 50 );
      mapTrack_3Stubs_Chi2_PtEta[ mapKey ]->Sumw2();

      histoName.str("");  histoName << "hTrack_3Stubs_Chi2Red_Pt" << iPt << "_Eta" << iEta;
      histoTitle.str(""); histoTitle << "Track 3Stubs #chi^{2}/dof, " << vStringPt.at(iPt).c_str() << " " << vStringEta.at(iEta).c_str();
      mapTrack_3Stubs_Chi2Red_PtEta[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(), 50, 0, 10 );
      mapTrack_3Stubs_Chi2Red_PtEta[ mapKey ]->Sumw2();
    }
  }

  double minPt = 0;
  double minEta = 0;
  for ( unsigned int iPt = 0; iPt < maxPtBinRes; iPt++ )
  {
    minPt = ptBinSize * iPt;
    for ( unsigned int iEta = 0; iEta < maxEtaBinRes; iEta++ )
    {
      std::pair< unsigned int, unsigned int > mapKey = std::make_pair( iPt, iEta );

      minEta = etaBinSize * iEta;

      histoName.str("");  histoName << "hTrack_3Stubs_PtRes_Pt" << iPt << "_Eta" << iEta;
      histoTitle.str(""); histoTitle << "Track p_{T} - TPart p_{T}, p_{T} in [" << minPt << ", " << (minPt + ptBinSize) <<
                                                               "), |#eta| in [" << minEta << ", " << (minEta + etaBinSize) << ")";
      mapTrack_3Stubs_PtRes_PtEta[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(), 200, -4, 4 );
      mapTrack_3Stubs_PtRes_PtEta[ mapKey ]->Sumw2();

      histoName.str("");  histoName << "hTrack_3Stubs_InvPtRes_Pt" << iPt << "_Eta" << iEta;
      histoTitle.str(""); histoTitle << "Track p_{T}^{-1} - TPart p_{T}^{-1}, p_{T} in [" << minPt << ", " << (minPt + ptBinSize) <<
                                                                         "), |#eta| in [" << minEta << ", " << (minEta + etaBinSize) << ")";
      mapTrack_3Stubs_InvPtRes_PtEta[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(), 200, -1, 1 );
      mapTrack_3Stubs_InvPtRes_PtEta[ mapKey ]->Sumw2();

      histoName.str("");  histoName << "hTrack_3Stubs_RelPtRes_Pt" << iPt << "_Eta" << iEta;
      histoTitle.str(""); histoTitle << "Track p_{T}/TPart p_{T} - 1, p_{T} in [" << minPt << ", " << (minPt + ptBinSize) <<
                                                                 "), |#eta| in [" << minEta << ", " << (minEta + etaBinSize) << ")";
      mapTrack_3Stubs_RelPtRes_PtEta[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(), 200, -0.5, 0.5 );
      mapTrack_3Stubs_RelPtRes_PtEta[ mapKey ]->Sumw2();

      histoName.str("");  histoName << "hTrack_3Stubs_PhiRes_Pt" << iPt << "_Eta" << iEta;
      histoTitle.str(""); histoTitle << "Track #phi - TPart #phi, p_{T} in [" << minPt << ", " << (minPt + ptBinSize) <<
                                                             "), |#eta| in [" << minEta << ", " << (minEta + etaBinSize) << ")";
      mapTrack_3Stubs_PhiRes_PtEta[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(), 200, -0.5, 0.5 );
      mapTrack_3Stubs_PhiRes_PtEta[ mapKey ]->Sumw2();

      histoName.str("");  histoName << "hTrack_3Stubs_EtaRes_Pt" << iPt << "_Eta" << iEta;
      histoTitle.str(""); histoTitle << "Track #eta - TPart #eta, p_{T} in [" << minPt << ", " << (minPt + ptBinSize) <<
                                                             "), |#eta| in [" << minEta << ", " << (minEta + etaBinSize) << ")";
      mapTrack_3Stubs_EtaRes_PtEta[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(), 200, -0.5, 0.5 );
      mapTrack_3Stubs_EtaRes_PtEta[ mapKey ]->Sumw2();

      histoName.str("");  histoName << "hTrack_3Stubs_CotThetaRes_Pt" << iPt << "_Eta" << iEta;
      histoTitle.str(""); histoTitle << "Track cot(#theta) - TPart cot(#theta), p_{T} in [" << minPt << ", " << (minPt + ptBinSize) <<
                                                                           "), |#eta| in [" << minEta << ", " << (minEta + etaBinSize) << ")";
      mapTrack_3Stubs_CotThetaRes_PtEta[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(), 200, -0.5, 0.5 );
      mapTrack_3Stubs_CotThetaRes_PtEta[ mapKey ]->Sumw2();


      histoName.str("");  histoName << "hTrack_3Stubs_VtxZ0Res_Pt" << iPt << "_Eta" << iEta;
      histoTitle.str(""); histoTitle << "Track z_{vtx} - TPart z_{vtx}, p_{T} in [" << minPt << ", " << (minPt + ptBinSize) <<
                                                                   "), |#eta| in [" << minEta << ", " << (minEta + etaBinSize) << ")";
      mapTrack_3Stubs_VtxZ0Res_PtEta[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(), 200, -0.5, 0.5 );
      mapTrack_3Stubs_VtxZ0Res_PtEta[ mapKey ]->Sumw2();
    }
  }

  hTrack_2Stubs_N                  = fs->make<TH1D>( "hTrack_2Stubs_N",                  "Number of TTTrack 2Stubs",                          100, -0.5, 99.5 );
  hTrack_2Stubs_Pt_TPart_Pt        = fs->make<TH2D>( "hTrack_2Stubs_Pt_TPart_Pt",        "TTTrack 2Stubs p_{T} vs. TPart p_{T}",              100, 0, 50, 100, 0, 50 );
  hTrack_2Stubs_PtRes_TPart_Eta    = fs->make<TH2D>( "hTrack_2Stubs_PtRes_TPart_Eta",    "TTTrack 2Stubs p_{T} - TPart p_{T} vs. TPart #eta", 180, -M_PI, M_PI, 200, -4.0, 4.0 );
  hTrack_2Stubs_InvPt_TPart_InvPt  = fs->make<TH2D>( "hTrack_2Stubs_InvPt_TPart_InvPt",  "TTTrack 2Stubs p_{T}^{-1} vs. TPart p_{T}^{-1}",    200, 0, 0.8, 200, 0, 0.8 );
  hTrack_2Stubs_InvPt_TPart_InvPt->GetXaxis()->Set( NumBins, BinVec );
  hTrack_2Stubs_InvPt_TPart_InvPt->GetYaxis()->Set( NumBins, BinVec );
  hTrack_2Stubs_InvPtRes_TPart_Eta = fs->make<TH2D>( "hTrack_2Stubs_InvPtRes_TPart_Eta", "TTTrack 2Stubs p_{T}^{-1} - TPart p_{T}^{-1}  vs. TPart #eta", 180, -M_PI, M_PI, 100, -1.0, 1.0 );
  hTrack_2Stubs_Phi_TPart_Phi      = fs->make<TH2D>( "hTrack_2Stubs_Phi_TPart_Phi",      "TTTrack 2Stubs #phi vs. TPart #phi",                           180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hTrack_2Stubs_PhiRes_TPart_Eta   = fs->make<TH2D>( "hTrack_2Stubs_PhiRes_TPart_Eta",   "TTTrack 2Stubs #phi - TPart #phi vs. TPart #eta",              180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hTrack_2Stubs_Eta_TPart_Eta      = fs->make<TH2D>( "hTrack_2Stubs_Eta_TPart_Eta",      "TTTrack 2Stubs #eta vs. TPart #eta",                           180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hTrack_2Stubs_EtaRes_TPart_Eta   = fs->make<TH2D>( "hTrack_2Stubs_EtaRes_TPart_Eta",   "TTTrack 2Stubs #eta - TPart #eta vs. TPart #eta",              180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hTrack_2Stubs_VtxZ0_TPart_VtxZ0  = fs->make<TH2D>( "hTrack_2Stubs_VtxZ0_TPart_VtxZ0",  "TTTrack 2Stubs z_{vtx} vs. TPart z_{vtx}",                     180, -30, 30, 180, -30, 30 );
  hTrack_2Stubs_VtxZ0Res_TPart_Eta = fs->make<TH2D>( "hTrack_2Stubs_VtxZ0Res_TPart_Eta", "TTTrack 2Stubs z_{vtx} - TPart z_{vtx} vs. TPart #eta",        180, -M_PI, M_PI, 100, -5, 5 );
  hTrack_2Stubs_Chi2_NStubs        = fs->make<TH2D>( "hTrack_2Stubs_Chi2_NStubs",        "TTTrack 2Stubs #chi^{2} vs. number of Stubs",                  20, -0.5, 19.5, 200, 0, 50 );
  hTrack_2Stubs_Chi2_TPart_Eta     = fs->make<TH2D>( "hTrack_2Stubs_Chi2_TPart_Eta",     "TTTrack 2Stubs #chi^{2} vs. TPart #eta",                       180, -M_PI, M_PI, 200, 0, 50 );
  hTrack_2Stubs_Chi2Red_NStubs     = fs->make<TH2D>( "hTrack_2Stubs_Chi2Red_NStubs",     "TTTrack 2Stubs #chi^{2}/dof vs. number of Stubs",              20, -0.5, 19.5, 200, 0, 10 );
  hTrack_2Stubs_Chi2Red_TPart_Eta  = fs->make<TH2D>( "hTrack_2Stubs_Chi2Red_TPart_Eta",  "TTTrack 2Stubs #chi^{2}/dof vs. TPart #eta",                   180, -M_PI, M_PI, 200, 0, 10 );

  hTrack_2Stubs_N->Sumw2();
  hTrack_2Stubs_Pt_TPart_Pt->Sumw2();
  hTrack_2Stubs_PtRes_TPart_Eta->Sumw2();
  hTrack_2Stubs_InvPt_TPart_InvPt->Sumw2();
  hTrack_2Stubs_InvPtRes_TPart_Eta->Sumw2();
  hTrack_2Stubs_Phi_TPart_Phi->Sumw2();
  hTrack_2Stubs_PhiRes_TPart_Eta->Sumw2();
  hTrack_2Stubs_Eta_TPart_Eta->Sumw2();
  hTrack_2Stubs_EtaRes_TPart_Eta->Sumw2();
  hTrack_2Stubs_VtxZ0_TPart_VtxZ0->Sumw2();
  hTrack_2Stubs_VtxZ0Res_TPart_Eta->Sumw2();
  hTrack_2Stubs_Chi2_NStubs->Sumw2();
  hTrack_2Stubs_Chi2_TPart_Eta->Sumw2();
  hTrack_2Stubs_Chi2Red_NStubs->Sumw2();
  hTrack_2Stubs_Chi2Red_TPart_Eta->Sumw2();

  /// End of things to be done before entering the event Loop
}

//////////
// ANALYZE
void AnalyzerL1Track::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  /// Track Trigger
  //edm::Handle< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > > PixelDigiTTClusterHandle;
  //edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > >    PixelDigiTTStubHandle;
  edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > >            PixelDigiTTTrackHandle;
  //iEvent.getByLabel( tagTTClusters, PixelDigiTTClusterHandle );
  //iEvent.getByLabel( tagTTStubs, PixelDigiTTStubHandle );
  iEvent.getByLabel( tagTTTracks, PixelDigiTTTrackHandle );

  /// Track Trigger MC Truth
  edm::Handle< TTClusterAssociationMap< Ref_PixelDigi_ > > MCTruthTTClusterHandle;
  edm::Handle< TTStubAssociationMap< Ref_PixelDigi_ > >    MCTruthTTStubHandle;
  edm::Handle< TTTrackAssociationMap< Ref_PixelDigi_ > >   MCTruthTTTrackHandle;
  iEvent.getByLabel( tagTTClusterMCTruth, MCTruthTTClusterHandle );
  iEvent.getByLabel( tagTTStubMCTruth, MCTruthTTStubHandle );
  iEvent.getByLabel( tagTTTrackMCTruth, MCTruthTTTrackHandle );

  /// TrackingParticles
  edm::Handle< std::vector< TrackingParticle > > TrackingParticleHandle;
  iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingParticleHandle );
  edm::Handle< std::vector< TrackingVertex > > TrackingVertexHandle;
  iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingVertexHandle );

  /// Loop over TrackingParticles
  if ( TrackingParticleHandle->size() > 0 )
  {
    unsigned int tpCnt = 0;
    std::vector< TrackingParticle >::const_iterator iterTP;
    for ( iterTP = TrackingParticleHandle->begin();
          iterTP != TrackingParticleHandle->end();
          ++iterTP )
    {
      /// Make the pointer
      edm::Ptr< TrackingParticle > tempTPPtr( TrackingParticleHandle, tpCnt++ );

      /// Skip non-primary track
      if ( tempTPPtr->vertex().rho() >= 2.0 )
         continue;

      /// Check if this TP produced any clusters
      if ( MCTruthTTClusterHandle->findTTClusterRefs( tempTPPtr ).size() > 0 )
      {
        hTPart_Cluster_Pt->Fill( tempTPPtr->p4().pt() );
        if ( tempTPPtr->p4().pt() > 5.0 )
        {
          hTPart_Cluster_Phi_Pt5->Fill( tempTPPtr->momentum().phi() );
          hTPart_Cluster_Eta_Pt5->Fill( tempTPPtr->momentum().eta() );
        }

        /// Check if the TP produced any stubs
        if ( MCTruthTTStubHandle->findTTStubRefs( tempTPPtr ).size() > 0 )
        {
          hTPart_Stub_Pt->Fill( tempTPPtr->p4().pt() );
          if ( tempTPPtr->p4().pt() > 5.0 )
          {
            hTPart_Stub_Phi_Pt5->Fill( tempTPPtr->momentum().phi() );
            hTPart_Stub_Eta_Pt5->Fill( tempTPPtr->momentum().eta() );
          }

          ///Check if the TP produced any Tracks
          if ( MCTruthTTTrackHandle->findTTTrackPtrs( tempTPPtr ).size() > 0 )
          {
            /// Distinguish between 2-stubs-only and 3-stubs-minimum
            bool found2stubs = false;
            bool found3stubs = false;
            std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > > theseTracks =  MCTruthTTTrackHandle->findTTTrackPtrs( tempTPPtr );
            for ( unsigned int it = 0; it < theseTracks.size(); it++ )
            {
              if ( found2stubs && found3stubs )
              {
                it = theseTracks.size();
                continue;
              }

              if ( theseTracks.at(it)->getStubRefs().size() == 2 )
              {
                found2stubs = true;
              }
              else if ( theseTracks.at(it)->getStubRefs().size() > 2 )
              {
                /// Additional cross check
                bool hasBL1 = theseTracks.at(it)->hasStubInBarrel(1) ;
                if ( hasBL1 )
                {
                }
                  found3stubs = true;
                //} /// Additional cross check
              }
            }

            if ( found2stubs )
            {
              hTPart_Track_2Stubs_Pt->Fill( tempTPPtr->p4().pt() );
              if ( tempTPPtr->p4().pt() > 5.0 )
              {
                hTPart_Track_2Stubs_Phi_Pt5->Fill( tempTPPtr->momentum().phi() );
                hTPart_Track_2Stubs_Eta_Pt5->Fill( tempTPPtr->momentum().eta() );
              }
            }

            if ( found3stubs )
            {
              hTPart_Track_3Stubs_Pt->Fill( tempTPPtr->p4().pt() );
              if ( tempTPPtr->p4().pt() > 5.0 )
              {
                hTPart_Track_3Stubs_Phi_Pt5->Fill( tempTPPtr->momentum().phi() );
                hTPart_Track_3Stubs_Eta_Pt5->Fill( tempTPPtr->momentum().eta() );
              }
            }
          }
        }
      }
    } /// End of loop over TrackingParticles
  }

  unsigned int num3Stubs = 0;
  unsigned int num2Stubs = 0;

  /// Go on only if there are TTTracks from PixelDigis
  if ( PixelDigiTTTrackHandle->size() > 0 )
  {
    /// Loop over TTTracks
    unsigned int tkCnt = 0;
    std::vector< TTTrack< Ref_PixelDigi_ > >::const_iterator iterTTTrack;
    for ( iterTTTrack = PixelDigiTTTrackHandle->begin();
          iterTTTrack != PixelDigiTTTrackHandle->end();
          ++iterTTTrack )
    {
      /// Make the pointer
      edm::Ptr< TTTrack< Ref_PixelDigi_ > > tempTrackPtr( PixelDigiTTTrackHandle, tkCnt++ );

      /// Get everything is relevant
      unsigned int nStubs     = tempTrackPtr->getStubRefs().size();
      unsigned int seedSector = tempTrackPtr->getSector();
      unsigned int seedWedge  = tempTrackPtr->getWedge();

      hTrack_NStubs_Sector->Fill( seedSector, nStubs );
      hTrack_NStubs_Wedge->Fill( seedWedge, nStubs );

      //double trackRInv  = tempTrackPtr->getRInv();
      double trackPt    = tempTrackPtr->getMomentum().perp();
      double trackPhi   = tempTrackPtr->getMomentum().phi();
      double trackEta   = tempTrackPtr->getMomentum().eta();
      double trackTheta = tempTrackPtr->getMomentum().theta();
      double trackVtxZ0 = tempTrackPtr->getPOCA().z();
      double trackChi2  = tempTrackPtr->getChi2();
      double trackChi2R = tempTrackPtr->getChi2Red();

      hTrack_Sector_Phi->Fill( trackPhi, seedSector );
      hTrack_Wedge_Eta->Fill( trackEta, seedWedge );

      bool genuineTrack = MCTruthTTTrackHandle->isGenuine( tempTrackPtr );

      if ( !genuineTrack ) continue;

      edm::Ptr< TrackingParticle > tpPtr = MCTruthTTTrackHandle->findTrackingParticlePtr( tempTrackPtr );

      /// Get the corresponding vertex and reject the track
      /// if its vertex is outside the beampipe
      if ( tpPtr->vertex().rho() >= 2 )
        continue;

      double tpPt = tpPtr->p4().pt();
      double tpEta = tpPtr->momentum().eta();
      double tpTheta = tpPtr->momentum().theta();
      double tpPhi = tpPtr->momentum().phi();
      double tpVtxZ0 = tpPtr->vertex().z();

      if ( nStubs > 2 )
      {

        hTPart_Eta_Normalization->Fill( tpEta );
        hTPart_Eta_NStubs->Fill( tpEta, nStubs );

        hTrack_3Stubs_Pt->Fill( trackPt );
        hTrack_3Stubs_Eta->Fill( trackEta );
        hTrack_3Stubs_Phi->Fill( trackPhi );

        num3Stubs++;
        hTrack_3Stubs_Pt_TPart_Pt->Fill( tpPt, trackPt );
        hTrack_3Stubs_PtRes_TPart_Eta->Fill( tpEta, trackPt - tpPt );
        hTrack_3Stubs_InvPt_TPart_InvPt->Fill( 1./tpPt, 1./trackPt );
        hTrack_3Stubs_InvPtRes_TPart_Eta->Fill( tpEta, 1./trackPt - 1./tpPt );
        hTrack_3Stubs_Phi_TPart_Phi->Fill( tpPhi, trackPhi );
        hTrack_3Stubs_PhiRes_TPart_Eta->Fill( tpEta, trackPhi - tpPhi );
        hTrack_3Stubs_Eta_TPart_Eta->Fill( tpEta, trackEta );
        hTrack_3Stubs_EtaRes_TPart_Eta->Fill( tpEta, trackEta - tpEta );
        hTrack_3Stubs_VtxZ0_TPart_VtxZ0->Fill( tpVtxZ0, trackVtxZ0 );
        hTrack_3Stubs_VtxZ0Res_TPart_Eta->Fill( tpEta, trackVtxZ0 - tpVtxZ0 );
        hTrack_3Stubs_Chi2_NStubs->Fill( nStubs, trackChi2 );
        hTrack_3Stubs_Chi2_TPart_Eta->Fill( tpEta, trackChi2 );
        hTrack_3Stubs_Chi2Red_NStubs->Fill( nStubs, trackChi2R );
        hTrack_3Stubs_Chi2Red_TPart_Eta->Fill( tpEta, trackChi2R );

        /// Additional cross check
        bool hasBL1 = tempTrackPtr->hasStubInBarrel(1);
        if ( hasBL1 )
        {
        }

        /// Find the Pt/Eta bin for the 1D performance plots
        unsigned int binPt = maxPtBin;
        unsigned int binEta = maxEtaBin;
        for ( unsigned int iPt = maxPtBin; iPt > 0; )
        {
          iPt--;
          if ( tpPt < vLimitsPt.at(iPt) )
            binPt = iPt;
        }
        for ( unsigned int iEta = maxEtaBin; iEta > 0; )
        {
          iEta--;
          if ( fabs(tpEta) < vLimitsEta.at(iEta) )
            binEta = iEta;
        }

        std::pair< unsigned int, unsigned int > mapKey = std::make_pair( binPt, binEta );

        mapTrack_3Stubs_Chi2_PtEta[ mapKey ]->Fill( trackChi2 );
        mapTrack_3Stubs_Chi2Red_PtEta[ mapKey ]->Fill( trackChi2R );

        /// Now, the different Pt/Eta mapping
        binPt = 0;
        binEta = 0;
        for ( unsigned int iPt = 0; iPt < maxPtBinRes; iPt++ )
        {
          if ( tpPt >= iPt*ptBinSize )
            binPt = iPt;
        }
        for ( unsigned int iEta = 0; iEta < maxEtaBinRes; iEta++ )
        {
          if ( fabs(tpEta) >= iEta*etaBinSize )
            binEta = iEta;
        }

        mapKey = std::make_pair( binPt, binEta );

        mapTrack_3Stubs_PtRes_PtEta[ mapKey ]->Fill( trackPt - tpPt );
        mapTrack_3Stubs_InvPtRes_PtEta[ mapKey ]->Fill( 1./trackPt - 1./tpPt );
        mapTrack_3Stubs_RelPtRes_PtEta[ mapKey ]->Fill( trackPt / tpPt - 1 );
        mapTrack_3Stubs_PhiRes_PtEta[ mapKey ]->Fill( trackPhi - tpPhi );
        mapTrack_3Stubs_EtaRes_PtEta[ mapKey ]->Fill( trackEta - tpEta );
        mapTrack_3Stubs_CotThetaRes_PtEta[ mapKey ]->Fill( 1./tan(trackTheta) - 1./tan(tpTheta) );
        mapTrack_3Stubs_VtxZ0Res_PtEta[ mapKey ]->Fill( trackVtxZ0 - tpVtxZ0 );

        //} /// Additional cross check
      }
      else
      {
        hTrack_2Stubs_Pt->Fill( trackPt );
        hTrack_2Stubs_Eta->Fill( trackEta );
        hTrack_2Stubs_Phi->Fill( trackPhi );

        num2Stubs++;
        hTrack_2Stubs_Pt_TPart_Pt->Fill( tpPt, trackPt );
        hTrack_2Stubs_PtRes_TPart_Eta->Fill( tpEta, trackPt - tpPt );
        hTrack_2Stubs_InvPt_TPart_InvPt->Fill( 1./tpPt, 1./trackPt );
        hTrack_2Stubs_InvPtRes_TPart_Eta->Fill( tpEta, 1./trackPt - 1./tpPt );
        hTrack_2Stubs_Phi_TPart_Phi->Fill( tpPhi, trackPhi );
        hTrack_2Stubs_PhiRes_TPart_Eta->Fill( tpEta, trackPhi - tpPhi );
        hTrack_2Stubs_Eta_TPart_Eta->Fill( tpEta, trackEta );
        hTrack_2Stubs_EtaRes_TPart_Eta->Fill( tpEta, trackEta - tpEta );
        hTrack_2Stubs_VtxZ0_TPart_VtxZ0->Fill( tpVtxZ0, trackVtxZ0 );
        hTrack_2Stubs_VtxZ0Res_TPart_Eta->Fill( tpEta, trackVtxZ0 - tpVtxZ0 );
        hTrack_2Stubs_Chi2_NStubs->Fill( nStubs, trackChi2 );
        hTrack_2Stubs_Chi2_TPart_Eta->Fill( tpEta, trackChi2 );
        hTrack_2Stubs_Chi2Red_NStubs->Fill( nStubs, trackChi2R );
        hTrack_2Stubs_Chi2Red_TPart_Eta->Fill( tpEta, trackChi2R );      
      }
    } /// End of loop over TTTracks
  }

  hTrack_2Stubs_N->Fill( num2Stubs );
  hTrack_3Stubs_N->Fill( num3Stubs );

} /// End of analyze()

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(AnalyzerL1Track);

