//#define DEBUG

/*! \brief   Checklist
 *  \details DT to TK Matches for the HL-LHC
 *
 *  \author Nicola Pozzobon
 *  \author Pierluigi Zotto
 *  \date   2014, Apr 1
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
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "L1Trigger/DTUtilities/interface/DTTrigGeom.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "SimDataFormats/DigiSimLinks/interface/DTDigiSimLinkCollection.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTBtiTrigger.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTTSPhiTrigger.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTTSThetaTrigger.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTMatch.h"

#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include <TH1D.h>
#include <TH2D.h>

/// Class definition
class AnalyzerDTMatches : public edm::EDAnalyzer
{
  /// Public methods
  public :

    /// Constructor/destructor
    explicit AnalyzerDTMatches( const edm::ParameterSet& iConfig );
    virtual ~AnalyzerDTMatches();

    // Typical methods used on Loops over events
    virtual void beginJob();
    virtual void endJob();
    virtual void analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup );

  /// Private methods and variables
  private :
    bool useBothMuonCharges;
    bool selectPositiveMuons;

    std::string theMethods[16] = {
      std::string("Mu_2_1"),
      std::string("Mu_3_1"), std::string("Mu_3_2"),
      std::string("Mu_4_1"), std::string("Mu_4_2"), std::string("Mu_4_3"),
      std::string("Mu_5_1"), std::string("Mu_5_2"), std::string("Mu_5_3"), std::string("Mu_5_4"),
      std::string("Mu_6_1"), std::string("Mu_6_2"), std::string("Mu_6_3"), std::string("Mu_6_4"), std::string("Mu_6_5"),
      std::string("TTTrack") };

    /// Histograms
    /// identified by pair< wheel, MB >
    /// each propagation goes into a vector< layer > where L0 gives the
    /// projection to the vertex
    std::map< std::pair< int, int >, TH1F* > mapWS_hPhiB_C;
    std::map< std::pair< int, int >, TH1F* > mapWS_hPhiB_H;
    std::map< std::pair< int, int >, TH1F* > mapWS_hPhiB_L;

    std::map< std::pair< int, int >, TH1F* > mapWS_hPhiG_C;
    std::map< std::pair< int, int >, TH1F* > mapWS_hPhiG_H;
    std::map< std::pair< int, int >, TH1F* > mapWS_hPhiG_L;

    std::map< std::pair< int, int >, TH1F* > mapWS_hPhi_C;
    std::map< std::pair< int, int >, TH1F* > mapWS_hPhi_H;
    std::map< std::pair< int, int >, TH1F* > mapWS_hPhi_L;

    std::map< std::pair< int, int >, TH1F* > mapWS_hTheta_C;
    std::map< std::pair< int, int >, TH1F* > mapWS_hTheta_H;
    std::map< std::pair< int, int >, TH1F* > mapWS_hTheta_L;

    std::map< std::pair< int, int >, TH1F* > mapWS_hTPPhi_C;
    std::map< std::pair< int, int >, TH1F* > mapWS_hTPPhi_H;
    std::map< std::pair< int, int >, TH1F* > mapWS_hTPPhi_L;

    std::map< std::pair< int, int >, TH1F* > mapWS_hTPTheta_C;
    std::map< std::pair< int, int >, TH1F* > mapWS_hTPTheta_H;
    std::map< std::pair< int, int >, TH1F* > mapWS_hTPTheta_L;

    std::map< std::pair< int, int >, TH1F* > mapWS_hDeltaPhi_TP_C;
    std::map< std::pair< int, int >, TH1F* > mapWS_hDeltaPhi_TP_H;
    std::map< std::pair< int, int >, TH1F* > mapWS_hDeltaPhi_TP_L;

    std::map< std::pair< int, int >, TH1F* > mapWS_hDeltaTheta_TP_C;
    std::map< std::pair< int, int >, TH1F* > mapWS_hDeltaTheta_TP_H;
    std::map< std::pair< int, int >, TH1F* > mapWS_hDeltaTheta_TP_L;

    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaPhi_TK_C;
    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaPhi_TK_H;
    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaPhi_TK_L;

    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaTheta_TK_C;
    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaTheta_TK_H;
    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaTheta_TK_L;

    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaPhiResid_TK_C;
    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaPhiResid_TK_H;
    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaPhiResid_TK_L;

    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaThetaResid_TK_C;
    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaThetaResid_TK_H;
    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaThetaResid_TK_L;

    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaPhiPull_TK_C;
    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaPhiPull_TK_H;
    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaPhiPull_TK_L;

    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaThetaPull_TK_C;
    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaThetaPull_TK_H;
    std::map< std::pair< int, int >, std::vector< TH1F* > > mapWS_v_hDeltaThetaPull_TK_L;

    std::map< std::pair< std::string, int >, TH1F* > mapMS_hDTMatch_InvPt; 

    TH1F* hMuonTP_Pt_DT;
    TH1F* hMuonTP_Pt_DTTF;
    TH1F* hMuonTP_Pt_DTMatch;
    TH1F* hMuonTP_Pt_DTMatch_TTTrack;
    TH1F* hMuonTP_Pt_DTMatch_Majority;
    TH1F* hMuonTP_Pt_DTMatch_MixedMode;
    TH1F* hMuonTP_Pt_DTMatch_MajorityFull;
    TH1F* hMuonTP_Pt_DTMatch_Priority;
    TH1F* hMuonTP_Pt_DTMatch_Average;
    TH1F* hMuonTP_Pt_DTMatch_TTTrackFullReso;

    TH1F* hMuonTP_PtBin_DT;
    TH1F* hMuonTP_PtBin_DTTF;
    TH1F* hMuonTP_PtBin_DTMatch;
    TH1F* hMuonTP_PtBin_DTMatch_TTTrack;
    TH1F* hMuonTP_PtBin_DTMatch_Majority;
    TH1F* hMuonTP_PtBin_DTMatch_MixedMode;
    TH1F* hMuonTP_PtBin_DTMatch_MajorityFull;
    TH1F* hMuonTP_PtBin_DTMatch_Priority;
    TH1F* hMuonTP_PtBin_DTMatch_Average;
    TH1F* hMuonTP_PtBin_DTMatch_TTTrackFullReso;

    TH1F* hMuonTP_Eta_DT;
    TH1F* hMuonTP_Eta_DTTF;
    TH1F* hMuonTP_Eta_DTMatch;
    TH1F* hMuonTP_Eta_DTMatch_TTTrack;
    TH1F* hMuonTP_Eta_DTMatch_Majority;
    TH1F* hMuonTP_Eta_DTMatch_MixedMode;
    TH1F* hMuonTP_Eta_DTMatch_MajorityFull;
    TH1F* hMuonTP_Eta_DTMatch_Priority;
    TH1F* hMuonTP_Eta_DTMatch_Average;
    TH1F* hMuonTP_Eta_DTMatch_TTTrackFullReso;

    TH2F* hDTTF_PtBin_MuonTP_Pt;
    TH2F* hDTMatch_TTrack_PtBin_MuonTP_Pt;
    TH2F* hDTMatch_Majority_PtBin_MuonTP_Pt;
    TH2F* hDTMatch_MixedMode_PtBin_MuonTP_Pt;
    TH2F* hDTMatch_MajorityFull_PtBin_MuonTP_Pt;
    TH2F* hDTMatch_Priority_PtBin_MuonTP_Pt;
    TH2F* hDTMatch_Average_PtBin_MuonTP_Pt;
    TH2F* hDTMatch_TTTrackFullReso_Pt_MuonTP_Pt;

    /// Containers of parameters passed by python
    /// configuration file
    edm::ParameterSet config;
};

/// Class implementation

/// Constructor
AnalyzerDTMatches::AnalyzerDTMatches( edm::ParameterSet const& iConfig )
 : config(iConfig)
{
  /// Get from the parameter set
  useBothMuonCharges = iConfig.getParameter< bool >("BothCharges");
  selectPositiveMuons = iConfig.getParameter< bool >("GetPositive");
}

/// Destructor
AnalyzerDTMatches::~AnalyzerDTMatches()
{
  /// Insert here what you need to delete
  /// when you close the class instance
}

/// End job
void AnalyzerDTMatches::endJob()//edm::Run& run, const edm::EventSetup& iSetup
{
  /// Things to be done at the exit of the event Loop
  std::cerr << " AnalyzerDTMatches::endJob" << std::endl;
  /// End of things to be done at the exit from the event Loop
}

/// Begin job
void AnalyzerDTMatches::beginJob()
{
  /// Initialize all slave variables
  /// mainly histogram ranges and resolution
  std::ostringstream histoName;
  std::ostringstream histoTitle;

  /// Things to be done before entering the event Loop
  std::cerr << " AnalyzerDTMatches::beginJob" << std::endl;

  /// Book histograms etc
  edm::Service< TFileService > fs;

  /// Prepare for DT Pt-bin way
  int NumBins = 25;
  double* BinVec = new double[NumBins+1];
  BinVec[0] = 0;
  BinVec[1] = 4;
  BinVec[2] = 5;
  BinVec[3] = 6;
  BinVec[4] = 7;
  BinVec[5] = 8;
  BinVec[6] = 10;
  BinVec[7] = 12;
  BinVec[8] = 14;
  BinVec[9] = 16;
  BinVec[10] = 18;
  BinVec[11] = 20;
  BinVec[12] = 25;
  BinVec[13] = 30;
  BinVec[14] = 35;
  BinVec[15] = 40;
  BinVec[16] = 45;
  BinVec[17] = 50;
  BinVec[18] = 60;
  BinVec[19] = 70;
  BinVec[20] = 80;
  BinVec[21] = 90;
  BinVec[22] = 100;
  BinVec[23] = 120;
  BinVec[24] = 140;
  BinVec[25] = 200;

  for ( int iWh = -2; iWh < 3; iWh++ )
  {
    for ( int iSt = 1; iSt < 3; iSt++ )
    {
      /// Prepare the key of the map
      std::pair< int, int > thisKey = std::make_pair( iWh, iSt );

      histoName.str("");  histoName << "hPhiB_C_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#phi_{B} in wheel " << iWh << " station " << iSt << ", correlated";
      mapWS_hPhiB_C[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 800, -400, 400 );
      mapWS_hPhiB_C[thisKey]->Sumw2();

      histoName.str("");  histoName << "hPhiB_H_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#phi_{B} in wheel " << iWh << " station " << iSt << ", H single";
      mapWS_hPhiB_H[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 800, -400, 400 );
      mapWS_hPhiB_H[thisKey]->Sumw2();

      histoName.str("");  histoName << "hPhiB_L_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#phi_{B} in wheel " << iWh << " station " << iSt << ", L single";
      mapWS_hPhiB_L[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 800, -400, 400 );
      mapWS_hPhiB_L[thisKey]->Sumw2();

      histoName.str("");  histoName << "hPhiG_C_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "CMS #phi in wheel " << iWh << " station " << iSt << ", correlated";
      mapWS_hPhiG_C[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 2600, 0, 26000 );
      mapWS_hPhiG_C[thisKey]->Sumw2();

      histoName.str("");  histoName << "hPhiG_H_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "CMS #phi in wheel " << iWh << " station " << iSt << ", H single";
      mapWS_hPhiG_H[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 2600, 0, 26000 );
      mapWS_hPhiG_H[thisKey]->Sumw2();

      histoName.str("");  histoName << "hPhiG_L_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "CMS #phi in wheel " << iWh << " station " << iSt << ", L single";
      mapWS_hPhiG_L[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 2600, 0, 26000 );
      mapWS_hPhiG_L[thisKey]->Sumw2();

      histoName.str("");  histoName << "hPhi_C_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#phi in wheel " << iWh << " station " << iSt << ", correlated";
      mapWS_hPhi_C[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 2600, 0, 26000 );
      mapWS_hPhi_C[thisKey]->Sumw2();

      histoName.str("");  histoName << "hPhi_H_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#phi in wheel " << iWh << " station " << iSt << ", H single";
      mapWS_hPhi_H[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 2600, 0, 26000 );
      mapWS_hPhi_H[thisKey]->Sumw2();

      histoName.str("");  histoName << "hPhi_L_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#phi in wheel " << iWh << " station " << iSt << ", L single";
      mapWS_hPhi_L[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 2600, 0, 26000 );
      mapWS_hPhi_L[thisKey]->Sumw2();

      histoName.str("");  histoName << "hTheta_C_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#theta in wheel " << iWh << " station " << iSt << ", correlated";
      mapWS_hTheta_C[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 1300, 0, 13000 );
      mapWS_hTheta_C[thisKey]->Sumw2();

      histoName.str("");  histoName << "hTheta_H_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#theta in wheel " << iWh << " station " << iSt << ", H single";
      mapWS_hTheta_H[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 1300, 0, 13000 );
      mapWS_hTheta_H[thisKey]->Sumw2();

      histoName.str("");  histoName << "hTheta_L_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#theta in wheel " << iWh << " station " << iSt << ", L single";
      mapWS_hTheta_L[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 1300, 0, 13000 );
      mapWS_hTheta_L[thisKey]->Sumw2();

      histoName.str("");  histoName << "hTPPhi_C_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "sim. muon #phi in wheel " << iWh << " station " << iSt << ", correlated";
      mapWS_hTPPhi_C[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 2600, 0, 26000 );
      mapWS_hTPPhi_C[thisKey]->Sumw2();

      histoName.str("");  histoName << "hTPPhi_H_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "sim. muon #phi in wheel " << iWh << " station " << iSt << ", H single";
      mapWS_hTPPhi_H[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 2600, 0, 26000 );
      mapWS_hTPPhi_H[thisKey]->Sumw2();

      histoName.str("");  histoName << "hTPPhi_L_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "sim. muon #phi in wheel " << iWh << " station " << iSt << ", L single";
      mapWS_hTPPhi_L[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 2600, 0, 26000 );
      mapWS_hTPPhi_L[thisKey]->Sumw2();

      histoName.str("");  histoName << "hTPTheta_C_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "sim. muon #theta in wheel " << iWh << " station " << iSt << ", correlated";
      mapWS_hTPTheta_C[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 1300, 0, 13000 );
      mapWS_hTPTheta_C[thisKey]->Sumw2();

      histoName.str("");  histoName << "hTPTheta_H_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "sim. muon #theta in wheel " << iWh << " station " << iSt << ", H single";
      mapWS_hTPTheta_H[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 1300, 0, 13000 );
      mapWS_hTPTheta_H[thisKey]->Sumw2();

      histoName.str("");  histoName << "hTPTheta_L_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "sim. muon #theta in wheel " << iWh << " station " << iSt << ", L single";
      mapWS_hTPTheta_L[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 1300, 0, 13000 );
      mapWS_hTPTheta_L[thisKey]->Sumw2();

      histoName.str("");  histoName << "hDeltaPhi_TP_C_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#Delta#phi in wheel (wrt Sim) " << iWh << " station " << iSt << ", correlated";
      mapWS_hDeltaPhi_TP_C[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 4000, -4000, 4000 );
      mapWS_hDeltaPhi_TP_C[thisKey]->Sumw2();

      histoName.str("");  histoName << "hDeltaPhi_TP_H_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#Delta#phi in wheel (wrt Sim) " << iWh << " station " << iSt << ", H single";
      mapWS_hDeltaPhi_TP_H[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 4000, -4000, 4000 );
      mapWS_hDeltaPhi_TP_H[thisKey]->Sumw2();

      histoName.str("");  histoName << "hDeltaPhi_TP_L_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#Delta#phi in wheel (wrt Sim) " << iWh << " station " << iSt << ", L single";
      mapWS_hDeltaPhi_TP_L[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 4000, -4000, 4000 );
      mapWS_hDeltaPhi_TP_L[thisKey]->Sumw2();

      histoName.str("");  histoName << "hDeltaTheta_TP_C_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#Delta#theta in wheel (wrt Sim) " << iWh << " station " << iSt << ", correlated";
      mapWS_hDeltaTheta_TP_C[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 400, -2000, 2000 );
      mapWS_hDeltaTheta_TP_C[thisKey]->Sumw2();

      histoName.str("");  histoName << "hDeltaTheta_TP_H_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#Delta#theta in wheel (wrt Sim) " << iWh << " station " << iSt << ", H single";
      mapWS_hDeltaTheta_TP_H[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 400, -2000, 2000 );
      mapWS_hDeltaTheta_TP_H[thisKey]->Sumw2();

      histoName.str("");  histoName << "hDeltaTheta_TP_L_" << iWh+2 << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "#Delta#theta in wheel (wrt Sim) " << iWh << " station " << iSt << ", L single";
      mapWS_hDeltaTheta_TP_L[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 400, -2000, 2000 );
      mapWS_hDeltaTheta_TP_L[thisKey]->Sumw2();

      /// Projections
      mapWS_v_hDeltaPhi_TK_C[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaPhi_TK_H[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaPhi_TK_L[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaTheta_TK_C[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaTheta_TK_H[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaTheta_TK_L[thisKey] = std::vector< TH1F* >();

      for ( unsigned int iLay = 0; iLay < 7; iLay++ )
      {
        histoName.str("");  histoName << "hDeltaPhi_TK_C_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "#Delta#phi in wheel (wrt Tk) " << iWh << " station " << iSt << " layer " << iLay << ", correlated";
        mapWS_v_hDeltaPhi_TK_C[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 4000, -4000, 4000 ) );
        mapWS_v_hDeltaPhi_TK_C[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaPhi_TK_H_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "#Delta#phi in wheel (wrt Tk) " << iWh << " station " << iSt << " layer " << iLay << ", H single";
        mapWS_v_hDeltaPhi_TK_H[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 4000, -4000, 4000 ) );
        mapWS_v_hDeltaPhi_TK_H[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaPhi_TK_L_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "#Delta#phi in wheel (wrt Tk) " << iWh << " station " << iSt << " layer " << iLay << ", L single";
        mapWS_v_hDeltaPhi_TK_L[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 4000, -4000, 4000 ) );
        mapWS_v_hDeltaPhi_TK_L[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaTheta_TK_C_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "#Delta#theta in wheel (wrt Tk) " << iWh << " station " << iSt << " layer " << iLay << ", correlated";
        mapWS_v_hDeltaTheta_TK_C[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 400, -2000, 2000 ) );
        mapWS_v_hDeltaTheta_TK_C[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaTheta_TK_H_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "#Delta#theta in wheel (wrt Tk) " << iWh << " station " << iSt << " layer " << iLay << ", H single";
        mapWS_v_hDeltaTheta_TK_H[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 400, -2000, 2000 ) );
        mapWS_v_hDeltaTheta_TK_H[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaTheta_TK_L_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "#Delta#theta in wheel (wrt Tk) " << iWh << " station " << iSt << " layer " << iLay << ", L single";
        mapWS_v_hDeltaTheta_TK_L[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 400, -2000, 2000 ) );
        mapWS_v_hDeltaTheta_TK_L[thisKey].at(iLay)->Sumw2();
      } /// Projections

      /// Residuals
      mapWS_v_hDeltaPhiResid_TK_C[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaPhiResid_TK_H[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaPhiResid_TK_L[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaThetaResid_TK_C[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaThetaResid_TK_H[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaThetaResid_TK_L[thisKey] = std::vector< TH1F* >();

      for ( unsigned int iLay = 0; iLay < 7; iLay++ )
      {
        histoName.str("");  histoName << "hDeltaPhiResid_TK_C_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "#Delta#phi in wheel (Tk wrt Pred Tk) " << iWh << " station " << iSt << " layer " << iLay << ", correlated";
        mapWS_v_hDeltaPhiResid_TK_C[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 4000, -4000, 4000 ) );
        mapWS_v_hDeltaPhiResid_TK_C[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaPhiResid_TK_H_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "#Delta#phi in wheel (Tk wrt Pred Tk) " << iWh << " station " << iSt << " layer " << iLay << ", H single";
        mapWS_v_hDeltaPhiResid_TK_H[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 4000, -4000, 4000 ) );
        mapWS_v_hDeltaPhiResid_TK_H[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaPhiResid_TK_L_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "#Delta#phi in wheel (Tk wrt Pred Tk) " << iWh << " station " << iSt << " layer " << iLay << ", L single";
        mapWS_v_hDeltaPhiResid_TK_L[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 4000, -4000, 4000 ) );
        mapWS_v_hDeltaPhiResid_TK_L[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaThetaResid_TK_C_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "#Delta#theta in wheel (Tk wrt Pred Tk) " << iWh << " station " << iSt << " layer " << iLay << ", correlated";
        mapWS_v_hDeltaThetaResid_TK_C[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 800, -4000, 4000 ) );
        mapWS_v_hDeltaThetaResid_TK_C[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaThetaResid_TK_H_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "#Delta#theta in wheel (Tk wrt Pred Tk) " << iWh << " station " << iSt << " layer " << iLay << ", H single";
        mapWS_v_hDeltaThetaResid_TK_H[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 800, -4000, 4000 ) );
        mapWS_v_hDeltaThetaResid_TK_H[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaThetaResid_TK_L_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "#Delta#theta in wheel (Tk wrt Pred Tk) " << iWh << " station " << iSt << " layer " << iLay << ", L single";
        mapWS_v_hDeltaThetaResid_TK_L[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 800, -4000, 4000 ) );
        mapWS_v_hDeltaThetaResid_TK_L[thisKey].at(iLay)->Sumw2();
      } /// End residuals

      /// Pulls
      mapWS_v_hDeltaPhiPull_TK_C[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaPhiPull_TK_H[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaPhiPull_TK_L[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaThetaPull_TK_C[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaThetaPull_TK_H[thisKey] = std::vector< TH1F* >();
      mapWS_v_hDeltaThetaPull_TK_L[thisKey] = std::vector< TH1F* >();

      for ( unsigned int iLay = 0; iLay < 7; iLay++ )
      {
        histoName.str("");  histoName << "hDeltaPhiPull_TK_C_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "Pull #Delta#phi in wheel (Tk wrt Pred Tk) " << iWh << " station " << iSt << " layer " << iLay << ", correlated";
        mapWS_v_hDeltaPhiPull_TK_C[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 400, -10, 10 ) );
        mapWS_v_hDeltaPhiPull_TK_C[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaPhiPull_TK_H_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "Pull #Delta#phi in wheel (Tk wrt Pred Tk) " << iWh << " station " << iSt << " layer " << iLay << ", H single";
        mapWS_v_hDeltaPhiPull_TK_H[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 400, -10, 10 ) );
        mapWS_v_hDeltaPhiPull_TK_H[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaPhiPull_TK_L_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "Pull #Delta#phi in wheel (Tk wrt Pred Tk) " << iWh << " station " << iSt << " layer " << iLay << ", L single";
        mapWS_v_hDeltaPhiPull_TK_L[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 400, -10, 10 ) );
        mapWS_v_hDeltaPhiPull_TK_L[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaThetaPull_TK_C_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "Pull #Delta#theta in wheel (Tk wrt Pred Tk) " << iWh << " station " << iSt << " layer " << iLay << ", correlated";
        mapWS_v_hDeltaThetaPull_TK_C[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 400, -10, 10 ) );
        mapWS_v_hDeltaThetaPull_TK_C[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaThetaPull_TK_H_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "Pull #Delta#theta in wheel (Tk wrt Pred Tk) " << iWh << " station " << iSt << " layer " << iLay << ", H single";
        mapWS_v_hDeltaThetaPull_TK_H[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 400, -10, 10 ) );
        mapWS_v_hDeltaThetaPull_TK_H[thisKey].at(iLay)->Sumw2();

        histoName.str("");  histoName << "hDeltaThetaPull_TK_L_" << iWh+2 << "_" << iSt-1 << "_" << iLay;
        histoTitle.str(""); histoTitle << "Pull #Delta#theta in wheel (Tk wrt Pred Tk) " << iWh << " station " << iSt << " layer " << iLay << ", L single";
        mapWS_v_hDeltaThetaPull_TK_L[thisKey].push_back( fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 400, -10, 10 ) );
        mapWS_v_hDeltaThetaPull_TK_L[thisKey].at(iLay)->Sumw2();
      } /// End pulls
    }
  }

  for ( int iMeth = 0; iMeth < 16; iMeth++ )
  {
    for ( int iSt = 1; iSt < 3; iSt++ )
    {
      std::pair< std::string, int > thisKey = std::make_pair(theMethods[iMeth], iSt);

      histoName.str("");  histoName << "hDTMatch_InvPt_" << theMethods[iMeth].c_str() << "_" << iSt-1;
      histoTitle.str(""); histoTitle << "Pt Method " << theMethods[iMeth].c_str() << " station " << iSt;
      mapMS_hDTMatch_InvPt[thisKey] = fs->make<TH1F>( histoName.str().c_str(), histoTitle.str().c_str(), 1000, 0, 0.5 );
      mapMS_hDTMatch_InvPt[thisKey]->Sumw2();
    }
  }

  /// Pt for matching efficiencies
  hMuonTP_Pt_DT = fs->make<TH1F>( "hMuonTP_Pt_DT", "sim muon p_{T}, signal in DT", 200, 0, 200 );
  hMuonTP_Pt_DTTF = fs->make<TH1F>( "hMuonTP_Pt_DTTF", "sim muon p_{T}, w/ DTTF", 200, 0, 200 );
  hMuonTP_Pt_DTMatch = fs->make<TH1F>( "hMuonTP_Pt_DTMatch", "sim muon p_{T}, DTMatch", 200, 0, 200 );
  hMuonTP_Pt_DTMatch_TTTrack = fs->make<TH1F>( "hMuonTP_Pt_DTMatch_TTTrack", "sim muon p_{T}, DTMatch + L1 Track", 200, 0, 200 );
  hMuonTP_Pt_DTMatch_Majority = fs->make<TH1F>( "hMuonTP_Pt_DTMatch_Majority", "sim muon p_{T}, DTMatch + Stubs, Majority", 200, 0, 200 );
  hMuonTP_Pt_DTMatch_MixedMode = fs->make<TH1F>( "hMuonTP_Pt_DTMatch_MixedMode", "sim muon p_{T}, DTMatch + Stubs, Mixed Mode", 200, 0, 200 );
  hMuonTP_Pt_DTMatch_MajorityFull = fs->make<TH1F>( "hMuonTP_Pt_DTMatch_MajorityFull", "sim muon p_{T}, DTMatch + Stubs, Majority Full Tk", 200, 0, 200 );
  hMuonTP_Pt_DTMatch_Priority = fs->make<TH1F>( "hMuonTP_Pt_DTMatch_Priority", "sim muon p_{T}, DTMatch + Stubs, Priority", 200, 0, 200 );
  hMuonTP_Pt_DTMatch_Average = fs->make<TH1F>( "hMuonTP_Pt_DTMatch_Average", "sim muon p_{T}, DTMatch + Stubs, Average", 200, 0, 200 );
  hMuonTP_Pt_DTMatch_TTTrackFullReso = fs->make<TH1F>( "hMuonTP_Pt_DTMatch_TTTrackFullReso", "sim muon p_{T}, DTMatch + L1 Track", 200, 0, 200 );

  hMuonTP_PtBin_DT = fs->make<TH1F>( "hMuonTP_PtBin_DT", "sim muon p_{T}, signal in DT", 200, 0, 200 );
  hMuonTP_PtBin_DTTF = fs->make<TH1F>( "hMuonTP_PtBin_DTTF", "sim muon p_{T}, w/ DTTF", 200, 0, 200 );
  hMuonTP_PtBin_DTMatch = fs->make<TH1F>( "hMuonTP_PtBin_DTMatch", "sim muon p_{T}, DTMatch", 200, 0, 200 );
  hMuonTP_PtBin_DTMatch_TTTrack = fs->make<TH1F>( "hMuonTP_PtBin_DTMatch_TTTrack", "sim muon p_{T}, DTMatch + L1 Track", 200, 0, 200 );
  hMuonTP_PtBin_DTMatch_Majority = fs->make<TH1F>( "hMuonTP_PtBin_DTMatch_Majority", "sim muon p_{T}, DTMatch + Stubs, Majority", 200, 0, 200 );
  hMuonTP_PtBin_DTMatch_MixedMode = fs->make<TH1F>( "hMuonTP_PtBin_DTMatch_MixedMode", "sim muon p_{T}, DTMatch + Stubs, Mixed Mode", 200, 0, 200 );
  hMuonTP_PtBin_DTMatch_MajorityFull = fs->make<TH1F>( "hMuonTP_PtBin_DTMatch_MajorityFull", "sim muon p_{T}, DTMatch + Stubs, Majority Full Tk", 200, 0, 200 );
  hMuonTP_PtBin_DTMatch_Priority = fs->make<TH1F>( "hMuonTP_PtBin_DTMatch_Priority", "sim muon p_{T}, DTMatch + Stubs, Priority", 200, 0, 200 );
  hMuonTP_PtBin_DTMatch_Average = fs->make<TH1F>( "hMuonTP_PtBin_DTMatch_Average", "sim muon p_{T}, DTMatch + Stubs, Average", 200, 0, 200 );
  hMuonTP_PtBin_DTMatch_TTTrackFullReso = fs->make<TH1F>( "hMuonTP_PtBin_DTMatch_TTTrackFullReso", "sim muon p_{T}, DTMatch + L1 Track", 200, 0, 200 );

  hMuonTP_PtBin_DT->GetXaxis()->Set( NumBins, BinVec );
  hMuonTP_PtBin_DTTF->GetXaxis()->Set( NumBins, BinVec );
  hMuonTP_PtBin_DTMatch->GetXaxis()->Set( NumBins, BinVec );
  hMuonTP_PtBin_DTMatch_TTTrack->GetXaxis()->Set( NumBins, BinVec );
  hMuonTP_PtBin_DTMatch_Majority->GetXaxis()->Set( NumBins, BinVec );
  hMuonTP_PtBin_DTMatch_MixedMode->GetXaxis()->Set( NumBins, BinVec );
  hMuonTP_PtBin_DTMatch_MajorityFull->GetXaxis()->Set( NumBins, BinVec );
  hMuonTP_PtBin_DTMatch_Priority->GetXaxis()->Set( NumBins, BinVec );
  hMuonTP_PtBin_DTMatch_Average->GetXaxis()->Set( NumBins, BinVec );
  hMuonTP_PtBin_DTMatch_TTTrackFullReso->GetXaxis()->Set( NumBins, BinVec );

  hMuonTP_Pt_DT->Sumw2();
  hMuonTP_Pt_DTTF->Sumw2();
  hMuonTP_Pt_DTMatch->Sumw2();
  hMuonTP_Pt_DTMatch_TTTrack->Sumw2();
  hMuonTP_Pt_DTMatch_Majority->Sumw2();
  hMuonTP_Pt_DTMatch_MixedMode->Sumw2();
  hMuonTP_Pt_DTMatch_MajorityFull->Sumw2();
  hMuonTP_Pt_DTMatch_Priority->Sumw2();
  hMuonTP_Pt_DTMatch_Average->Sumw2();
  hMuonTP_Pt_DTMatch_TTTrackFullReso->Sumw2();

  hMuonTP_PtBin_DT->Sumw2();
  hMuonTP_PtBin_DTTF->Sumw2();
  hMuonTP_PtBin_DTMatch->Sumw2();
  hMuonTP_PtBin_DTMatch_TTTrack->Sumw2();
  hMuonTP_PtBin_DTMatch_Majority->Sumw2();
  hMuonTP_PtBin_DTMatch_MixedMode->Sumw2();
  hMuonTP_PtBin_DTMatch_MajorityFull->Sumw2();
  hMuonTP_PtBin_DTMatch_Priority->Sumw2();
  hMuonTP_PtBin_DTMatch_Average->Sumw2();
  hMuonTP_PtBin_DTMatch_TTTrackFullReso->Sumw2();

  hMuonTP_Eta_DT = fs->make<TH1F>( "hMuonTP_Eta_DT", "sim muon #eta, signal in DT", 100, -1, 1 );
  hMuonTP_Eta_DTTF = fs->make<TH1F>( "hMuonTP_Eta_DTTF", "sim muon #eta, w/ DTTF", 100, -1, 1 );
  hMuonTP_Eta_DTMatch = fs->make<TH1F>( "hMuonTP_Eta_DTMatch", "sim muon #eta, DTMatch", 100, -1, 1 );
  hMuonTP_Eta_DTMatch_TTTrack = fs->make<TH1F>( "hMuonTP_Eta_DTMatch_TTTrack", "sim muon #eta, DTMatch + L1 Track", 100, -1, 1 );
  hMuonTP_Eta_DTMatch_Majority = fs->make<TH1F>( "hMuonTP_Eta_DTMatch_Majority", "sim muon #eta, DTMatch + Stubs, Majority", 100, -1, 1 );
  hMuonTP_Eta_DTMatch_MixedMode = fs->make<TH1F>( "hMuonTP_Eta_DTMatch_MixedMode", "sim muon #eta, DTMatch + Stubs, Mixed Mode", 100, -1, 1 );
  hMuonTP_Eta_DTMatch_MajorityFull = fs->make<TH1F>( "hMuonTP_Eta_DTMatch_MajorityFull", "sim muon #eta, DTMatch + Stubs, Majority Full Tk", 100, -1, 1 );
  hMuonTP_Eta_DTMatch_Priority = fs->make<TH1F>( "hMuonTP_Eta_DTMatch_Priority", "sim muon #eta, DTMatch + Stubs, Priority", 100, -1, 1 );
  hMuonTP_Eta_DTMatch_Average = fs->make<TH1F>( "hMuonTP_Eta_DTMatch_Average", "sim muon #eta, DTMatch + Stubs, Average", 100, -1, 1 );
  hMuonTP_Eta_DTMatch_TTTrackFullReso = fs->make<TH1F>( "hMuonTP_Eta_DTMatch_TTTrackFullReso", "sim muon #eta, DTMatch + L1 Track", 100, -1, 1 );

  hMuonTP_Eta_DT->Sumw2();
  hMuonTP_Eta_DTTF->Sumw2();
  hMuonTP_Eta_DTMatch->Sumw2();
  hMuonTP_Eta_DTMatch_TTTrack->Sumw2();
  hMuonTP_Eta_DTMatch_Majority->Sumw2();
  hMuonTP_Eta_DTMatch_MixedMode->Sumw2();
  hMuonTP_Eta_DTMatch_MajorityFull->Sumw2();
  hMuonTP_Eta_DTMatch_Priority->Sumw2();
  hMuonTP_Eta_DTMatch_Average->Sumw2();
  hMuonTP_Eta_DTMatch_TTTrackFullReso->Sumw2();

  hDTTF_PtBin_MuonTP_Pt = fs->make<TH2F>( "hDTTF_PtBin_MuonTP_Pt", "DTTF p_{T} vs sim muon p_{T}", 200, 0, 200, 200, 0, 200 );
  hDTMatch_TTrack_PtBin_MuonTP_Pt = fs->make<TH2F>( "hDTMatch_TTrack_PtBin_MuonTP_Pt", "DTMatch TTTrack p_{T} vs sim muon p_{T}", 200, 0, 200, 200, 0, 200 );
  hDTMatch_Majority_PtBin_MuonTP_Pt = fs->make<TH2F>( "hDTMatch_Majority_PtBin_MuonTP_Pt", "DTMatch Majority p_{T} vs sim muon p_{T}", 200, 0, 200, 200, 0, 200 );
  hDTMatch_MixedMode_PtBin_MuonTP_Pt = fs->make<TH2F>( "hDTMatch_MixedMode_PtBin_MuonTP_Pt", "DTMatch MixedMode p_{T} vs sim muon p_{T}", 200, 0, 200, 200, 0, 200 );
  hDTMatch_MajorityFull_PtBin_MuonTP_Pt = fs->make<TH2F>( "hDTMatch_MajorityFull_PtBin_MuonTP_Pt", "DTMatch Majority Full Tk p_{T} vs sim muon p_{T}", 200, 0, 200, 200, 0, 200 );
  hDTMatch_Priority_PtBin_MuonTP_Pt = fs->make<TH2F>( "hDTMatch_Priority_PtBin_MuonTP_Pt", "DTMatch Priority p_{T} vs sim muon p_{T}", 200, 0, 200, 200, 0, 200 );
  hDTMatch_Average_PtBin_MuonTP_Pt = fs->make<TH2F>( "hDTMatch_Average_PtBin_MuonTP_Pt", "DTMatch Average p_{T} vs sim muon p_{T}", 200, 0, 200, 200, 0, 200 );
  hDTMatch_TTTrackFullReso_Pt_MuonTP_Pt = fs->make<TH2F>( "hDTMatch_TTTrackFullReso_Pt_MuonTP_Pt", "DTMatch TTTrack Full Reso p_{T} vs sim muon p_{T}", 200, 0, 200, 200, 0, 200 );

  hDTTF_PtBin_MuonTP_Pt->GetYaxis()->Set( NumBins, BinVec );
  hDTMatch_TTrack_PtBin_MuonTP_Pt->GetYaxis()->Set( NumBins, BinVec );
  hDTMatch_Majority_PtBin_MuonTP_Pt->GetYaxis()->Set( NumBins, BinVec );
  hDTMatch_MixedMode_PtBin_MuonTP_Pt->GetYaxis()->Set( NumBins, BinVec );
  hDTMatch_MajorityFull_PtBin_MuonTP_Pt->GetYaxis()->Set( NumBins, BinVec );
  hDTMatch_Priority_PtBin_MuonTP_Pt->GetYaxis()->Set( NumBins, BinVec );
  hDTMatch_Average_PtBin_MuonTP_Pt->GetYaxis()->Set( NumBins, BinVec );
  hDTMatch_TTTrackFullReso_Pt_MuonTP_Pt->GetYaxis()->Set( NumBins, BinVec );

  hDTTF_PtBin_MuonTP_Pt->Sumw2();
  hDTMatch_TTrack_PtBin_MuonTP_Pt->Sumw2();
  hDTMatch_Majority_PtBin_MuonTP_Pt->Sumw2();
  hDTMatch_MixedMode_PtBin_MuonTP_Pt->Sumw2();
  hDTMatch_MajorityFull_PtBin_MuonTP_Pt->Sumw2();
  hDTMatch_Priority_PtBin_MuonTP_Pt->Sumw2();
  hDTMatch_Average_PtBin_MuonTP_Pt->Sumw2();
  hDTMatch_TTTrackFullReso_Pt_MuonTP_Pt->Sumw2();

  /// End of things to be done before entering the event Loop
}

/// Analyze
void AnalyzerDTMatches::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  /// Geometry handles etc
  edm::ESHandle< TrackerGeometry >                GeometryHandle;
  edm::ESHandle< StackedTrackerGeometry >         StackedGeometryHandle;
  const StackedTrackerGeometry*                   theStackedGeometry;
  StackedTrackerGeometry::StackContainerIterator  StackedTrackerIterator;

  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get< TrackerDigiGeometryRecord >().get(GeometryHandle);
  /// Set pointers to Stacked Modules
  iSetup.get< StackedTrackerGeometryRecord >().get(StackedGeometryHandle);
  theStackedGeometry = StackedGeometryHandle.product(); /// Note this is different
                                                        /// from the "global" geometry

  /// DT Geometry handle etc
  edm::ESHandle< DTGeometry > DTGeometryHandle;
  iSetup.get< MuonGeometryRecord >().get( DTGeometryHandle );

/*
  /// Magnetic Field
  edm::ESHandle< MagneticField > magneticFieldHandle;
  iSetup.get< IdealMagneticFieldRecord >().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();
*/

  /// TrackingParticles
  edm::Handle< std::vector< TrackingParticle > > TrackingParticleHandle;
  iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingParticleHandle );
  edm::Handle< std::vector< TrackingVertex > > TrackingVertexHandle;
  iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingVertexHandle );

  /// Track Trigger
  edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > TTStubHandle;
  iEvent.getByLabel( "TTStubsFromPixelDigis", "StubAccepted", TTStubHandle );
  edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > > TTTrackHandle;
  iEvent.getByLabel( "TTTracksFromPixelDigis", "Level1TTTracks", TTTrackHandle );

  /// Track Trigger MC Truth
  edm::Handle< TTStubAssociationMap< Ref_PixelDigi_ > > MCTruthTTStubHandle;
  iEvent.getByLabel( "TTStubAssociatorFromPixelDigis", "StubAccepted", MCTruthTTStubHandle );
  edm::Handle< TTTrackAssociationMap< Ref_PixelDigi_ > > MCTruthTTTrackHandle;
  iEvent.getByLabel( "TTTrackAssociatorFromPixelDigis", "Level1TTTracks", MCTruthTTTrackHandle );

  /// DT Trigger
#ifdef DEBUG
  edm::Handle< std::vector< DTBtiTrigger > >   DTBtiTriggerHandle;
#endif
  edm::Handle< std::vector< DTMatch > >     DTMatchHandle;

#ifdef DEBUG
  iEvent.getByLabel( "DTPlusTrackProducer", DTBtiTriggerHandle );
#endif
  iEvent.getByLabel( "DTPlusTrackProducer", DTMatchHandle );

  /// DT Digis
  edm::Handle< DTDigiCollection > DTDigiHandle;
  iEvent.getByLabel( "simMuonDTDigis", DTDigiHandle );

  /// DT Digi Sim Link
  edm::Handle< DTDigiSimLinkCollection > DTDigiSimLinkHandle;
  iEvent.getByLabel( "simMuonDTDigis", DTDigiSimLinkHandle );

#ifdef DEBUG
  /// Loop over the DT Triggers
  std::vector< DTBtiTrigger >::const_iterator iterBti;
  for ( iterBti = DTBtiTriggerHandle->begin();
        iterBti != DTBtiTriggerHandle->end();
        ++iterBti )
  {
    //iterBti->print();
  }
#endif

  /// Preliminary task: map SimTracks by TrackingParticle
  /// Prepare the map
  std::map< std::pair< unsigned int, EncodedEventId >, edm::Ptr< TrackingParticle > > mapSimTrackUniqueToTP;
  mapSimTrackUniqueToTP.clear();

  if ( TrackingParticleHandle->size() != 0 )
  {
    /// Loop over TrackingParticles
    unsigned int tpCnt = 0;
    std::vector< TrackingParticle >::const_iterator iterTPart;
    for ( iterTPart = TrackingParticleHandle->begin();
          iterTPart != TrackingParticleHandle->end();
          ++iterTPart )
    {
      /// Make the pointer to the TrackingParticle
      edm::Ptr< TrackingParticle > tempTPPtr( TrackingParticleHandle, tpCnt++ );

      /// Get the EncodedEventId
      EncodedEventId eventId = EncodedEventId( tempTPPtr->eventId() );

      /// Loop over SimTracks inside TrackingParticle
      std::vector< SimTrack >::const_iterator iterSimTrack;
      for ( iterSimTrack = tempTPPtr->g4Tracks().begin();
            iterSimTrack != tempTPPtr->g4Tracks().end();
            ++iterSimTrack )
      {
        /// Build the unique SimTrack Id (which is SimTrack ID + EncodedEventId)
        std::pair< unsigned int, EncodedEventId > simTrackUniqueId( iterSimTrack->trackId(), eventId );
        mapSimTrackUniqueToTP.insert( std::make_pair( simTrackUniqueId, tempTPPtr ) );
      }
    } /// End of loop over TrackingParticles
  }

  /// Prepare container for muons that leave signal in the DT wires
  std::map< edm::Ptr< TrackingParticle >, std::vector< bool > > mapMuonTrackingParticleInDTDigi;
  mapMuonTrackingParticleInDTDigi.clear();

  /// This is the idea: if a muon is found in the DT digis, it goes into the map and it is
  /// contributor to the denominator for efficiencies by definition.
  /// Then, the vector of booleans is used to check if the same muon is found in
  /// other matches, according to the following index definition
  /// 0) DTTF
  /// 1) DTMatch
  /// 2) DTMatch + Stubs, Priority
  /// 3) DTMatch + Stubs, Average
  /// 4) DTMatch + Stubs, Majority Full Tracker
  /// 5) DTMatch + Stubs, Majority
  /// 6) DTMatch + Stubs, Mixed Mode
  /// 7) DTMatch + TTTrack

  /// Prepare the maps for the DT Digi to DT Digi SimLink association
  std::map< DTBtiId, std::vector< edm::Ptr< TrackingParticle > > > mapTrackingParticleByBti;
  mapTrackingParticleByBti.clear();

  /// Here's the idea:
  /// DTDigi can be associated 1-to-1 with DTDigiSimLink using DTWireId and number/time
  /// Then from DTDigiSimLink one can get the unique id std::pair< unsigned int, EncodedEventId >
  /// of the TP, so if they are stored simultaneously in the second element of the pairs
  /// in the maps above, one will have that each DTDigiSimLink in the vector is associated
  /// 1-to-1 to the TrackingParticle
  /// DTTrigGeom allows to convert a wire into the corresponding BTI, so that every TrackingParticle
  /// can be mapped with respect to the BTI, i.e. each BTI can be associated to all the TrackingParticles
  /// that fired it

  /// Loop over the DT Digi SimLink
  DTDigiSimLinkCollection::DigiRangeIterator iterDTSLUnit;
  std::vector< DTDigiSimLink >::const_iterator iterDTSL;
  for ( iterDTSLUnit = DTDigiSimLinkHandle->begin();
        iterDTSLUnit != DTDigiSimLinkHandle->end();
        ++iterDTSLUnit )
  {
    /// Get the layer, chamber Id's and the DT Trigger geometry for this chamber
    /// all of these are needed to find the BTI Id of the wire associated to this DigiSimLink
    const DTLayerId& thisLayId = (*iterDTSLUnit).first;
    const DTChamberId thisChambId = thisLayId.superlayerId().chamberId();
    const DTChamber* thisChamber = DTGeometryHandle->chamber(thisChambId);
    DTTrigGeom* thisChamberGeometry = new DTTrigGeom( const_cast< DTChamber* >(thisChamber), false );
    const DTDigiSimLinkCollection::Range& theseDigiSimLinksRange = (*iterDTSLUnit).second;

    for ( iterDTSL = theseDigiSimLinksRange.first;
          iterDTSL != theseDigiSimLinksRange.second;
          ++iterDTSL )
    {
      /// Build the unique Id to find the TrackingParticle
      std::pair< unsigned int, EncodedEventId > thisUniqueId = std::make_pair( iterDTSL->SimTrackId(), iterDTSL->eventId() );

      /// TrackingParticle pointer to be retrieved (if present) from the map
      edm::Ptr< TrackingParticle > thisTPPtr;

      /// Check if the TP is available in the map
      if ( mapSimTrackUniqueToTP.find( thisUniqueId ) != mapSimTrackUniqueToTP.end() )
      {
        thisTPPtr = mapSimTrackUniqueToTP[ thisUniqueId ];
      }
      else
      {
        thisTPPtr = edm::Ptr< TrackingParticle >();
      }

      /// Get the wire Id
      DTWireId thisWireId( thisLayId, iterDTSL->wire() );

      /// Build the channel number
      int channelNum = thisChamberGeometry->mapTubeInFEch( thisLayId.superlayerId().superlayer(), thisLayId.layer(), thisWireId.wire() );

      /// Build the BTI Id
      /// Note that each channel can be mapped into 2 or 3 BTI's
      /// Channel N goes into BTI N and N+1
      /// Only for Layer 4: also in N+2
      int maxShift = 1;
      if ( thisLayId.layer() == 4 )
      {
        maxShift = 2;
      }

      for ( int jShift = -maxShift; jShift <= maxShift; jShift++ )
      {
        /// Build the BTI number
        int btiNum = channelNum + jShift;

        if ( btiNum < 0 )
        {
          continue;
        }

        /// Build the BTI Id
        DTBtiId thisBtiId( thisChambId, thisLayId.superlayerId().superlayer(), btiNum );

        /// Check if the BTI is already in the map or not
        //if ( mapDTSimLinkByBti.find( thisBtiId ) != mapDTSimLinkByBti.end() )
        if ( mapTrackingParticleByBti.find( thisBtiId ) != mapTrackingParticleByBti.end() )
        {
          mapTrackingParticleByBti[ thisBtiId ].push_back( thisTPPtr );
        }
        else
        {
          std::vector< edm::Ptr< TrackingParticle > > tempVector2;
          tempVector2.push_back( thisTPPtr );
          mapTrackingParticleByBti.insert( std::make_pair( thisBtiId, tempVector2 ) );
        }
      } /// End of loop over possible shifts of the BTI

      /// Check if it is a muon
      if ( thisTPPtr.isNull() == false &&
           fabs(thisTPPtr->pdgId()) == 13 )
      {
        /// Check if the muon is already in the container for the efficiencies
        if ( mapMuonTrackingParticleInDTDigi.find( thisTPPtr ) == mapMuonTrackingParticleInDTDigi.end() )
        {
          /// Add it to the map if not present
          std::vector< bool > tempVec;
          for ( unsigned int jK = 0; jK < 9; jK++ )
          {
            /// 0) DTTF
            /// 1) DTMatch
            /// 2) DTMatch + Stubs, Priority
            /// 3) DTMatch + Stubs, Average
            /// 4) DTMatch + Stubs, Majority Full Tracker
            /// 5) DTMatch + Stubs, Majority
            /// 6) DTMatch + Stubs, Mixed Mode
            /// 7) DTMatch + TTTrack
            /// 8) DTMatch + TTTrack, Full Resolution

            tempVec.push_back(false);
          }

          mapMuonTrackingParticleInDTDigi.insert( std::make_pair( thisTPPtr, tempVec ) );
        }
      } /// End of it is a muon

    } /// End of loop over the DTDigiSimLinks
  } /// End of loop over the DT unit

#ifdef DEBUG
  std::cerr << "MAP OF SIM MUONS THAT FIRED DTWires has " << mapMuonTrackingParticleInDTDigi.size() << std::endl;
#endif

  /// Now, convert the map< BTI, vector< TP > > to a map< BTI, vector< muon TP > >
  /// and to a map< BTI, one single muon TP >
  std::map< DTBtiId, std::vector< edm::Ptr< TrackingParticle > > > mapSimMuonsByBti;
  std::map< DTBtiId, edm::Ptr< TrackingParticle > > mapTheSimMuonByBti;
  mapSimMuonsByBti.clear();
  mapTheSimMuonByBti.clear();

  std::map< DTBtiId, std::vector< edm::Ptr< TrackingParticle > > >::const_iterator iterMapTrackingParticleByBti;

  for ( iterMapTrackingParticleByBti = mapTrackingParticleByBti.begin();
        iterMapTrackingParticleByBti != mapTrackingParticleByBti.end();
        ++iterMapTrackingParticleByBti )
  {
    DTBtiId thisBtiId = iterMapTrackingParticleByBti->first;
    std::vector< edm::Ptr< TrackingParticle > > theseTrackingParticles = mapTrackingParticleByBti[ thisBtiId ]; /// Safe by construction!

    /// Prepare the map for muons purged
    std::vector< edm::Ptr< TrackingParticle > > theseSimMuons;
    theseSimMuons.clear();

    /// Loop over the TrackingParticles associated to the current BTI
    for ( unsigned int j = 0; j < theseTrackingParticles.size(); j++ )
    {
      if ( theseTrackingParticles.at(j).isNull() )
      {
        continue;
      }
      else
      {
        /// Here the TrackingParticle is found in thisBtiId AND is not null
        edm::Ptr< TrackingParticle > thisTPPtr = theseTrackingParticles.at(j);
        if ( fabs(thisTPPtr->pdgId()) == 13 )
        {
          /// Push back only non-duplicate muons!
          bool muonFound = false;
          for ( unsigned int k = 0; k < theseSimMuons.size() && !muonFound; k++ )
          {
            if ( thisTPPtr.get() == theseSimMuons.at(k).get() )
            {
              muonFound = true;
            }
          }

          if ( !muonFound )
          {
            theseSimMuons.push_back( thisTPPtr );
          }
        } /// End of non-null muons
      } /// End of non-null TrackingParticles
    } /// End of loop over TrackingParticles

    if ( theseSimMuons.size() > 0 )
    {
      mapSimMuonsByBti.insert( std::make_pair( thisBtiId, theseSimMuons ) );
    }
    if ( theseSimMuons.size() == 1 )
    {
      mapTheSimMuonByBti.insert( std::make_pair( thisBtiId, theseSimMuons.at(0) ) );
    }

#ifdef DEBUG
    std::cerr << thisBtiId.SLId() << " " << thisBtiId.bti() << std::endl;
    if ( mapSimMuonsByBti.find( thisBtiId ) != mapSimMuonsByBti.end() )
    {
      //std::cerr << "TPs   = " << mapTrackingParticleByBti[ thisBtiId ].size() << std::endl;
      //std::cerr << "MUONS = " << mapSimMuonsByBti[ thisBtiId ].size() << std::endl;
    }
    if ( mapTheSimMuonByBti.find( thisBtiId ) != mapTheSimMuonByBti.end() )
    {
      std::cerr << "  >>> THE MUON IS " << mapTheSimMuonByBti[ thisBtiId ]->momentum() << std::endl;
    }
#endif

#ifdef DEBUG
/*
    std::cerr << thisBtiId.SLId() << " " << thisBtiId.bti() << std::endl;
    if ( theseDTSimLinks.size() != theseTrackingParticles.size() ) std::cerr << "DISASTER HAPPENED" << std::endl;
    else
    {
      for ( unsigned int j = 0; j < theseDTSimLinks.size(); j++ )
      {
        std::cerr << theseDTSimLinks.at(j).number() << " " << theseDTSimLinks.at(j).time() << std::endl;
        std::cerr << theseTrackingParticles.at(j).isNull();
        if ( theseTrackingParticles.at(j).isNull() ) std::cerr << std::endl;
        else
        {
          edm::Ptr< TrackingParticle > thisTPPtr = theseTrackingParticles.at(j);
          std::cerr << " " << thisTPPtr->pdgId() << " " << thisTPPtr->momentum() << std::endl;
        }
      }
    }
*/
#endif
  } /// End of loop over BTI Id

#ifdef DEBUG
  /// Loop over the DT Digis
  DTDigiCollection::DigiRangeIterator iterDTUnit;
  std::vector< DTDigi >::const_iterator iterDTDigi;
  for ( iterDTUnit = DTDigiHandle->begin();
        iterDTUnit != DTDigiHandle->end();
        ++iterDTUnit )
  {
    const DTLayerId& thisLayId = (*iterDTUnit).first;
//    const DTChamberId thisChambId = thisLayId.superlayerId().chamberId();
    const DTDigiCollection::Range& theseDigisRange = (*iterDTUnit).second;

    for ( iterDTDigi = theseDigisRange.first;
          iterDTDigi != theseDigisRange.second;
          ++iterDTDigi )
    {
      DTWireId thisWireId( thisLayId, iterDTDigi->wire() );
//      std::cerr<< thisLayId << " " << thisChambId << " " << thisWireId << " " << iterDTDigi->time() << " ";
//      iterDTDigi->print();
    }
  }
#endif

#ifdef DEBUG
  std::cerr << "number of DTMatches " << DTMatchHandle->size() << std::endl;
#endif

  /// Loop over the DTMatches
  std::vector< DTMatch >::const_iterator iterDTM;
  for ( iterDTM = DTMatchHandle->begin();
        iterDTM != DTMatchHandle->end();
        ++iterDTM )
  {
    /// Skip if wrong BX
    bool flagBXOk = iterDTM->getFlagBXOK();

    if ( !flagBXOk )
    {
      continue;
    }

    /// Get the Id of all the BTIs of the match
    DTBtiId innerBti = iterDTM->getInnerBtiId();
    DTBtiId outerBti = iterDTM->getOuterBtiId();
    DTBtiId matchedBti = iterDTM->getMatchedBtiId();

#ifdef DEBUG
    std::cerr << " I " << innerBti.SLId() << " " << innerBti.bti() << std::endl;
    if ( mapTheSimMuonByBti.find( innerBti ) != mapTheSimMuonByBti.end() )
      std::cerr << " has one muon " << mapTheSimMuonByBti[ innerBti ]->momentum() << std::endl;
    else if ( mapSimMuonsByBti.find( innerBti ) != mapSimMuonsByBti.end() )
      std::cerr << " has " << mapSimMuonsByBti[ innerBti ].size() << " muons" << std::endl;
    else if ( mapTrackingParticleByBti.find( innerBti ) != mapTrackingParticleByBti.end() )
      std::cerr << " has " << mapTrackingParticleByBti[ innerBti ].size() << " non muon particles" << std::endl;

    std::cerr << " O " << outerBti.SLId() << " " << outerBti.bti() << std::endl;
    if ( mapTheSimMuonByBti.find( outerBti ) != mapTheSimMuonByBti.end() )
      std::cerr << " has one muon " << mapTheSimMuonByBti[ outerBti ]->momentum() << std::endl;
    else if ( mapSimMuonsByBti.find( outerBti ) != mapSimMuonsByBti.end() )
      std::cerr << " has " << mapSimMuonsByBti[ outerBti ].size() << " muons" << std::endl;
    else if ( mapTrackingParticleByBti.find( outerBti ) != mapTrackingParticleByBti.end() )
      std::cerr << " has " << mapTrackingParticleByBti[ outerBti ].size() << " non muon particles" << std::endl;

    std::cerr << " M " << matchedBti.SLId() << " " << matchedBti.bti() << std::endl;
    if ( mapTheSimMuonByBti.find( matchedBti ) != mapTheSimMuonByBti.end() )
      std::cerr << " has one muon " << mapTheSimMuonByBti[ matchedBti ]->momentum() << std::endl;
    else if ( mapSimMuonsByBti.find( matchedBti ) != mapSimMuonsByBti.end() )
      std::cerr << " has " << mapSimMuonsByBti[ matchedBti ].size() << " muons" << std::endl;
    else if ( mapTrackingParticleByBti.find( matchedBti ) != mapTrackingParticleByBti.end() )
      std::cerr << " has " << mapTrackingParticleByBti[ outerBti ].size() << " non muon particles" << std::endl;

    std::cerr<<std::endl;
#endif

    /// Get the TrackingParticle muons for the three BTI's that build up this DT seed
    std::map< edm::Ptr< TrackingParticle >, std::vector< DTBtiId > > mapBtiByFiringMuon;

    if ( mapTheSimMuonByBti.find( innerBti ) != mapTheSimMuonByBti.end() )
    {
      /// Only 1 muon fires the BTI
      if ( mapTheSimMuonByBti[ innerBti ].isNull() == false ) /// Robustness
      {
        if ( mapBtiByFiringMuon.find( mapTheSimMuonByBti[ innerBti ] ) != mapBtiByFiringMuon.end() )
        {
          mapBtiByFiringMuon[ mapTheSimMuonByBti[ innerBti ] ].push_back( innerBti );
        }
        else
        {
          std::vector< DTBtiId > tempBtiVec;
          tempBtiVec.push_back( innerBti );
          mapBtiByFiringMuon.insert( std::make_pair( mapTheSimMuonByBti[ innerBti ], tempBtiVec ) );
        }
      }
    }
    else if ( mapSimMuonsByBti.find( innerBti ) != mapSimMuonsByBti.end() )
    {
      /// Case when > 1 muon fires the BTI
      for ( unsigned int q = 0; q < mapSimMuonsByBti[ innerBti ].size(); q++ )
      {
        if ( mapSimMuonsByBti[ innerBti ].at(q).isNull() == false )
        {
          if ( mapBtiByFiringMuon.find( mapSimMuonsByBti[ innerBti ].at(q) ) != mapBtiByFiringMuon.end() )
          {
            mapBtiByFiringMuon[ mapSimMuonsByBti[ innerBti ].at(q) ].push_back( innerBti );
          }
          else
          {
            std::vector< DTBtiId > tempBtiVec;
            tempBtiVec.push_back( innerBti );
            mapBtiByFiringMuon.insert( std::make_pair( mapSimMuonsByBti[ innerBti ].at(q), tempBtiVec ) );
          }
        }
      }
    }

    if ( mapTheSimMuonByBti.find( outerBti ) != mapTheSimMuonByBti.end() )
    {
      /// Only 1 muon fires the BTI
      if ( mapTheSimMuonByBti[ outerBti ].isNull() == false ) /// Robustness
      {
        if ( mapBtiByFiringMuon.find( mapTheSimMuonByBti[ outerBti ] ) != mapBtiByFiringMuon.end() )
        {
          mapBtiByFiringMuon[ mapTheSimMuonByBti[ outerBti ] ].push_back( outerBti );
        }
        else
        {
          std::vector< DTBtiId > tempBtiVec;
          tempBtiVec.push_back( outerBti );
          mapBtiByFiringMuon.insert( std::make_pair( mapTheSimMuonByBti[ outerBti ], tempBtiVec ) );
        }
      }
    }
    else if ( mapSimMuonsByBti.find( outerBti ) != mapSimMuonsByBti.end() )
    {
      /// Case when > 1 muon fires the BTI
      for ( unsigned int q = 0; q < mapSimMuonsByBti[ outerBti ].size(); q++ )
      {
        if ( mapSimMuonsByBti[ outerBti ].at(q).isNull() == false )
        {
          if ( mapBtiByFiringMuon.find( mapSimMuonsByBti[ outerBti ].at(q) ) != mapBtiByFiringMuon.end() )
          {
            mapBtiByFiringMuon[ mapSimMuonsByBti[ outerBti ].at(q) ].push_back( outerBti );
          }
          else
          {
            std::vector< DTBtiId > tempBtiVec;
            tempBtiVec.push_back( outerBti );
            mapBtiByFiringMuon.insert( std::make_pair( mapSimMuonsByBti[ outerBti ].at(q), tempBtiVec ) );
          }
        }
      }
    }

    if ( mapTheSimMuonByBti.find( matchedBti ) != mapTheSimMuonByBti.end() )
    {
      /// Only 1 muon fires the BTI
      if ( mapTheSimMuonByBti[ matchedBti ].isNull() == false ) /// Robustness
      {
        if ( mapBtiByFiringMuon.find( mapTheSimMuonByBti[ matchedBti ] ) != mapBtiByFiringMuon.end() )
        {
          mapBtiByFiringMuon[ mapTheSimMuonByBti[ matchedBti ] ].push_back( matchedBti );
        }
        else
        {
          std::vector< DTBtiId > tempBtiVec;
          tempBtiVec.push_back( matchedBti );
          mapBtiByFiringMuon.insert( std::make_pair( mapTheSimMuonByBti[ matchedBti ], tempBtiVec ) );
        }
      }
    }
    else if ( mapSimMuonsByBti.find( matchedBti ) != mapSimMuonsByBti.end() )
    {
      /// Case when > 1 muon fires the BTI
      for ( unsigned int q = 0; q < mapSimMuonsByBti[ matchedBti ].size(); q++ )
      {
        if ( mapSimMuonsByBti[ matchedBti ].at(q).isNull() == false )
        {
          if ( mapBtiByFiringMuon.find( mapSimMuonsByBti[ matchedBti ].at(q) ) != mapBtiByFiringMuon.end() )
          {
            mapBtiByFiringMuon[ mapSimMuonsByBti[ matchedBti ].at(q) ].push_back( matchedBti );
          }
          else
          {
            std::vector< DTBtiId > tempBtiVec;
            tempBtiVec.push_back( matchedBti );
            mapBtiByFiringMuon.insert( std::make_pair( mapSimMuonsByBti[ matchedBti ].at(q), tempBtiVec ) );
          }
        }
      }
    }

#ifdef DEBUG
    std::cerr << "MAP OF SIM MUONS THAT FIRED BTI in DTMatch has " << mapBtiByFiringMuon.size() << std::endl;
#endif

    /// Now we have a map where each muon firing a BTI --> points to the list of fired Bti
    /// Basic criterion: only 1 muon that fires min. 1/3 BTI
    edm::Ptr< TrackingParticle > theFiringMuon = edm::Ptr< TrackingParticle >();
    std::map< edm::Ptr< TrackingParticle >, std::vector< DTBtiId > >::const_iterator iterMapBtiByFiringMuon;

    bool escapeFlag = false; /// Originally the code was using only iterMapBtiByFiringMuon
                             /// but this happened to prevent infinite loops
    for ( iterMapBtiByFiringMuon = mapBtiByFiringMuon.begin();
          iterMapBtiByFiringMuon != mapBtiByFiringMuon.end() && !escapeFlag;
          ++iterMapBtiByFiringMuon )
    {
      if ( theFiringMuon.isNull() )
      {
        /// First item in the map
        if ( iterMapBtiByFiringMuon->second.size() > 0 )
        {
          theFiringMuon = iterMapBtiByFiringMuon->first;
        }
      }
      else
      {
        /// There is already a muon set! If we get here,
        /// the muon has already been assigned and the present one
        /// is another muon. If also this has min 1/3 BTI matched,
        /// no unique assignment is possible, so force to exit
        if ( iterMapBtiByFiringMuon->second.size() > 0 )
        {
          /// Force to exit and reset muon to null
          theFiringMuon = edm::Ptr< TrackingParticle >();
          escapeFlag = true;
        }
      }

      /// For the efficiency plots: numerators can be filled for all the muons
      /// that fire at least 1 BTI out of the 3 that make the DTMatch
      edm::Ptr< TrackingParticle > thisMuon = iterMapBtiByFiringMuon->first;

      if ( thisMuon.isNull() )
      {
        continue;
      }
      if ( iterMapBtiByFiringMuon->second.size() == 0 )
      {
        continue;
      }

      /// Fill the container for the efficiency plots
      if ( mapMuonTrackingParticleInDTDigi.find( thisMuon ) != mapMuonTrackingParticleInDTDigi.end() )
      {
        double simPt = thisMuon->p4().pt();

        /// 0) DTTF
        /// 1) DTMatch
        /// 2) DTMatch + Stubs, Priority
        /// 3) DTMatch + Stubs, Average
        /// 4) DTMatch + Stubs, Majority Full Tracker
        /// 5) DTMatch + Stubs, Majority
        /// 6) DTMatch + Stubs, Mixed Mode
        /// 7) DTMatch + TTTrack
        /// 8) DTMatch + TTTrack, Full Resolution

        /// 1) DTMatch
        mapMuonTrackingParticleInDTDigi[ thisMuon ][1] = true;

        /// 2) DTMatch + Stubs, Priority
        if ( iterDTM->getPtPriorityBin() > 0. )
        {
          mapMuonTrackingParticleInDTDigi[ thisMuon ][2] = true;
          hDTMatch_Priority_PtBin_MuonTP_Pt->Fill( simPt, iterDTM->getPtPriorityBin() );
        }

        /// 3) DTMatch + Stubs, Average
        if ( iterDTM->getPtAverageBin() > 0. )
        {
          mapMuonTrackingParticleInDTDigi[ thisMuon ][3] = true;
          hDTMatch_Average_PtBin_MuonTP_Pt->Fill( simPt, iterDTM->getPtAverageBin() );
        }

        /// 4) DTMatch + Stubs, Majority Full Tracker
        if ( iterDTM->getPtMajorityFullTkBin() > 0. )
        {
          mapMuonTrackingParticleInDTDigi[ thisMuon ][4] = true;
          hDTMatch_MajorityFull_PtBin_MuonTP_Pt->Fill( simPt, iterDTM->getPtMajorityFullTkBin() );
        }

        /// 5) DTMatch + Stubs, Majority
        if ( iterDTM->getPtMajorityBin() > 0. )
        {
          mapMuonTrackingParticleInDTDigi[ thisMuon ][5] = true;
          hDTMatch_Majority_PtBin_MuonTP_Pt->Fill( simPt, iterDTM->getPtMajorityBin() );
        }

        /// 6) DTMatch + Stubs, Mixed Mode
        if ( iterDTM->getPtMixedModeBin() > 0. )
        {
          mapMuonTrackingParticleInDTDigi[ thisMuon ][6] = true;
          hDTMatch_MixedMode_PtBin_MuonTP_Pt->Fill( simPt, iterDTM->getPtMixedModeBin() );
        }

        /// 7) DTMatch + TTTrack
        if ( iterDTM->getPtTTTrackBin() > 0. )
        {
          mapMuonTrackingParticleInDTDigi[ thisMuon ][7] = true;
          hDTMatch_TTrack_PtBin_MuonTP_Pt->Fill( simPt, iterDTM->getPtTTTrackBin() );
        }

        /// 8) DTMatch + TTTrack, Full Resolution
        if ( iterDTM->getPtMatchedTrackPtr().isNull() == false )
        {
          mapMuonTrackingParticleInDTDigi[ thisMuon ][8] = true;
          hDTMatch_TTTrackFullReso_Pt_MuonTP_Pt->Fill( simPt, iterDTM->getPtMatchedTrackPtr()->getMomentum().perp() );
        }


      }

    } /// End of loop over all possible firing muons

    /// Here, if there is one good muon that builds up this TP seed, it is stored in theFiringMuon
    /// if theFiringMuon.isNull(), it means that either no muon 2/3 BTI is found, either more than 1 were found
#ifdef DEBUG
    if ( theFiringMuon.isNull() )
    {
      std::cerr << " I " << innerBti.SLId() << " " << innerBti.bti() << std::endl;
      std::cerr << " O " << outerBti.SLId() << " " << outerBti.bti() << std::endl;
      std::cerr << " M " << matchedBti.SLId() << " " << matchedBti.bti() << std::endl;
      std::cerr << "NULL firing muon" << std::endl;
    }
#endif

    if ( theFiringMuon.isNull() == false )
    {
#ifdef DEBUG
      std::cerr << " I " << innerBti.SLId() << " " << innerBti.bti() << std::endl;
      std::cerr << " O " << outerBti.SLId() << " " << outerBti.bti() << std::endl;
      std::cerr << " M " << matchedBti.SLId() << " " << matchedBti.bti() << std::endl;
      std::cerr << "FIRED by muon " << theFiringMuon->momentum() << std::endl;
#endif

      /// Select only the charges you want
      int muonCharge = theFiringMuon->charge();
      if ( !useBothMuonCharges )
      {
        if ( selectPositiveMuons && ( muonCharge < 0 ) )
        {
          continue;
        }
        if ( !selectPositiveMuons && ( muonCharge > 0 ) )
        {
          continue;
        }
      }

      /// Get the DT Wheel, Station etc. information
      int thisDTWheel = iterDTM->getDTWheel();
      int thisDTStation = iterDTM->getDTStation();
      int thisDTCode = iterDTM->getDTCode();

#ifdef DEBUG
      std::cerr << " * wheel and station ******* " << thisDTWheel << " * " << thisDTStation << std::endl;
#endif

      bool isCorr = false;
      bool isHs = false;
      bool isLs = false;

      if ( thisDTStation == 1 )
      {
        if ( thisDTCode == 27 || thisDTCode == 26 ||
             thisDTCode == 23 || thisDTCode == 22 ||
             thisDTCode == 19 || thisDTCode == 18 )
        {
          isCorr = true;
        }
        else if ( thisDTCode == 15 || thisDTCode == 14 ||
                  thisDTCode == 11 || thisDTCode == 10 )
        {
          isHs = true;
        }
        else if ( thisDTCode == 7 || thisDTCode == 6 ||
                  thisDTCode == 3 || thisDTCode == 2 )
        {
          isLs = true;
        }
      }
      if ( thisDTStation == 2 )
      {
        if ( thisDTCode == 25 || thisDTCode == 24 ||
             thisDTCode == 21 || thisDTCode == 20 ||
             thisDTCode == 17 || thisDTCode == 16 )
        {
          isCorr = true;
        }
        else if ( thisDTCode == 13 || thisDTCode == 12 ||
                  thisDTCode == 9 || thisDTCode == 8 )
        {
          isHs = true;
        }
        else if ( thisDTCode == 5 || thisDTCode == 4 ||
                  thisDTCode == 1 || thisDTCode == 0 )
        {
          isLs = true;
        }
      }

      if ( isCorr == false && isHs == false && isLs == false )
      {
        /// Exit if the code cannot be assigned to any kind of TP seed
        continue;
      }

      /// Build the map key
      std::pair< int, int > thisKey = std::make_pair( thisDTWheel, thisDTStation );

      /// Get information from the DT seed
      int thisDTTSPhiB = iterDTM->getDTTSPhiB();
      double thisDTGloPhi = iterDTM->getDTPosition().phi();
      if ( thisDTGloPhi < 0 ) thisDTGloPhi += 2 * M_PI;
      if ( thisDTGloPhi >= 2 * M_PI ) thisDTGloPhi -= 2 * M_PI;

      int thisDTGloPhiInt = static_cast< int >( thisDTGloPhi * 4096. );
      int thisDTTSPhiCorr = iterDTM->getDTTSPhi() + static_cast< int >( (iterDTM->getDTSector() - 1) * M_PI/6. * 4096.);
      int thisDTTSTheta = iterDTM->getDTTSTheta();

      /// Get information from the generated muon
      TrackingParticle::Vector simMuonMomentum = theFiringMuon->momentum();
      double simMuonPhi = simMuonMomentum.phi();
      if ( simMuonPhi < 0 ) simMuonPhi += 2 * M_PI;
      if ( simMuonPhi >= 2 * M_PI ) simMuonPhi -= 2 * M_PI;

      int simMuonPhiInt = static_cast< int >( simMuonPhi * 4096. );
      int simMuonThetaInt = static_cast< int >( simMuonMomentum.theta() * 4096. );

      if ( isCorr )
      {
        mapWS_hPhiB_C[thisKey]->Fill( thisDTTSPhiB );
        mapWS_hPhiG_C[thisKey]->Fill( thisDTGloPhiInt );
        mapWS_hPhi_C[thisKey]->Fill( thisDTTSPhiCorr );
        mapWS_hTheta_C[thisKey]->Fill( thisDTTSTheta );
        mapWS_hTPPhi_C[thisKey]->Fill( simMuonPhiInt );
        mapWS_hTPTheta_C[thisKey]->Fill( simMuonThetaInt );
      }
      else if ( isHs )
      {
        mapWS_hPhiB_H[thisKey]->Fill( thisDTTSPhiB );
        mapWS_hPhiG_H[thisKey]->Fill( thisDTGloPhiInt );
        mapWS_hPhi_H[thisKey]->Fill( thisDTTSPhiCorr );
        mapWS_hTheta_H[thisKey]->Fill( thisDTTSTheta );
        mapWS_hTPPhi_H[thisKey]->Fill( simMuonPhiInt );
        mapWS_hTPTheta_H[thisKey]->Fill( simMuonThetaInt );
      }
      else if ( isLs )
      {
        mapWS_hPhiB_L[thisKey]->Fill( thisDTTSPhiB );
        mapWS_hPhiG_L[thisKey]->Fill( thisDTGloPhiInt );
        mapWS_hPhi_L[thisKey]->Fill( thisDTTSPhiCorr );
        mapWS_hTheta_L[thisKey]->Fill( thisDTTSTheta );
        mapWS_hTPPhi_L[thisKey]->Fill( simMuonPhiInt );
        mapWS_hTPTheta_L[thisKey]->Fill( simMuonThetaInt );
      }

      /// Prepare for calculations of DeltaPhi
      int IMPI = static_cast< int >( M_PI * 4096. );
      int tempPhi1, tempPhi2, tempDeltaPhi;
      int tempPhi3;

      /// Difference w.r.t. Gen
      tempPhi1 = thisDTTSPhiCorr;
      tempPhi2 = simMuonPhiInt;
      if ( tempPhi1 < 0 ) tempPhi1 += 2 * IMPI;
      if ( tempPhi2 < 0 ) tempPhi2 += 2 * IMPI;
      if ( tempPhi1 >= 2 * IMPI ) tempPhi1 -= 2 * IMPI;
      if ( tempPhi2 >= 2 * IMPI ) tempPhi2 -= 2 * IMPI;
      tempDeltaPhi = tempPhi1 - tempPhi2;
      if ( tempDeltaPhi > IMPI ) tempDeltaPhi = 2 * IMPI - tempDeltaPhi;
      if ( tempDeltaPhi < -IMPI ) tempDeltaPhi = - 2 * IMPI + tempDeltaPhi;

      if ( isCorr )
      {
        mapWS_hDeltaPhi_TP_C[thisKey]->Fill( tempDeltaPhi );
        mapWS_hDeltaTheta_TP_C[thisKey]->Fill( thisDTTSTheta - simMuonThetaInt );
      }
      else if ( isHs )
      {
        mapWS_hDeltaPhi_TP_H[thisKey]->Fill( tempDeltaPhi );
        mapWS_hDeltaTheta_TP_H[thisKey]->Fill( thisDTTSTheta - simMuonThetaInt );
      }
      else if ( isLs )
      {
        mapWS_hDeltaPhi_TP_L[thisKey]->Fill( tempDeltaPhi );
        mapWS_hDeltaTheta_TP_L[thisKey]->Fill( thisDTTSTheta - simMuonThetaInt );
      }

      /// Get the L1 tracks associated with the current muon
      std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > > theseTracks = MCTruthTTTrackHandle->findTTTrackPtrs( theFiringMuon );

      if ( theseTracks.size() > 0 )
      {
        for ( unsigned int iTk = 0; iTk < theseTracks.size(); iTk++ )
        {
          edm::Ptr< TTTrack< Ref_PixelDigi_ > > tempTrackPtr = theseTracks.at( iTk );
          bool genuineTrack = MCTruthTTTrackHandle->isGenuine( tempTrackPtr );

          if ( !genuineTrack ) continue;

          double trackPhi = tempTrackPtr->getMomentum().phi();
          if ( trackPhi < 0 ) trackPhi += 2 * M_PI;
          if ( trackPhi >= 2 * M_PI ) trackPhi -= 2 * M_PI;
          tempPhi2 = static_cast< int >( trackPhi * 4096. );
          if ( tempPhi2 < 0 ) tempPhi2 += 2 * IMPI;
          if ( tempPhi2 >= 2 * IMPI ) tempPhi2 -= 2 * IMPI;
          tempDeltaPhi = tempPhi1 - tempPhi2;
          if ( tempDeltaPhi > IMPI ) tempDeltaPhi = 2 * IMPI - tempDeltaPhi;
          if ( tempDeltaPhi < -IMPI ) tempDeltaPhi = - 2 * IMPI + tempDeltaPhi;

          int trackThetaInt = static_cast< int >( tempTrackPtr->getMomentum().theta() * 4096. );

          /// Difference seed to Tk
          if ( isCorr )
          {
            mapWS_v_hDeltaPhi_TK_C[thisKey].at(0)->Fill( tempDeltaPhi );
            mapWS_v_hDeltaTheta_TK_C[thisKey].at(0)->Fill( thisDTTSTheta - trackThetaInt );
          }
          else if ( isHs )
          {
            mapWS_v_hDeltaPhi_TK_H[thisKey].at(0)->Fill( tempDeltaPhi );
            mapWS_v_hDeltaTheta_TK_H[thisKey].at(0)->Fill( thisDTTSTheta - trackThetaInt );
          }
          else if ( isLs )
          {
            mapWS_v_hDeltaPhi_TK_L[thisKey].at(0)->Fill( tempDeltaPhi );
            mapWS_v_hDeltaTheta_TK_L[thisKey].at(0)->Fill( thisDTTSTheta - trackThetaInt );
          }

          /// Difference seed Tk prediction to Tk
          tempPhi3 = iterDTM->getPredVtxPhi();
          if ( tempPhi3 < 0 ) tempPhi2 += 2 * IMPI;
          if ( tempPhi3 >= 2 * IMPI ) tempPhi3 -= 2 * IMPI;
          tempDeltaPhi = tempPhi3 - tempPhi2;
          if ( tempDeltaPhi > IMPI ) tempDeltaPhi = 2 * IMPI - tempDeltaPhi;
          if ( tempDeltaPhi < -IMPI ) tempDeltaPhi = - 2 * IMPI + tempDeltaPhi;

          int predVtxTheta = iterDTM->getPredVtxTheta();
          float predVtxPhiError = static_cast< float >(iterDTM->getPredVtxSigmaPhi());
          float predVtxThetaError = static_cast< float >(iterDTM->getPredVtxSigmaTheta());

          if ( isCorr )
          {
            mapWS_v_hDeltaPhiResid_TK_C[thisKey].at(0)->Fill( tempDeltaPhi );
            mapWS_v_hDeltaThetaResid_TK_C[thisKey].at(0)->Fill( predVtxTheta - trackThetaInt );
            mapWS_v_hDeltaPhiPull_TK_C[thisKey].at(0)->Fill( tempDeltaPhi/predVtxPhiError );
            mapWS_v_hDeltaThetaPull_TK_C[thisKey].at(0)->Fill( (predVtxTheta - trackThetaInt)/predVtxThetaError );
          }
          else if ( isHs )
          {
            mapWS_v_hDeltaPhiResid_TK_H[thisKey].at(0)->Fill( tempDeltaPhi );
            mapWS_v_hDeltaThetaResid_TK_H[thisKey].at(0)->Fill( predVtxTheta - trackThetaInt );
            mapWS_v_hDeltaPhiPull_TK_H[thisKey].at(0)->Fill( tempDeltaPhi/predVtxPhiError );
            mapWS_v_hDeltaThetaPull_TK_H[thisKey].at(0)->Fill( (predVtxTheta - trackThetaInt)/predVtxThetaError );
          }
          else if ( isLs )
          {
            mapWS_v_hDeltaPhiResid_TK_L[thisKey].at(0)->Fill( tempDeltaPhi );
            mapWS_v_hDeltaThetaResid_TK_L[thisKey].at(0)->Fill( predVtxTheta - trackThetaInt );
            mapWS_v_hDeltaPhiPull_TK_L[thisKey].at(0)->Fill( tempDeltaPhi/predVtxPhiError );
            mapWS_v_hDeltaThetaPull_TK_L[thisKey].at(0)->Fill( (predVtxTheta - trackThetaInt)/predVtxThetaError );
          }

        } /// End of loop over L1 tracks
      }

      /// Get also the stubs associated with the current muon
      std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > theseStubs = MCTruthTTStubHandle->findTTStubRefs( theFiringMuon );

      if ( theseStubs.size() > 0 )
      {
        for ( unsigned int iStub = 0; iStub < theseStubs.size(); iStub++ )
        {
          edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > tempStubRef = theseStubs.at( iStub );

          StackedTrackerDetId tkDetId( tempStubRef->getDetId() );
          if ( tkDetId.isBarrel() == false )
          {
            continue;
          }

          bool genuineStub = MCTruthTTStubHandle->isGenuine( tempStubRef );

          if ( !genuineStub ) continue;

          unsigned int jLayer = tkDetId.iLayer();

          GlobalPoint stubPos = theStackedGeometry->findGlobalPosition( tempStubRef.get() );

          double stubPhi = stubPos.phi();
          if ( stubPhi < 0 ) stubPhi += 2 * M_PI;
          if ( stubPhi >= 2 * M_PI ) stubPhi -= 2 * M_PI;
          tempPhi2 = static_cast< int >( stubPhi * 4096. );
          if ( tempPhi2 < 0 ) tempPhi2 += 2 * IMPI;
          if ( tempPhi2 >= 2 * IMPI ) tempPhi2 -= 2 * IMPI;
          tempDeltaPhi = tempPhi1 - tempPhi2;
          if ( tempDeltaPhi > IMPI ) tempDeltaPhi = 2 * IMPI - tempDeltaPhi;
          if ( tempDeltaPhi < -IMPI ) tempDeltaPhi = - 2 * IMPI + tempDeltaPhi;

          int stubThetaInt = static_cast< int >( stubPos.theta() * 4096. );

          if ( isCorr )
          {
            mapWS_v_hDeltaPhi_TK_C[thisKey].at(jLayer)->Fill( tempDeltaPhi );
            mapWS_v_hDeltaTheta_TK_C[thisKey].at(jLayer)->Fill( thisDTTSTheta - stubThetaInt );
          }
          else if ( isHs )
          {
            mapWS_v_hDeltaPhi_TK_H[thisKey].at(jLayer)->Fill( tempDeltaPhi );
            mapWS_v_hDeltaTheta_TK_H[thisKey].at(jLayer)->Fill( thisDTTSTheta - stubThetaInt );
          }
          else if ( isLs )
          {
            mapWS_v_hDeltaPhi_TK_L[thisKey].at(jLayer)->Fill( tempDeltaPhi );
            mapWS_v_hDeltaTheta_TK_L[thisKey].at(jLayer)->Fill( thisDTTSTheta - stubThetaInt );
          }

          tempPhi3 = iterDTM->getPredStubPhi(jLayer);
          if ( tempPhi3 < 0 ) tempPhi2 += 2 * IMPI;
          if ( tempPhi3 >= 2 * IMPI ) tempPhi3 -= 2 * IMPI;
          tempDeltaPhi = tempPhi3 - tempPhi2;
          if ( tempDeltaPhi > IMPI ) tempDeltaPhi = 2 * IMPI - tempDeltaPhi;
          if ( tempDeltaPhi < -IMPI ) tempDeltaPhi = - 2 * IMPI + tempDeltaPhi;

          int predStubTheta = iterDTM->getPredStubTheta(jLayer);
          float predStubPhiError = static_cast< float >(iterDTM->getPredStubSigmaPhi(jLayer));
          float predStubThetaError = static_cast< float >(iterDTM->getPredStubSigmaTheta(jLayer));

          if ( isCorr )
          {
            mapWS_v_hDeltaPhiResid_TK_C[thisKey].at(jLayer)->Fill( tempDeltaPhi );
            mapWS_v_hDeltaThetaResid_TK_C[thisKey].at(jLayer)->Fill( predStubTheta - stubThetaInt );
            mapWS_v_hDeltaPhiPull_TK_C[thisKey].at(jLayer)->Fill( tempDeltaPhi/predStubPhiError );
            mapWS_v_hDeltaThetaPull_TK_C[thisKey].at(jLayer)->Fill( (predStubTheta - stubThetaInt)/predStubThetaError );
          }
          else if ( isHs )
          {
            mapWS_v_hDeltaPhiResid_TK_H[thisKey].at(jLayer)->Fill( tempDeltaPhi );
            mapWS_v_hDeltaThetaResid_TK_H[thisKey].at(jLayer)->Fill( predStubTheta - stubThetaInt );
            mapWS_v_hDeltaPhiPull_TK_H[thisKey].at(jLayer)->Fill( tempDeltaPhi/predStubPhiError );
            mapWS_v_hDeltaThetaPull_TK_H[thisKey].at(jLayer)->Fill( (predStubTheta - stubThetaInt)/predStubThetaError );
          }
          else if ( isLs )
          {
            mapWS_v_hDeltaPhiResid_TK_L[thisKey].at(jLayer)->Fill( tempDeltaPhi );
            mapWS_v_hDeltaThetaResid_TK_L[thisKey].at(jLayer)->Fill( predStubTheta - stubThetaInt );
            mapWS_v_hDeltaPhiPull_TK_L[thisKey].at(jLayer)->Fill( tempDeltaPhi/predStubPhiError );
            mapWS_v_hDeltaThetaPull_TK_L[thisKey].at(jLayer)->Fill( (predStubTheta - stubThetaInt)/predStubThetaError );
          }

        } /// End of loop over stubs
      }

      /// Prepare curvature plots
      /// Availability of stubs is checked using Pt method strings from DTMatchBasePtMethods
      std::map< std::string, DTMatchPt* > thisPtMethodMap = iterDTM->getPtMethodsMap();

      /// Loop over the methods (theMethods[0] = "Mu_2_1" ... theMethods[14] = "Mu_6_5")
      for ( unsigned int iMethod = 0; iMethod < 15; iMethod++ )
      {
        /// Check if the method is available
        if ( thisPtMethodMap.find( theMethods[iMethod] ) != thisPtMethodMap.end() )
        {
          /// Build the key
          std::pair< std::string, int > anotherKey = std::make_pair(theMethods[iMethod], thisDTStation);

          float thisPt = iterDTM->getPt( theMethods[iMethod] );
          float thisPtInv = 1./thisPt; 
          mapMS_hDTMatch_InvPt[anotherKey]->Fill( thisPtInv );
        }
      }

      /// Check also for L1 Track (theMethods[15] == "TTTrack")
      if ( iterDTM->getPtMatchedTrackPtr().isNull() == false )
      {
        std::pair< std::string, int > anotherKey = std::make_pair(theMethods[15], thisDTStation);
        float thisPt = iterDTM->getPtMatchedTrackPtr()->getMomentum().perp();
        float thisPtInv = 1./thisPt;
        mapMS_hDTMatch_InvPt[anotherKey]->Fill( thisPtInv );
      }

    } /// End of check over null muon TrackingParticle

  } /// End of loop over DTMatches


  /// Fill the plots for the efficiency
  std::map< edm::Ptr< TrackingParticle >, std::vector< bool > >::const_iterator iterMapMu;

  for ( iterMapMu = mapMuonTrackingParticleInDTDigi.begin();
        iterMapMu != mapMuonTrackingParticleInDTDigi.end();
        ++iterMapMu )
  {
    edm::Ptr< TrackingParticle > thisMuon = iterMapMu->first;
    std::vector< bool > theseFlags = iterMapMu->second;

    hMuonTP_Pt_DT->Fill( thisMuon->p4().pt() );
    hMuonTP_PtBin_DT->Fill( thisMuon->p4().pt() );
    hMuonTP_Eta_DT->Fill( thisMuon->p4().eta() );

    /// 0) DTTF
    if ( theseFlags.at(0) == true )
    {
      hMuonTP_Pt_DTTF->Fill( thisMuon->p4().pt() );
      hMuonTP_PtBin_DTTF->Fill( thisMuon->p4().pt() );
      hMuonTP_Eta_DTTF->Fill( thisMuon->p4().eta() );
    }

    /// 1) DTMatch
    if ( theseFlags.at(1) == true )
    {
      hMuonTP_Pt_DTMatch->Fill( thisMuon->p4().pt() );
      hMuonTP_PtBin_DTMatch->Fill( thisMuon->p4().pt() );
      hMuonTP_Eta_DTMatch->Fill( thisMuon->p4().eta() );
    }

    /// 2) DTMatch + Stubs, Priority
    if ( theseFlags.at(2) == true )
    {
      hMuonTP_Pt_DTMatch_Priority->Fill( thisMuon->p4().pt() );
      hMuonTP_PtBin_DTMatch_Priority->Fill( thisMuon->p4().pt() );
      hMuonTP_Eta_DTMatch_Priority->Fill( thisMuon->p4().eta() );
    }

    /// 3) DTMatch + Stubs, Average
    if ( theseFlags.at(3) == true )
    {
      hMuonTP_Pt_DTMatch_Average->Fill( thisMuon->p4().pt() );
      hMuonTP_PtBin_DTMatch_Average->Fill( thisMuon->p4().pt() );
      hMuonTP_Eta_DTMatch_Average->Fill( thisMuon->p4().eta() );
    }

    /// 4) DTMatch + Stubs, Majority Full Tracker
    if ( theseFlags.at(4) == true )
    {
      hMuonTP_Pt_DTMatch_MajorityFull->Fill( thisMuon->p4().pt() );
      hMuonTP_PtBin_DTMatch_MajorityFull->Fill( thisMuon->p4().pt() );
      hMuonTP_Eta_DTMatch_MajorityFull->Fill( thisMuon->p4().eta() );
    }

    /// 5) DTMatch + Stubs, Majority
    if ( theseFlags.at(5) == true )
    {
      hMuonTP_Pt_DTMatch_Majority->Fill( thisMuon->p4().pt() );
      hMuonTP_PtBin_DTMatch_Majority->Fill( thisMuon->p4().pt() );
      hMuonTP_Eta_DTMatch_Majority->Fill( thisMuon->p4().eta() );
    }

    /// 6) DTMatch + Stubs, Mixed Mode
    if ( theseFlags.at(6) == true )
    {
      hMuonTP_Pt_DTMatch_MixedMode->Fill( thisMuon->p4().pt() );
      hMuonTP_PtBin_DTMatch_MixedMode->Fill( thisMuon->p4().pt() );
      hMuonTP_Eta_DTMatch_MixedMode->Fill( thisMuon->p4().eta() );
    }

    /// 7) DTMatch + TTTrack
    if ( theseFlags.at(7) == true )
    {
      hMuonTP_Pt_DTMatch_TTTrack->Fill( thisMuon->p4().pt() );
      hMuonTP_PtBin_DTMatch_TTTrack->Fill( thisMuon->p4().pt() );
      hMuonTP_Eta_DTMatch_TTTrack->Fill( thisMuon->p4().eta() );
    }

    /// 8) DTMatch + TTTrack, Full Resolution
    if ( theseFlags.at(8) == true )
    {
      hMuonTP_Pt_DTMatch_TTTrackFullReso->Fill( thisMuon->p4().pt() );
      hMuonTP_PtBin_DTMatch_TTTrackFullReso->Fill( thisMuon->p4().pt() );
      hMuonTP_Eta_DTMatch_TTTrackFullReso->Fill( thisMuon->p4().eta() );
    }






  } /// End of loop over sim muons

} /// End of analyze()

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(AnalyzerDTMatches);

