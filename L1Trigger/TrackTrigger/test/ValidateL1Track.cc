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

class ValidateL1Track : public edm::EDAnalyzer
{
  /// Public methods
  public:
    /// Constructor/destructor
    explicit ValidateL1Track(const edm::ParameterSet& iConfig);
    virtual ~ValidateL1Track();
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
    TH2D* hTrack_RInv_Seed_RInv;
    TH2D* hTrack_RInvRes_Track_Eta;
    TH2D* hTrack_InvPt_Seed_InvPt;
    TH2D* hTrack_InvPtRes_Track_Eta;
    TH2D* hTrack_Pt_Seed_Pt;
    TH2D* hTrack_PtRes_Track_Eta;
    TH2D* hTrack_Phi_Seed_Phi;
    TH2D* hTrack_PhiRes_Track_Eta;
    TH2D* hTrack_Eta_Seed_Eta;
    TH2D* hTrack_EtaRes_Track_Eta;
    TH2D* hTrack_VtxZ0_Seed_VtxZ0;
    TH2D* hTrack_VtxZ0Res_Track_Eta;

    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropBB_deltaRhoPhi_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropBB_deltaRhoPhi;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropBB_deltaZ_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropBB_deltaZ;

    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropBE_deltaRhoPhi_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropBE_deltaRhoPhi;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropBE_deltaR_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropBE_deltaR;

    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropEB_deltaRhoPhi_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropEB_deltaRhoPhi;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropEB_deltaZ_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropEB_deltaZ;

    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropEE_deltaRhoPhi_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropEE_deltaRhoPhi;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropEE_deltaR_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropEE_deltaR;

    TH1D* hTrack_3Stubs_Pt;
    TH1D* hTrack_3Stubs_Phi;
    TH1D* hTrack_3Stubs_Eta;

    TH1D* hTrack_2Stubs_Pt;
    TH1D* hTrack_2Stubs_Phi;
    TH1D* hTrack_2Stubs_Eta;

    TH1D* hTrack_Seed_Pt;
    TH1D* hTrack_Seed_Phi;
    TH1D* hTrack_Seed_Eta;

    TH1D* hSeed_Pt;
    TH1D* hSeed_Phi;
    TH1D* hSeed_Eta;

    TH1D* hTPart_Track_3Stubs_Pt;
    TH1D* hTPart_Track_3Stubs_Phi_Pt5;
    TH1D* hTPart_Track_3Stubs_Eta_Pt5;

    TH1D* hTPart_Track_2Stubs_Pt;
    TH1D* hTPart_Track_2Stubs_Phi_Pt5;
    TH1D* hTPart_Track_2Stubs_Eta_Pt5;

    TH1D* hTPart_Seed_Pt;
    TH1D* hTPart_Seed_Phi_Pt5;
    TH1D* hTPart_Seed_Eta_Pt5;

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

    TH1D* hSeed_N;
    TH2D* hSeed_Pt_TPart_Pt;
    TH2D* hSeed_PtRes_TPart_Eta;
    TH2D* hSeed_InvPt_TPart_InvPt;
    TH2D* hSeed_InvPtRes_TPart_Eta;
    TH2D* hSeed_Phi_TPart_Phi;
    TH2D* hSeed_PhiRes_TPart_Eta;
    TH2D* hSeed_Eta_TPart_Eta;
    TH2D* hSeed_EtaRes_TPart_Eta;
    TH2D* hSeed_VtxZ0_TPart_VtxZ0;
    TH2D* hSeed_VtxZ0Res_TPart_Eta;
    TH2D* hSeed_Chi2_NStubs;
    TH2D* hSeed_Chi2_TPart_Eta;
    TH2D* hSeed_Chi2Red_NStubs;
    TH2D* hSeed_Chi2Red_TPart_Eta;

};

//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
ValidateL1Track::ValidateL1Track(edm::ParameterSet const& iConfig) 
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

}

/////////////
// DESTRUCTOR
ValidateL1Track::~ValidateL1Track()
{
  /// Insert here what you need to delete
  /// when you close the class instance
}  

//////////
// END JOB
void ValidateL1Track::endJob()
{
  /// Things to be done at the exit of the event Loop
  std::cerr << " ValidateL1Track::endJob" << std::endl;
  /// End of things to be done at the exit from the event Loop

  std::cerr<<"DeltaRhoPhi BB"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      mapTrackPropBB_deltaRhoPhi[ mapKey0 ]->GetXaxis()->SetRangeUser(-0.6, 0.6);
      std::cerr<< mapTrackPropBB_deltaRhoPhi[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaRhoPhi BE"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      mapTrackPropBE_deltaRhoPhi[ mapKey0 ]->GetXaxis()->SetRangeUser(-0.6, 0.6);
      std::cerr<< mapTrackPropBE_deltaRhoPhi[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaRhoPhi EB"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      mapTrackPropEB_deltaRhoPhi[ mapKey0 ]->GetXaxis()->SetRangeUser(-0.6, 0.6);
      std::cerr<< mapTrackPropEB_deltaRhoPhi[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaRhoPhi EE"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      mapTrackPropEE_deltaRhoPhi[ mapKey0 ]->GetXaxis()->SetRangeUser(-0.6, 0.6);
      std::cerr<< mapTrackPropEE_deltaRhoPhi[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaZ BB"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      std::cerr<< mapTrackPropBB_deltaZ[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaZ BE"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      std::cerr<< mapTrackPropBE_deltaR[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaZ EB"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      std::cerr<< mapTrackPropEB_deltaZ[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaZ EE"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      std::cerr<< mapTrackPropEE_deltaR[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

}

////////////
// BEGIN JOB
void ValidateL1Track::beginJob()
{
  /// Initialize all slave variables
  /// mainly histogram ranges and resolution
  std::ostringstream histoName;
  std::ostringstream histoTitle;

  /// Things to be done before entering the event Loop
  std::cerr << " ValidateL1Track::beginJob" << std::endl;

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

  hTrack_NStubs_Sector      = fs->make<TH2D>( "hTrack_NStubs_Sector",      "TTTrack number of Stubs vs. Seed sector", 35, -0.5, 34.5, 20, -0.5, 19.5 );
  hTrack_NStubs_Wedge       = fs->make<TH2D>( "hTrack_NStubs_Wedge",       "TTTrack number of Stubs vs. Seed wedge",  35, -0.5, 34.5, 20, -0.5, 19.5 );
  hTrack_Sector_Phi         = fs->make<TH2D>( "hTrack_Sector_Phi",         "Seed sector vs. TTTrack #phi",            180, -M_PI, M_PI, 35, -0.5, 34.5 );
  hTrack_Wedge_Eta          = fs->make<TH2D>( "hTrack_Wedge_Eta",          "Seed wedge vs. TTTrack #eta",             180, -M_PI, M_PI, 35, -0.5, 34.5 );
  hTrack_RInv_Seed_RInv     = fs->make<TH2D>( "hTrack_RInv_Seed_RInv",     "TTTrack radius vs. Seed radius",          200, -0.01, 0.01, 200, -0.01, 0.01 );
  hTrack_RInvRes_Track_Eta  = fs->make<TH2D>( "hTrack_RInvRes_Track_Eta",  "TTTrack radius res. vs #eta",             180, -M_PI, M_PI, 100, -0.005, 0.005 );
  hTrack_Pt_Seed_Pt         = fs->make<TH2D>( "hTrack_Pt_Seed_Pt",         "TTTrack p_{T} vs. Seed p_{T}",            100, 0, 50, 100, 0, 50 );
  hTrack_PtRes_Track_Eta    = fs->make<TH2D>( "hTrack_PtRes_Track_Eta",    "TTTrack p_{T} res. vs #eta",              180, -M_PI, M_PI, 100, -4.0, 4.0 );
  hTrack_InvPt_Seed_InvPt   = fs->make<TH2D>( "hTrack_InvPt_Seed_InvPt",   "TTTrack p_{T}^{-1} vs. Seed p_{T}^{-1}",  200, 0, 0.8, 200, 0, 0.8 );
  hTrack_InvPt_Seed_InvPt->GetXaxis()->Set( NumBins, BinVec );
  hTrack_InvPt_Seed_InvPt->GetYaxis()->Set( NumBins, BinVec );
  hTrack_InvPtRes_Track_Eta = fs->make<TH2D>( "hTrack_InvPtRes_Track_Eta", "TTTrack p_{T}^{-1} res. vs #eta",         180, -M_PI, M_PI, 100, -0.4, 0.4 );
  hTrack_Phi_Seed_Phi       = fs->make<TH2D>( "hTrack_Phi_Seed_Phi",       "TTTrack #phi vs. Seed #phi",              180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hTrack_PhiRes_Track_Eta   = fs->make<TH2D>( "hTrack_PhiRes_Track_Eta",   "TTTrack #phi res. vs #eta",               180, -M_PI, M_PI, 100, -0.1, 0.1 );
  hTrack_Eta_Seed_Eta       = fs->make<TH2D>( "hTrack_Eta_Seed_Eta",       "TTTrack #eta vs. Seed #eta",              180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hTrack_EtaRes_Track_Eta   = fs->make<TH2D>( "hTrack_EtaRes_Track_Eta",   "TTTrack #eta res. vs #eta",               180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hTrack_VtxZ0_Seed_VtxZ0   = fs->make<TH2D>( "hTrack_VtxZ0_Seed_VtxZ0",   "TTTrack z_{vtx} vs. Seed z_{vtx}",        180, -30, 30, 180, -30, 30 );
  hTrack_VtxZ0Res_Track_Eta = fs->make<TH2D>( "hTrack_VtxZ0Res_Track_Eta", "TTTrack z_{vtx} res. vs #eta",            180, -M_PI, M_PI, 100, -20, 20 );

  hTrack_NStubs_Sector->Sumw2();
  hTrack_NStubs_Wedge->Sumw2();
  hTrack_Sector_Phi->Sumw2();
  hTrack_Wedge_Eta->Sumw2();
  hTrack_RInv_Seed_RInv->Sumw2();
  hTrack_RInvRes_Track_Eta->Sumw2();
  hTrack_Pt_Seed_Pt->Sumw2();
  hTrack_PtRes_Track_Eta->Sumw2();
  hTrack_Phi_Seed_Phi->Sumw2();
  hTrack_PhiRes_Track_Eta->Sumw2();
  hTrack_Eta_Seed_Eta->Sumw2();
  hTrack_EtaRes_Track_Eta->Sumw2();
  hTrack_VtxZ0_Seed_VtxZ0->Sumw2();
  hTrack_VtxZ0Res_Track_Eta->Sumw2();

  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey = std::make_pair( seed, targ );

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_Eta_L" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi vs seed #eta, Propagation from L " << seed << " to L " << targ;
      mapTrackPropBB_deltaRhoPhi_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                 180, -M_PI, M_PI, 100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_L" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi, Propagation from L " << seed << " to L " << targ;
      mapTrackPropBB_deltaRhoPhi[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                             100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaZ_Eta_L" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Deltaz vs seed #eta, Propagation from L " << seed << " to L " << targ;
      mapTrackPropBB_deltaZ_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            180, -M_PI, M_PI, 200, -4, 4 );

      histoName.str("");  histoName << "hTrackProp_deltaZ_L" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Deltaz, Propagation from L " << seed << " to L " << targ;
      mapTrackPropBB_deltaZ[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                        200, -4, 4 );

      mapTrackPropBB_deltaRhoPhi_Eta[ mapKey ]->Sumw2();
      mapTrackPropBB_deltaRhoPhi[ mapKey ]->Sumw2();
      mapTrackPropBB_deltaZ_Eta[ mapKey ]->Sumw2();
      mapTrackPropBB_deltaZ[ mapKey ]->Sumw2();

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_Eta_L" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi vs seed #eta, Propagation from L " << seed << " to D " << targ;
      mapTrackPropBE_deltaRhoPhi_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                 180, -M_PI, M_PI, 100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_L" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi, Propagation from L " << seed << " to D " << targ;
      mapTrackPropBE_deltaRhoPhi[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                             100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaR_Eta_L" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho vs seed #eta, Propagation from L " << seed << " to D " << targ;
      mapTrackPropBE_deltaR_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            180, -M_PI, M_PI, 200, -4, 4 );

      histoName.str("");  histoName << "hTrackProp_deltaR_L" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho, Propagation from L " << seed << " to D " << targ;
      mapTrackPropBE_deltaR[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                        200, -4, 4 );

      mapTrackPropBE_deltaRhoPhi_Eta[ mapKey ]->Sumw2();
      mapTrackPropBE_deltaRhoPhi[ mapKey ]->Sumw2();
      mapTrackPropBE_deltaR_Eta[ mapKey ]->Sumw2();
      mapTrackPropBE_deltaR[ mapKey ]->Sumw2();

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_Eta_D" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi vs seed #eta, Propagation from D " << seed << " to L " << targ;
      mapTrackPropEB_deltaRhoPhi_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                 180, -M_PI, M_PI, 100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_D" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi, Propagation from D " << seed << " to L " << targ;
      mapTrackPropEB_deltaRhoPhi[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                             100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaZ_Eta_D" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Deltaz vs seed #eta, Propagation from D " << seed << " to L " << targ;
      mapTrackPropEB_deltaZ_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            180, -M_PI, M_PI, 400, -8, 8 );

      histoName.str("");  histoName << "hTrackProp_deltaZ_D" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Deltaz, Propagation from D " << seed << " to L " << targ;
      mapTrackPropEB_deltaZ[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                        400, -8, 8 );

      mapTrackPropEB_deltaRhoPhi_Eta[ mapKey ]->Sumw2();
      mapTrackPropEB_deltaRhoPhi[ mapKey ]->Sumw2();
      mapTrackPropEB_deltaZ_Eta[ mapKey ]->Sumw2();
      mapTrackPropEB_deltaZ[ mapKey ]->Sumw2();

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_Eta_D" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi vs seed #eta, Propagation from D " << seed << " to D " << targ;
      mapTrackPropEE_deltaRhoPhi_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                 180, -M_PI, M_PI, 100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_D" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi, Propagation from D " << seed << " to D " << targ;
      mapTrackPropEE_deltaRhoPhi[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                             100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaR_Eta_D" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho vs seed #eta, Propagation from D " << seed << " to D " << targ;
      mapTrackPropEE_deltaR_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            180, -M_PI, M_PI, 200, -4, 4 );

      histoName.str("");  histoName << "hTrackProp_deltaR_D" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho, Propagation from D " << seed << " to D " << targ;
      mapTrackPropEE_deltaR[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                        200, -4, 4 );

      mapTrackPropEE_deltaRhoPhi_Eta[ mapKey ]->Sumw2();
      mapTrackPropEE_deltaRhoPhi[ mapKey ]->Sumw2();
      mapTrackPropEE_deltaR_Eta[ mapKey ]->Sumw2();
      mapTrackPropEE_deltaR[ mapKey ]->Sumw2();
    }
  }

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

  hTrack_Seed_Pt   = fs->make<TH1D>( "hTrack_Seed_Pt",  "TTTrack Seed p_{T}", 100, 0, 50 );
  hTrack_Seed_Phi  = fs->make<TH1D>( "hTrack_Seed_Phi", "TTTrack Seed #phi",  180, -M_PI, M_PI );
  hTrack_Seed_Eta  = fs->make<TH1D>( "hTrack_Seed_Eta", "TTTrack Seed #eta",  180, -M_PI, M_PI );
  hTrack_Seed_Pt->Sumw2();
  hTrack_Seed_Phi->Sumw2();
  hTrack_Seed_Eta->Sumw2();

  hSeed_Pt         = fs->make<TH1D>( "hSeed_Pt",  "Seed p_{T}", 100, 0, 50 );
  hSeed_Phi        = fs->make<TH1D>( "hSeed_Phi", "Seed #phi",  180, -M_PI, M_PI );
  hSeed_Eta        = fs->make<TH1D>( "hSeed_Eta", "Seed #eta",  180, -M_PI, M_PI );
  hSeed_Pt->Sumw2();
  hSeed_Phi->Sumw2();
  hSeed_Eta->Sumw2();

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

  hTPart_Seed_Pt           = fs->make<TH1D>( "hTPart_Seed_Pt",      "Seed TPart p_{T}", 100, 0, 50 );
  hTPart_Seed_Phi_Pt5      = fs->make<TH1D>( "hTPart_Seed_Phi_Pt5", "Seed TPart #phi",  180, -M_PI, M_PI );
  hTPart_Seed_Eta_Pt5      = fs->make<TH1D>( "hTPart_Seed_Eta_Pt5", "Seed TPart #eta",  180, -M_PI, M_PI );
  hTPart_Seed_Pt->Sumw2();
  hTPart_Seed_Phi_Pt5->Sumw2();
  hTPart_Seed_Eta_Pt5->Sumw2();

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

  hSeed_N                  = fs->make<TH1D>( "hSeed_N",                  "Number of Seed",                          100, -0.5, 99.5 );
  hSeed_Pt_TPart_Pt        = fs->make<TH2D>( "hSeed_Pt_TPart_Pt",        "Seed p_{T} vs. TPart p_{T}",              100, 0, 50, 100, 0, 50 );
  hSeed_PtRes_TPart_Eta    = fs->make<TH2D>( "hSeed_PtRes_TPart_Eta",    "Seed p_{T} - TPart p_{T} vs. TPart #eta", 180, -M_PI, M_PI, 200, -4.0, 4.0 );
  hSeed_InvPt_TPart_InvPt  = fs->make<TH2D>( "hSeed_InvPt_TPart_InvPt",  "Seed p_{T}^{-1} vs. TPart p_{T}^{-1}",    200, 0, 0.8, 200, 0, 0.8 );
  hSeed_InvPt_TPart_InvPt->GetXaxis()->Set( NumBins, BinVec );
  hSeed_InvPt_TPart_InvPt->GetYaxis()->Set( NumBins, BinVec );
  hSeed_InvPtRes_TPart_Eta = fs->make<TH2D>( "hSeed_InvPtRes_TPart_Eta", "Seed p_{T}^{-1} - TPart p_{T}^{-1}  vs. TPart #eta", 180, -M_PI, M_PI, 100, -1.0, 1.0 );
  hSeed_Phi_TPart_Phi      = fs->make<TH2D>( "hSeed_Phi_TPart_Phi",      "Seed #phi vs. TPart #phi",                           180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hSeed_PhiRes_TPart_Eta   = fs->make<TH2D>( "hSeed_PhiRes_TPart_Eta",   "Seed #phi - TPart #phi vs. TPart #eta",              180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hSeed_Eta_TPart_Eta      = fs->make<TH2D>( "hSeed_Eta_TPart_Eta",      "Seed #eta vs. TPart #eta",                           180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hSeed_EtaRes_TPart_Eta   = fs->make<TH2D>( "hSeed_EtaRes_TPart_Eta",   "Seed #eta - TPart #eta vs. TPart #eta",              180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hSeed_VtxZ0_TPart_VtxZ0  = fs->make<TH2D>( "hSeed_VtxZ0_TPart_VtxZ0",  "Seed z_{vtx} vs. TPart z_{vtx}",                     180, -30, 30, 180, -30, 30 );
  hSeed_VtxZ0Res_TPart_Eta = fs->make<TH2D>( "hSeed_VtxZ0Res_TPart_Eta", "Seed z_{vtx} - TPart z_{vtx} vs. TPart #eta",        180, -M_PI, M_PI, 100, -5, 5 );
  hSeed_Chi2_NStubs        = fs->make<TH2D>( "hSeed_Chi2_NStubs",        "Seed #chi^{2} vs. number of Stubs",                  20, -0.5, 19.5, 200, 0, 50 );
  hSeed_Chi2_TPart_Eta     = fs->make<TH2D>( "hSeed_Chi2_TPart_Eta",     "Seed #chi^{2} vs. TPart #eta",                       180, -M_PI, M_PI, 200, 0, 50 );
  hSeed_Chi2Red_NStubs     = fs->make<TH2D>( "hSeed_Chi2Red_NStubs",     "Seed #chi^{2}/dof vs. number of Stubs",              20, -0.5, 19.5, 200, 0, 10 );
  hSeed_Chi2Red_TPart_Eta  = fs->make<TH2D>( "hSeed_Chi2Red_TPart_Eta",  "Seed #chi^{2}/dof vs. TPart #eta",                   180, -M_PI, M_PI, 200, 0, 10 );
  hSeed_N->Sumw2();
  hSeed_Pt_TPart_Pt->Sumw2();
  hSeed_PtRes_TPart_Eta->Sumw2();
  hSeed_InvPt_TPart_InvPt->Sumw2();
  hSeed_InvPtRes_TPart_Eta->Sumw2();
  hSeed_Phi_TPart_Phi->Sumw2();
  hSeed_PhiRes_TPart_Eta->Sumw2();
  hSeed_Eta_TPart_Eta->Sumw2();
  hSeed_EtaRes_TPart_Eta->Sumw2();
  hSeed_VtxZ0_TPart_VtxZ0->Sumw2();
  hSeed_VtxZ0Res_TPart_Eta->Sumw2();
  hSeed_Chi2_NStubs->Sumw2();
  hSeed_Chi2_TPart_Eta->Sumw2();
  hSeed_Chi2Red_NStubs->Sumw2();
  hSeed_Chi2Red_TPart_Eta->Sumw2();

  /// End of things to be done before entering the event Loop
}

//////////
// ANALYZE
void ValidateL1Track::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  /// Get geometry
  edm::ESHandle< StackedTrackerGeometry >  StackedGeometryHandle;
  const StackedTrackerGeometry*            theStackedGeometry;
  iSetup.get< StackedTrackerGeometryRecord >().get(StackedGeometryHandle);
  theStackedGeometry = StackedGeometryHandle.product();

  /// Track Trigger
  edm::Handle< std::vector< TTCluster< Ref_PixelDigi_ > > > PixelDigiTTClusterHandle;
  edm::Handle< std::vector< TTStub< Ref_PixelDigi_ > > >    PixelDigiTTStubHandle;
  iEvent.getByLabel( "TTClustersFromPixelDigis",            PixelDigiTTClusterHandle );
  iEvent.getByLabel( "TTStubsFromPixelDigis", "StubsPass",  PixelDigiTTStubHandle );
  edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > >   PixelDigiTTSeedHandle;
  edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > >   PixelDigiTTTrackHandle;
  iEvent.getByLabel( "TTTracksFromPixelDigis", "Seeds",     PixelDigiTTSeedHandle );
  iEvent.getByLabel( "TTTracksFromPixelDigis", "NoDup",     PixelDigiTTTrackHandle );

  /// Track Trigger MC Truth
  edm::Handle< TTClusterAssociationMap< Ref_PixelDigi_ > > MCTruthTTClusterHandle;
  edm::Handle< TTStubAssociationMap< Ref_PixelDigi_ > >    MCTruthTTStubHandle;
  iEvent.getByLabel( "TTClusterAssociatorFromPixelDigis",  MCTruthTTClusterHandle );
  iEvent.getByLabel( "TTStubAssociatorFromPixelDigis",     MCTruthTTStubHandle );
  edm::Handle< TTTrackAssociationMap< Ref_PixelDigi_ > >   MCTruthTTTrackHandle;
  edm::Handle< TTTrackAssociationMap< Ref_PixelDigi_ > >   MCTruthTTSeedHandle;
  iEvent.getByLabel( "TTTrackAssociatorFromPixelDigis", "NoDup", MCTruthTTTrackHandle );
  iEvent.getByLabel( "TTTrackAssociatorFromPixelDigis", "Seeds", MCTruthTTSeedHandle );

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
      if ( MCTruthTTClusterHandle->findTTClusterPtrs( tempTPPtr ).size() > 0 )
      {
        hTPart_Cluster_Pt->Fill( tempTPPtr->p4().pt() );
        if ( tempTPPtr->p4().pt() > 5.0 )
        {
          hTPart_Cluster_Phi_Pt5->Fill( tempTPPtr->momentum().phi() );
          hTPart_Cluster_Eta_Pt5->Fill( tempTPPtr->momentum().eta() );
        }

        /// Check if the TP produced any stubs
        if ( MCTruthTTStubHandle->findTTStubPtrs( tempTPPtr ).size() > 0 )
        {
          hTPart_Stub_Pt->Fill( tempTPPtr->p4().pt() );
          if ( tempTPPtr->p4().pt() > 5.0 )
          {
            hTPart_Stub_Phi_Pt5->Fill( tempTPPtr->momentum().phi() );
            hTPart_Stub_Eta_Pt5->Fill( tempTPPtr->momentum().eta() );
          }

          /// Check if the TP produced any seeds
          if ( MCTruthTTSeedHandle->findTTTrackPtrs( tempTPPtr ).size() > 0 )
          {
            hTPart_Seed_Pt->Fill( tempTPPtr->p4().pt() );
            if ( tempTPPtr->p4().pt() > 5.0 )
            {
              hTPart_Seed_Phi_Pt5->Fill( tempTPPtr->momentum().phi() );
              hTPart_Seed_Eta_Pt5->Fill( tempTPPtr->momentum().eta() );
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

                if ( theseTracks.at(it)->getStubPtrs().size() == 2 )
                {
                  found2stubs = true;
                }
                else if ( theseTracks.at(it)->getStubPtrs().size() > 2 )
                {


        /// Additional cross check
        bool hasBL1 = theseTracks.at(it)->hasStubInBarrel(1) ;
if ( hasBL1 )
{
}
                  found3stubs = true;
//}

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
      unsigned int nStubs     = tempTrackPtr->getStubPtrs().size();
      unsigned int seedSector = tempTrackPtr->getSector();
      unsigned int seedWedge  = tempTrackPtr->getWedge();

      hTrack_NStubs_Sector->Fill( seedSector, nStubs );
      hTrack_NStubs_Wedge->Fill( seedWedge, nStubs );

      double trackRInv  = tempTrackPtr->getRInv();
      double trackPt    = tempTrackPtr->getMomentum().perp();
      double trackPhi   = tempTrackPtr->getMomentum().phi();
      double trackEta   = tempTrackPtr->getMomentum().eta();
      double trackVtxZ0 = tempTrackPtr->getVertex().z();
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
      double tpPhi = tpPtr->momentum().phi();
      double tpVtxZ0 = tpPtr->vertex().z();

      if ( nStubs > 2 )
      {
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
        mapTrack_3Stubs_VtxZ0Res_PtEta[ mapKey ]->Fill( trackVtxZ0 - tpVtxZ0 );
//}
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

      /// Go on only if there are TTTracks from PixelDigis
      if ( PixelDigiTTSeedHandle->size() > 0 )
      {
        /// Loop over TTTrack seeds
        std::vector< TTTrack< Ref_PixelDigi_ > >::const_iterator iterSeed;
        for ( iterSeed = PixelDigiTTSeedHandle->begin();
              iterSeed != PixelDigiTTSeedHandle->end();
              ++iterSeed )
        {
          /// Check the track is the same
          bool dontSkip = iterTTTrack->isTheSameAs( *iterSeed );
          if ( !dontSkip ) continue;

          /// Get everything is relevant
          double seedRInv  = iterSeed->getRInv();
          double seedPt    = iterSeed->getMomentum().perp();
          double seedPhi   = iterSeed->getMomentum().phi();
          double seedEta   = iterSeed->getMomentum().eta();
          double seedVtxZ0 = iterSeed->getVertex().z();

          hTrack_RInv_Seed_RInv->Fill( seedRInv, trackRInv );
          hTrack_RInvRes_Track_Eta->Fill( trackEta, trackRInv - seedRInv );
          hTrack_Pt_Seed_Pt->Fill( seedPt, trackPt );
          hTrack_PtRes_Track_Eta->Fill( trackEta, trackPt - seedPt );
          hTrack_InvPt_Seed_InvPt->Fill( 1./seedPt, 1./trackPt );
          hTrack_InvPtRes_Track_Eta->Fill( trackEta, 1./trackPt - 1./seedPt );
          hTrack_Phi_Seed_Phi->Fill( seedPhi, trackPhi );
          hTrack_PhiRes_Track_Eta->Fill( trackEta, trackPhi - seedPhi );
          hTrack_Eta_Seed_Eta->Fill( seedEta, trackEta );
          hTrack_EtaRes_Track_Eta->Fill( trackEta, trackEta - seedEta );
          hTrack_VtxZ0_Seed_VtxZ0->Fill( seedVtxZ0, trackVtxZ0 );
          hTrack_VtxZ0Res_Track_Eta->Fill( trackEta, trackVtxZ0 - seedVtxZ0 );

          /// Propagate seed and check distances
          StackedTrackerDetId detIdInner( iterSeed->getStubPtrs().at(0)->getDetId() );
//          unsigned int seedSL = (unsigned int)((detIdInner.iLayer() + 1)/2);
//          seedSL = ( seedSL > 3 ) ? 3 : seedSL; /// Renormalize 1-3

          unsigned int seedBarrel0 = 0;
          unsigned int seedEndcap0 = 0;
          if ( detIdInner.isBarrel() ) seedBarrel0 = detIdInner.iLayer();
          else if ( detIdInner.isEndcap() ) seedEndcap0 = detIdInner.iDisk();

          /// Loop over track stubs
          for ( unsigned int js = 0; js < iterTTTrack->getStubPtrs().size(); js++ )
          {
            /// Skip Stubs in the Seed
            bool isInSeed = false;
            for ( unsigned int ks = 0; ks < iterSeed->getStubPtrs().size(); ks++ )
            {
              if ( iterTTTrack->getStubPtrs().at(js) == iterSeed->getStubPtrs().at(ks) )
                isInSeed = true;
            }
            if ( isInSeed ) continue;

            /// Candidate SL
            StackedTrackerDetId detIdCand( iterTTTrack->getStubPtrs().at(js)->getDetId() );
//            unsigned int candSL = (unsigned int)((detIdCand.iLayer() + 1)/2);
//            candSL = ( candSL > 3 ) ? 3 : candSL; /// Renormalize 1-3

            unsigned int candBarrel = 0;
            unsigned int candEndcap = 0;
            if ( detIdCand.isBarrel() ) candBarrel = detIdCand.iLayer();
            else if ( detIdCand.isEndcap() ) candEndcap = detIdCand.iDisk();

//            if ( candSL == seedSL ) continue;

            GlobalPoint posStub = theStackedGeometry->findGlobalPosition( &(*iterTTTrack->getStubPtrs().at(js)) );

            if ( candBarrel )
            {
              /// Propagation
              double propPsi = asin( posStub.perp() * 0.5 * seedRInv );
              double propPhi = seedPhi - propPsi;
              double propRhoPsi = 2 * propPsi / seedRInv;
              double propZ = seedVtxZ0 + propRhoPsi * tan( M_PI_2 - iterSeed->getMomentum().theta() );

              /// Calculate displacement
              /// Perform standard trigonometric operations
              double deltaPhi = posStub.phi() - propPhi;
              if ( fabs(deltaPhi) >= M_PI )
              {
                if ( deltaPhi>0 )
                  deltaPhi = deltaPhi - 2*M_PI;
                else
                  deltaPhi = 2*M_PI + deltaPhi;
              }
              double deltaRPhi = deltaPhi * posStub.perp();
              double deltaZ = propZ - posStub.z();

              if ( seedBarrel0 )
              {
                std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seedBarrel0, candBarrel );
                mapTrackPropBB_deltaRhoPhi_Eta[ mapKey0 ]->Fill( seedEta, deltaRPhi );
                mapTrackPropBB_deltaRhoPhi[ mapKey0 ]->Fill( deltaRPhi );
                mapTrackPropBB_deltaZ_Eta[ mapKey0 ]->Fill( seedEta, deltaZ );
                mapTrackPropBB_deltaZ[ mapKey0 ]->Fill( deltaZ );
              }
              else if ( seedEndcap0 )
              {
                std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seedEndcap0, candBarrel );
                mapTrackPropEB_deltaRhoPhi_Eta[ mapKey0 ]->Fill( seedEta, deltaRPhi );
                mapTrackPropEB_deltaRhoPhi[ mapKey0 ]->Fill( deltaRPhi );
                mapTrackPropEB_deltaZ_Eta[ mapKey0 ]->Fill( seedEta, deltaZ );
                mapTrackPropEB_deltaZ[ mapKey0 ]->Fill( deltaZ );
              }
            }
            else if ( candEndcap )
            {
              /// Propagation
              double propPsi = 0.5*( posStub.z() - seedVtxZ0 ) * seedRInv / tan( M_PI_2 - iterSeed->getMomentum().theta() );
              double propPhi = seedPhi - propPsi;
              double propRho = 2 * sin( propPsi ) / seedRInv;
              double deltaPhi = posStub.phi() - propPhi;

              /// Calculate displacement
              if ( fabs(deltaPhi) >= M_PI )
              {
                if ( deltaPhi>0 )
                  deltaPhi = deltaPhi - 2*M_PI;
                else
                  deltaPhi = 2*M_PI + deltaPhi;
              }
              double deltaRPhi = deltaPhi * posStub.perp(); /// OLD VERSION (updated few lines below)
              double deltaR = posStub.perp() - propRho;

              /// NEW VERSION - non-pointing strips correction
              double rhoTrack = 2.0 * sin( 0.5 * seedRInv * ( posStub.z() - seedVtxZ0 ) / tan( M_PI_2 - iterSeed->getMomentum().theta() ) ) / seedRInv;
              double phiTrack = iterSeed->getMomentum().phi() - 0.5 * seedRInv * ( posStub.z() - seedVtxZ0 ) / tan( M_PI_2 - iterSeed->getMomentum().theta() );

              /// Calculate a correction for non-pointing-strips in square modules
              /// Relevant angle is the one between hit and module center, with
              /// vertex at (0, 0). Take snippet from HitMatchingAlgorithm_window201*
              /// POSITION IN TERMS OF PITCH MULTIPLES:
              ///       0 1 2 3 4 5 5 6 8 9 ...
              /// COORD: 0 1 2 3 4 5 6 7 8 9 ...
              /// OUT   | | | | | |x| | | | | | | | | |
              ///
              /// IN    | | | |x|x| | | | | | | | | | |
              ///             THIS is 3.5 (COORD) and 4.0 (POS)
              /// The center of the module is at NROWS/2 (position) and NROWS-0.5 (coordinates)
              const GeomDetUnit* det0 = theStackedGeometry->idToDetUnit( detIdCand, 0 );
              const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
              const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
              std::pair< float, float > pitch0 = top0->pitch();
              MeasurementPoint stubCoord = iterTTTrack->getStubPtrs().at(js)->getClusterPtr(0)->findAverageLocalCoordinates();
              double stubTransvDispl = pitch0.first * ( stubCoord.x() - (top0->nrows()/2 - 0.5) ); /// Difference in coordinates is the same as difference in position
              if ( posStub.z() > 0 )
              {
                stubTransvDispl = - stubTransvDispl;
              }
              double stubPhiCorr = asin( stubTransvDispl / posStub.perp() );
              deltaRPhi = stubTransvDispl - rhoTrack * sin( stubPhiCorr - phiTrack + posStub.phi() );

              if ( seedBarrel0 )
              {
                std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seedBarrel0, candEndcap );
                mapTrackPropBE_deltaRhoPhi_Eta[ mapKey0 ]->Fill( seedEta, deltaRPhi );
                mapTrackPropBE_deltaRhoPhi[ mapKey0 ]->Fill( deltaRPhi );
                mapTrackPropBE_deltaR_Eta[ mapKey0 ]->Fill( seedEta, deltaR);
                mapTrackPropBE_deltaR[ mapKey0 ]->Fill( deltaR );
              }
              else if ( seedEndcap0 )
              {
                std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seedEndcap0, candEndcap );
                mapTrackPropEE_deltaRhoPhi_Eta[ mapKey0 ]->Fill( seedEta, deltaRPhi );
                mapTrackPropEE_deltaRhoPhi[ mapKey0 ]->Fill( deltaRPhi );
                mapTrackPropEE_deltaR_Eta[ mapKey0 ]->Fill( seedEta, deltaR );
                mapTrackPropEE_deltaR[ mapKey0 ]->Fill( deltaR );
              }
            }

          } /// End of loop over track stubs
        } /// End of loop over TTTrack seeds
      }
    } /// End of loop over TTTracks
  }

  hTrack_2Stubs_N->Fill( num2Stubs );
  hTrack_3Stubs_N->Fill( num3Stubs );

  /// Operations needing reversed-nesting
  unsigned int numSeeds = 0;

  if ( PixelDigiTTSeedHandle->size() > 0 )
  {
    /// Loop over Seeds
    unsigned int seedCnt = 0;
    std::vector< TTTrack< Ref_PixelDigi_ > >::const_iterator iterSeed;
    for ( iterSeed = PixelDigiTTSeedHandle->begin();
          iterSeed != PixelDigiTTSeedHandle->end();
          ++iterSeed )
    {
      /// Make the pointer
      edm::Ptr< TTTrack< Ref_PixelDigi_ > > tempSeedPtr( PixelDigiTTSeedHandle, seedCnt++ );

      bool genuineSeed = MCTruthTTSeedHandle->isGenuine( tempSeedPtr );

      if ( !genuineSeed ) continue;
      
      double seedPt    = tempSeedPtr->getMomentum().perp();
      double seedPhi   = tempSeedPtr->getMomentum().phi();
      double seedEta   = tempSeedPtr->getMomentum().eta();
      double seedVtxZ0 = tempSeedPtr->getVertex().z();
      double seedChi2  = tempSeedPtr->getChi2();
      double seedChi2R = tempSeedPtr->getChi2Red();

      hSeed_Pt->Fill( seedPt );
      hSeed_Phi->Fill( seedPhi );
      hSeed_Eta->Fill( seedEta );

      edm::Ptr< TrackingParticle > tpPtr = MCTruthTTSeedHandle->findTrackingParticlePtr( tempSeedPtr );

      /// Get the corresponding vertex and reject the track
      /// if its vertex is outside the beampipe
      if ( tpPtr->vertex().rho() >= 2 )
        continue;

      double tpPt = tpPtr->p4().pt();
      double tpEta = tpPtr->momentum().eta();
      double tpPhi = tpPtr->momentum().phi();
      double tpVtxZ0 = tpPtr->vertex().z();

      numSeeds++;
      hSeed_Pt_TPart_Pt->Fill( tpPt, seedPt );
      hSeed_PtRes_TPart_Eta->Fill( tpEta, seedPt - tpPt);
      hSeed_InvPt_TPart_InvPt->Fill( 1./tpPt, 1./seedPt );
      hSeed_InvPtRes_TPart_Eta->Fill( tpEta, 1./seedPt - 1./tpPt);
      hSeed_Phi_TPart_Phi->Fill( tpPhi, seedPhi );
      hSeed_PhiRes_TPart_Eta->Fill( tpEta, seedPhi - tpPhi);
      hSeed_Eta_TPart_Eta->Fill( tpEta, seedEta );
      hSeed_EtaRes_TPart_Eta->Fill( tpEta, seedEta - tpEta);
      hSeed_VtxZ0_TPart_VtxZ0->Fill( tpVtxZ0, seedVtxZ0);
      hSeed_VtxZ0Res_TPart_Eta->Fill( tpEta, seedVtxZ0 - tpVtxZ0 );
      hSeed_Chi2_NStubs->Fill( iterSeed->getStubPtrs().size(), seedChi2 );
      hSeed_Chi2_TPart_Eta->Fill( tpEta, seedChi2 );
      hSeed_Chi2Red_NStubs->Fill( iterSeed->getStubPtrs().size(), seedChi2R );
      hSeed_Chi2Red_TPart_Eta->Fill( tpEta, seedChi2R );

      unsigned int q = 0;

      if ( PixelDigiTTTrackHandle->size() > 0 )
      {
        std::vector< TTTrack< Ref_PixelDigi_ > >::const_iterator iterTTTrack;
        for ( iterTTTrack = PixelDigiTTTrackHandle->begin();
              iterTTTrack != PixelDigiTTTrackHandle->end();
              ++iterTTTrack )
        {
          unsigned int nStubs = iterTTTrack->getStubPtrs().size();
          if ( nStubs < 3 ) continue;

          bool dontSkip = iterTTTrack->isTheSameAs( *iterSeed );
          if ( !dontSkip ) continue;

          q++;
          hTrack_Seed_Pt->Fill( seedPt );
          hTrack_Seed_Phi->Fill( seedPhi );
          hTrack_Seed_Eta->Fill( seedEta );
        }
      }

      if ( q > 1 ) std::cerr << "q is " << q << std::endl;
    }
  }

  hSeed_N->Fill( numSeeds );

} /// End of analyze()

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(ValidateL1Track);

