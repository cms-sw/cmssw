/****************************************************************************
*
* This is a part of CTPPS offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*   Nicola Minafra
*   Laurent Forthomme
*
****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/CTPPSDigi/interface/TotemFEDInfo.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"

#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"

#include <string>

#include "TRandom3.h"
const double SEC_PER_LUMI_SECTION = 23.31;
const double HPTDC_BIN_WIDTH = 25e-9/1024;

//----------------------------------------------------------------------------------------------------

 
class CTPPSDiamondDQMSource : public DQMEDAnalyzer
{
  public:
    CTPPSDiamondDQMSource( const edm::ParameterSet& );
    virtual ~CTPPSDiamondDQMSource();
  
  protected:
    void dqmBeginRun( const edm::Run&, const edm::EventSetup& ) override;
    void bookHistograms( DQMStore::IBooker&, const edm::Run&, const edm::EventSetup& ) override;
    void analyze( const edm::Event&, const edm::EventSetup& );
    void beginLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& );
    void endLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& );
    void endRun( const edm::Run&, const edm::EventSetup& );

  private:

    edm::EDGetTokenT< edm::DetSetVector<TotemVFATStatus> > tokenStatus_;
    edm::EDGetTokenT< edm::DetSetVector<TotemRPLocalTrack> > tokenLocalTrack_;
    edm::EDGetTokenT< edm::DetSetVector<CTPPSDiamondDigi> > tokenDigi_;
    edm::EDGetTokenT< edm::DetSetVector<CTPPSDiamondRecHit> > tokenDiamondHit_;
    edm::EDGetTokenT< edm::DetSetVector<CTPPSDiamondLocalTrack> > tokenDiamondTrack_;
    edm::EDGetTokenT< std::vector<TotemFEDInfo> > tokenFEDInfo_;

    bool excludeMultipleHits_;
    double minimumStripAngleForTomography_;
    unsigned int verbosity_;
    
    //Rnd generator used to "fake" pad width
    TRandom3 randomGen;

    /// plots related to the whole system
    struct GlobalPlots
    {
      MonitorElement *h_trackCorr_hor = NULL;

      GlobalPlots() {};
      GlobalPlots(DQMStore::IBooker &ibooker);
    };

    GlobalPlots globalPlot_;
    
    /// plots related to one Diamond detector package
    struct PotPlots
    {
      MonitorElement* activity_per_bx = NULL, *activity_per_bx_short = NULL;
      MonitorElement* activity_per_bx_plus1 = NULL, *activity_per_bx_short_plus1 = NULL;
      MonitorElement* activity_per_bx_minus1 = NULL, *activity_per_bx_short_minus1 = NULL;
      MonitorElement* activity_per_fedbx = NULL, *activity_per_fedbx_short = NULL;

      MonitorElement* hitDistribution2d = NULL;
      MonitorElement* hitDistribution2dOOT = NULL;
      MonitorElement* activePlanes = NULL;

      MonitorElement* trackDistribution = NULL;
      MonitorElement* trackDistributionOOT = NULL;

      MonitorElement* stripTomographyAllFar = NULL, *stripTomographyAllNear = NULL;
      MonitorElement* stripTomographyAllFar_plus1 = NULL, *stripTomographyAllNear_plus1 = NULL;
      MonitorElement* stripTomographyAllFar_minus1 = NULL, *stripTomographyAllNear_minus1 = NULL;

      MonitorElement* leadingEdgeCumulativePot = NULL, *timeOverThresholdCumulativePot = NULL, *leadingTrailingCorrelationPot = NULL;
      MonitorElement* leading_without_trailing_cumulative_pot = NULL;

      MonitorElement* ec_check = NULL;

      MonitorElement* error_flags_cumulative = NULL;
      
      MonitorElement* clock_Digi1_le = NULL;
      MonitorElement* clock_Digi1_te = NULL;
      MonitorElement* clock_Digi3_le = NULL;
      MonitorElement* clock_Digi3_te = NULL;

      PotPlots() {};
      PotPlots( DQMStore::IBooker& ibooker, unsigned int id );
    };

    std::map<unsigned int, PotPlots> potPlots_;
    int diff_tmp_582_, diff_tmp_583_;
    
    /// plots related to one Diamond plane
    struct PlanePlots
    {
      MonitorElement* digi_profile_cumulative = NULL;
      MonitorElement* hitProfile = NULL;
      MonitorElement* hit_multiplicity = NULL;
      MonitorElement* threshold_voltage = NULL;

      MonitorElement* stripTomography_far = NULL;
      MonitorElement* stripTomography_near = NULL;

      PlanePlots() {}
      PlanePlots( DQMStore::IBooker& ibooker, unsigned int id );
    };

    std::map<unsigned int, PlanePlots> planePlots_;
    
    /// plots related to one Diamond channel
    struct ChannelPlots
    {
      MonitorElement* error_flags = NULL;
      MonitorElement* leading_edge_cumulative = NULL;
      MonitorElement* time_over_threshold_cumulative = NULL;
      MonitorElement* leading_trailing_correlation = NULL;
      MonitorElement* leading_without_trailing = NULL;
      MonitorElement* efficiency = NULL;
      MonitorElement* stripTomography_far = NULL;
      MonitorElement* stripTomography_near = NULL;
      MonitorElement* hit_rate = NULL;
      MonitorElement* channel_ec_check = NULL;
      unsigned long hitsCounterPerLumisection;

      ChannelPlots() : hitsCounterPerLumisection(0) {}
      ChannelPlots( DQMStore::IBooker &ibooker, unsigned int id );
    };

    std::map<unsigned int, ChannelPlots> channelPlots_;
};

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDQMSource::GlobalPlots::GlobalPlots(DQMStore::IBooker &ibooker)
{
  ibooker.setCurrentFolder("CTPPS");

  h_trackCorr_hor = ibooker.book2D("track correlation all hor", "rp, all, hor", 6, -0.5, 5.5, 6, -0.5, 5.5);
  TH2F *hist = h_trackCorr_hor->getTH2F();
  TAxis *xa = hist->GetXaxis(), *ya = hist->GetYaxis();
  xa->SetBinLabel(6, "45, 210, near"); ya->SetBinLabel(1, "45, 210, near");
  xa->SetBinLabel(5, "45, 210, far"); ya->SetBinLabel(2, "45, 210, far");
  xa->SetBinLabel(4, "45, 220, cyl"); ya->SetBinLabel(3, "45, 220, cyl");
  xa->SetBinLabel(3, "56, 210, near"); ya->SetBinLabel(4, "56, 210, near");
  xa->SetBinLabel(2, "56, 210, far"); ya->SetBinLabel(5, "56, 210, far");
  xa->SetBinLabel(1, "56, 220, cyl"); ya->SetBinLabel(6, "56, 220, cyl");
}

//----------------------------------------------------------------------------------------------------


CTPPSDiamondDQMSource::PotPlots::PotPlots( DQMStore::IBooker& ibooker, unsigned int id )
{
  std::string path, title;
  CTPPSDiamondDetId( id ).rpName( path, CTPPSDiamondDetId::nPath );
  ibooker.setCurrentFolder( path );

  CTPPSDiamondDetId( id ).rpName( title, CTPPSDiamondDetId::nFull );

  activity_per_bx = ibooker.book1D( "activity per BX", title+" activity per BX;Event.BX", 4002, -1.5, 4000. + 0.5 );
  activity_per_bx_short = ibooker.book1D( "activity per BX (short)", title+" activity per BX (short);Event.BX", 102, -1.5, 100. + 0.5 );

  activity_per_bx_plus1 = ibooker.book1D( "activity per BX OOT +1", title+" activity per BX OOT +1;Event.BX", 4002, -1.5, 4000. + 0.5 );
  activity_per_bx_short_plus1 = ibooker.book1D( "activity per BX OOT +1 (short)", title+" activity per BX OOT +1 (short);Event.BX", 102, -1.5, 100. + 0.5 );

  activity_per_bx_minus1 = ibooker.book1D( "activity per BX OOT -1", title+" activity per BX OOT -1;Event.BX", 4002, -1.5, 4000. + 0.5 );
  activity_per_bx_short_minus1 = ibooker.book1D( "activity per BX OOT -1 (short)", title+" activity per BX OOT -1 (short);Event.BX", 102, -1.5, 100. + 0.5 );

  activity_per_fedbx = ibooker.book1D( "activity per FED BX", title+" activity per FED BX;Event.BX", 4002, -1.5, 4000. + 0.5 );
  activity_per_fedbx_short = ibooker.book1D( "activity per FED BX (short)", title+" activity per FED BX (short);Event.BX", 102, -1.5, 100. + 0.5 );

  hitDistribution2d = ibooker.book2D( "hits in planes", title+" hits in planes;plane number;x (mm)", 9, -0.5, 4, 190, -1, 18 );
  hitDistribution2dOOT= ibooker.book2D( "hits with OOT in planes", title+" hits with OOT in planes;plane number + 0.25 OOT;x (mm)", 17, -0.25, 4, 60, 0, 18 );
  activePlanes = ibooker.book1D( "active planes", title+" active planes;number of active planes", 6, -0.5, 5.5);

  trackDistribution = ibooker.book1D( "tracks", title+" tracks;x (mm)", 95, -1, 18 );
  trackDistributionOOT = ibooker.book2D( "tracks with OOT", title+" tracks with OOT;plane number;x (mm)", 9, -0.5, 4, 60, 0, 18 );

  stripTomographyAllFar = ibooker.book2D( "tomography all far", title+" tomography with strips far (all planes);x + 50*plane(mm);y (mm)", 200, 0, 200, 101, -50, 50 );
  stripTomographyAllNear = ibooker.book2D( "tomography all near", title+" tomography with strips near (all planes);x + 50*plane(mm);y (mm)", 200, 0, 200, 101, -50, 50 );

  stripTomographyAllFar_plus1 = ibooker.book2D( "tomography all far OOT +1", title+" tomography with strips far (all planes);x + 50*plane(mm);y (mm)", 200, 0, 200, 101, -50, 50 );
  stripTomographyAllNear_plus1 = ibooker.book2D( "tomography all near OOT +1", title+" tomography with strips near (all planes);x + 50*plane(mm);y (mm)", 200, 0, 200, 101, -50, 50 );

  stripTomographyAllFar_minus1 = ibooker.book2D( "tomography all far OOT -1", title+" tomography with strips far (all planes);x + 50*plane(mm);y (mm)", 200, 0, 200, 101, -50, 50 );
  stripTomographyAllNear_minus1 = ibooker.book2D( "tomography all near OOT -1", title+" tomography with strips near (all planes);x + 50*plane(mm);y (mm)", 200, 0, 200, 101, -50, 50 );  

  leadingEdgeCumulativePot = ibooker.book1D( "leading edge", title+" leading edge;leading edge (ns)", 201, -100, 100 );
  timeOverThresholdCumulativePot = ibooker.book1D( "time over threshold", title+" time over threshold;time over threshold (ns)", 201, -100, 100 );
  leadingTrailingCorrelationPot = ibooker.book2D( "leading trailing correlation", title+" leading trailing correlation;leading edge (ns);trailing edge (ns)", 201, -100, 100, 201, -100, 100 );
  
  leading_without_trailing_cumulative_pot = ibooker.book1D( "leading edges without trailing", title+" leading edges without trailing;leading edges without trailing", 4, 0.5, 4.5 );
  leading_without_trailing_cumulative_pot->getTH1F()->GetXaxis()->SetBinLabel( 1, "Nothing" );
  leading_without_trailing_cumulative_pot->getTH1F()->GetXaxis()->SetBinLabel(2, "Leading only");
  leading_without_trailing_cumulative_pot->getTH1F()->GetXaxis()->SetBinLabel(3, "Trailing only");
  leading_without_trailing_cumulative_pot->getTH1F()->GetXaxis()->SetBinLabel(4, "Both");

  ec_check = ibooker.book1D("optorxEC(8bit) - vfatEC", title+" EC Error;optorxEC-vfatEC",512,-256,256);

  error_flags_cumulative = ibooker.book1D( "HPTDC Errors", title+" HPTDC Errors", 16, -0.5, 16.5 );
  for ( unsigned short error_index=1; error_index<16; ++error_index ) 
    error_flags_cumulative->getTH1F()->GetXaxis()->SetBinLabel( error_index, HPTDCErrorFlags::getHPTDCErrorName( error_index-1 ).c_str() );
  error_flags_cumulative->getTH1F()->GetXaxis()->SetBinLabel( 16, "MH" );
  
  
  ibooker.setCurrentFolder( path+"/clock/" );
  clock_Digi1_le = ibooker.book1D( "clock1 leading edge", title+" clock1;leading edge (ns)", 201, -100, 100 );
  clock_Digi1_te = ibooker.book1D( "clock1 trailing edge", title+" clock1;trailing edge (ns)", 201, -100, 100 );
  clock_Digi3_le = ibooker.book1D( "clock3 leading edge", title+" clock3;leading edge (ns)", 201, -100, 100 );
  clock_Digi3_te = ibooker.book1D( "clock3 trailing edge", title+" clock3;trailing edge (ns)", 201, -100, 100 );
}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDQMSource::PlanePlots::PlanePlots( DQMStore::IBooker& ibooker, unsigned int id )
{
  std::string path, title;
  CTPPSDiamondDetId( id ).planeName( path, CTPPSDiamondDetId::nPath );
  ibooker.setCurrentFolder( path );

  CTPPSDiamondDetId( id ).planeName( title, CTPPSDiamondDetId::nFull );

  digi_profile_cumulative = ibooker.book1D( "digi profile", title+" digi profile; ch number", 12, -0.5, 11.5 );
  hitProfile = ibooker.book1D( "hit profile", title+" hit profile;x (mm)", 180, 0, 18 );
  hit_multiplicity = ibooker.book1D( "channels per plane", title+" channels per plane; ch per plane", 13, -0.5, 12.5 );

  threshold_voltage = ibooker.book2D( "threshold I2C", title+" threshold I2C; channel; value", 12, -0.5, 11.5, 513, 0, 512 );

  stripTomography_far = ibooker.book2D( "tomography far", title+" tomography with strips far;x + 50 OOT (mm);y (mm)", 50, 0, 50, 150, -50, 100 );
  stripTomography_near = ibooker.book2D( "tomography near", title+" tomography with strips near;x + 50 OOT (mm);y (mm)", 50, 0, 50, 150, -50, 100 );
}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDQMSource::ChannelPlots::ChannelPlots( DQMStore::IBooker& ibooker, unsigned int id ) : hitsCounterPerLumisection(0)
{
  std::string path, title;
  CTPPSDiamondDetId( id ).channelName( path, CTPPSDiamondDetId::nPath );
  ibooker.setCurrentFolder( path );

  CTPPSDiamondDetId( id ).channelName( title, CTPPSDiamondDetId::nFull );

  leading_without_trailing = ibooker.book1D( "Leading Edges Without Trailing", title+" leading edges without trailing", 4, 0.5, 4.5 );
  leading_without_trailing->getTH1F()->GetXaxis()->SetBinLabel( 1, "Nothing" );
  leading_without_trailing->getTH1F()->GetXaxis()->SetBinLabel( 2, "Leading only" );
  leading_without_trailing->getTH1F()->GetXaxis()->SetBinLabel( 3, "Trailer only" );
  leading_without_trailing->getTH1F()->GetXaxis()->SetBinLabel( 4, "Full" );

  error_flags = ibooker.book1D( "hptdc_Errors", title+" HPTDC Errors", 16, -0.5, 16.5 );
  for ( unsigned short error_index=1; error_index<16; ++error_index )
    error_flags->getTH1F()->GetXaxis()->SetBinLabel( error_index, HPTDCErrorFlags::getHPTDCErrorName( error_index-1 ).c_str() );
  error_flags->getTH1F()->GetXaxis()->SetBinLabel( 16, "MH" );

  leading_edge_cumulative = ibooker.book1D( "leading edge", title+" leading edge; leading edge (ns)", 1001, 0, 100e-9 );
  time_over_threshold_cumulative = ibooker.book1D( "time over threshold", title+" time over threshold;time over threshold (ns)", 2e3, -1e-6, 1e-6 );
  leading_trailing_correlation = ibooker.book2D( "leading trailing correlation", title+" leading trailing correlation;leading edge (ns);trailing edge (ns)", 3e2, 0, 60e-9, 3e2, 0, 60e-9 );

  channel_ec_check = ibooker.book1D("optorxEC(8bit) - vfatEC vs optorxEC", title+" EC Error;optorxEC-vfatEC",512,-256,256);

  stripTomography_far = ibooker.book2D( "tomography far", "tomography with strips far;x + 50 OOT (mm);y (mm)", 50, 0, 50, 150, -50, 100 );
  stripTomography_near = ibooker.book2D( "tomography near", "tomography with strips near;x + 50 OOT (mm);y (mm)", 50, 0, 50, 150, -50, 100 );
  
  hit_rate = ibooker.book1D("hit rate", title+"hit rate;rate (Hz)", 100, 0, 1000);
}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDQMSource::CTPPSDiamondDQMSource( const edm::ParameterSet& ps ) :
  tokenStatus_      ( consumes< edm::DetSetVector<TotemVFATStatus> >       ( ps.getParameter<edm::InputTag>( "tagStatus" ) ) ),
  tokenLocalTrack_  ( consumes< edm::DetSetVector<TotemRPLocalTrack> >     ( ps.getParameter<edm::InputTag>( "tagLocalTrack" ) ) ),
  tokenDigi_        ( consumes< edm::DetSetVector<CTPPSDiamondDigi> >      ( ps.getParameter<edm::InputTag>( "tagDigi" ) ) ),
  tokenDiamondHit_  ( consumes< edm::DetSetVector<CTPPSDiamondRecHit> >    ( ps.getParameter<edm::InputTag>( "tagDiamondRecHits" ) ) ),
  tokenDiamondTrack_( consumes< edm::DetSetVector<CTPPSDiamondLocalTrack> >( ps.getParameter<edm::InputTag>( "tagDiamondLocalTracks" ) ) ),
  tokenFEDInfo_     ( consumes< std::vector<TotemFEDInfo> >                ( ps.getParameter<edm::InputTag>( "tagFEDInfo" ) ) ),
  excludeMultipleHits_( ps.getParameter<bool>( "excludeMultipleHits" ) ),
  minimumStripAngleForTomography_( ps.getParameter<double>( "minimumStripAngleForTomography" ) ),
  verbosity_          ( ps.getUntrackedParameter<unsigned int>( "verbosity", 0 ) ),
  diff_tmp_582_( -500 ), diff_tmp_583_( -500 )
{}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDQMSource::~CTPPSDiamondDQMSource()
{}

//----------------------------------------------------------------------------------------------------

void
CTPPSDiamondDQMSource::dqmBeginRun( const edm::Run&, const edm::EventSetup& )
{}

//----------------------------------------------------------------------------------------------------

void
CTPPSDiamondDQMSource::bookHistograms( DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup& )
{
  ibooker.cd();
  ibooker.setCurrentFolder( "CTPPS" );
  
  globalPlot_= GlobalPlots( ibooker );

  // loop over arms
  for ( unsigned short arm = 0; arm < 2; ++arm ) {
    // loop over stations
    for ( unsigned short st = 1; st < 2; st += 2 ) {
      // loop over RPs
      for ( unsigned short rp = 6; rp < 7; ++rp ) {
        const CTPPSDiamondDetId rpId( arm, st, rp );
        potPlots_[rpId] = PotPlots( ibooker, rpId );

        // loop over planes
        for ( unsigned short pl = 0; pl < 4; ++pl ) {
          const CTPPSDiamondDetId plId( arm, st, rp, pl );
          planePlots_[plId] = PlanePlots( ibooker, plId);
 
          // loop over channels
          for ( unsigned short ch = 0; ch < 12; ++ch ) {
            const CTPPSDiamondDetId chId( arm, st, rp, pl, ch );
            channelPlots_[chId] = ChannelPlots( ibooker, chId );
	  }  
        }
      }
    }
  }
}


//----------------------------------------------------------------------------------------------------

void
CTPPSDiamondDQMSource::beginLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& ) 
{
  for (auto& plot : channelPlots_) {
    if ( plot.second.hitsCounterPerLumisection != 0 ) {
      plot.second.hit_rate->Fill( (double) plot.second.hitsCounterPerLumisection / SEC_PER_LUMI_SECTION );
    }
    plot.second.hitsCounterPerLumisection = 0;
  }
}

//----------------------------------------------------------------------------------------------------

void
CTPPSDiamondDQMSource::analyze( const edm::Event& event, const edm::EventSetup& eventSetup )
{
  // get event data
  edm::Handle< edm::DetSetVector<TotemVFATStatus> > status;
  event.getByToken( tokenStatus_, status );

  edm::Handle< edm::DetSetVector<TotemRPLocalTrack> > StripTracks;
  event.getByToken( tokenLocalTrack_, StripTracks );

  edm::Handle< edm::DetSetVector<CTPPSDiamondDigi> > digi;
  event.getByToken( tokenDigi_, digi );

  edm::Handle< std::vector<TotemFEDInfo> > fedInfo;
  event.getByToken( tokenFEDInfo_, fedInfo );

  edm::Handle< edm::DetSetVector<CTPPSDiamondRecHit> > hits;
  event.getByToken( tokenDiamondHit_, hits );

  edm::Handle< edm::DetSetVector<CTPPSDiamondLocalTrack> > localTracks;
  event.getByToken( tokenDiamondTrack_, localTracks );

  // check validity
  bool valid = true;
  valid &= status.isValid();
  valid &= digi.isValid();
  valid &= fedInfo.isValid();

  if ( !valid ) {
    if ( verbosity_ ) {
      edm::LogProblem("CTPPSDiamondDQMSource")
        << "ERROR in TotemDQMModuleRP::analyze > some of the required inputs are not valid. Skipping this event.\n"
        << "    status.isValid = " << status.isValid() << "\n"
        << "    digi.isValid = " << digi.isValid() << "\n"
        << "    fedInfo.isValid = " << fedInfo.isValid();
    }

    return;
  }

  //------------------------------
  // RP Plots
  //------------------------------  
  
  //------------------------------
  // Correlation Plots
  //------------------------------
  
  for ( auto &ds1 : *StripTracks )
  {
    for ( auto &tr1 : ds1 )
    {
      if ( ! tr1.isValid() )  continue;
  
      CTPPSDetId rpId1( ds1.detId() );
      unsigned int arm1 = rpId1.arm();
      unsigned int stNum1 = rpId1.station();
      unsigned int rpNum1 = rpId1.rp();
      if (stNum1 != 0 || ( rpNum1 != 2 && rpNum1 != 3 ) )  continue;
      unsigned int idx1 = arm1*3 + rpNum1-2;

      for ( auto &ds2 : *StripTracks )
      {
        for ( auto &tr2 : ds2 )
        {
          if ( ! tr2.isValid() )  continue;
        
          CTPPSDetId rpId2(ds2.detId());
          unsigned int arm2 = rpId2.arm();
          unsigned int stNum2 = rpId2.station();
          unsigned int rpNum2 = rpId2.rp();
          if (stNum2 != 0 || ( rpNum2 != 2 && rpNum2 != 3 ) )  continue;
          unsigned int idx2 = arm2*3 + rpNum2-2;
  
          if ( idx1 >= idx2 ) globalPlot_.h_trackCorr_hor->Fill( 5-idx1, idx2 );        //Strips-strips
        }
      }
      for ( auto &ds2 : *localTracks )
      {
        for ( auto &tr2 : ds2 )
        {
          if ( ! tr2.isValid() ) continue;
          if ( tr2.getOOTIndex() != 1 ) continue;
          if ( excludeMultipleHits_ && tr2.getMultipleHits() > 0 ) continue;
          
          CTPPSDetId diamId2( ds2.detId() );
          unsigned int arm2 = diamId2.arm();
          if ( idx1 >= arm2*3+2 ) globalPlot_.h_trackCorr_hor->Fill( 5-idx1, arm2*3+2 );         //Strips-diamonds
          else globalPlot_.h_trackCorr_hor->Fill( 5-(arm2*3+2 ),idx1 );         //Strips-diamonds
        }
      }
    }
  }
  
  for ( auto &ds1 : *localTracks )
  {
    for ( auto &tr1 : ds1 )
    {
      if ( ! tr1.isValid() ) continue;
      if ( excludeMultipleHits_ && tr1.getMultipleHits() > 0 ) continue;
      if ( tr1.getOOTIndex() != 1 ) continue;
      
      CTPPSDetId diamId1( ds1.detId() );
      unsigned int arm1 = diamId1.arm();
      
      globalPlot_.h_trackCorr_hor->Fill( 5-(arm1*3+2), arm1*3+2 );      //diamonds-diamonds
      
      for ( auto &ds2 : *localTracks )
      {
        for ( auto &tr2 : ds2 )
        {
          if ( ! tr2.isValid() ) continue;
          if ( excludeMultipleHits_ && tr2.getMultipleHits() > 0 ) continue;
          if ( tr2.getOOTIndex() != 1 ) continue;
          
          CTPPSDetId diamId2( ds2.detId() );
          unsigned int arm2 = diamId2.arm();
          if ( arm1 > arm2 ) globalPlot_.h_trackCorr_hor->Fill( 5-(arm1*3+2), arm2*3+2 );      //diamonds-diamonds
        }
      }
    }
  }
  
  
  // Using CTPPSDiamondDigi
  for ( edm::DetSetVector<CTPPSDiamondDigi>::const_iterator it = digi->begin(); it != digi->end(); ++it ) {
    const CTPPSDiamondDetId detId( it->detId() );

    for ( edm::DetSet<CTPPSDiamondDigi>::const_iterator dit = it->begin(); dit != it->end(); ++dit ) {
      CTPPSDiamondDetId detId_pot( it->detId() );
      detId_pot.setPlane( 0 );
      detId_pot.setChannel( 0 );
      if ( potPlots_.find( detId_pot ) == potPlots_.end() ) continue;
      //Leading without trailing investigation
      if      ( dit->getLeadingEdge() == 0 && dit->getTrailingEdge() == 0 ) potPlots_[detId_pot].leading_without_trailing_cumulative_pot->Fill( 1 );
      else if ( dit->getLeadingEdge() != 0 && dit->getTrailingEdge() == 0 ) potPlots_[detId_pot].leading_without_trailing_cumulative_pot->Fill( 2 );
      else if ( dit->getLeadingEdge() == 0 && dit->getTrailingEdge() != 0 ) potPlots_[detId_pot].leading_without_trailing_cumulative_pot->Fill( 3 );
      else if ( dit->getLeadingEdge() != 0 && dit->getTrailingEdge() != 0 ) potPlots_[detId_pot].leading_without_trailing_cumulative_pot->Fill( 4 );

      if ( dit->getLeadingEdge() != 0 ) {
        // FED BX monitoring (for MINIDAQ)
        for ( auto& fit : *fedInfo ) {
          if ( ( detId.arm()==1 && fit.getFEDId()==582 ) || ( detId.arm() == 0 && fit.getFEDId() == 583 ) ) {
            potPlots_[detId_pot].activity_per_fedbx->Fill( fit.getBX() );
            potPlots_[detId_pot].activity_per_fedbx_short->Fill( fit.getBX() );
          }
        }
      }

      // HPTDC Errors
      const HPTDCErrorFlags hptdcErrors = dit->getHPTDCErrorFlags();
      for ( unsigned short hptdcErrorIndex = 1; hptdcErrorIndex < 16; ++hptdcErrorIndex )
        if ( hptdcErrors.getErrorId( hptdcErrorIndex-1 ) ) potPlots_[detId_pot].error_flags_cumulative->Fill( hptdcErrorIndex );
      if ( dit->getMultipleHit() ) potPlots_[detId_pot].error_flags_cumulative->Fill( 16 );
    }
  }

  // EC Errors
  for ( edm::DetSetVector<TotemVFATStatus>::const_iterator it = status->begin(); it != status->end(); ++it ) {
    const CTPPSDiamondDetId detId(it->detId());
    CTPPSDiamondDetId detId_pot( it->detId() );
    detId_pot.setPlane( 0 );
    detId_pot.setChannel( 0 );
    for ( edm::DetSet<TotemVFATStatus>::const_iterator dit = it->begin(); dit != it->end(); ++dit ) {
      if ( potPlots_.find(detId_pot) != potPlots_.end() ) {
	//Check Event Number
        for ( auto& optorx : *fedInfo ) {
          if ( detId.arm() == 1 && optorx.getFEDId() == 582 ) {
            potPlots_[detId_pot].ec_check->Fill((int)((optorx.getLV1()& 0xFF)-((unsigned int) dit->getEC() & 0xFF)) & 0xFF);
            if ( ( static_cast<int>( ( optorx.getLV1() & 0xFF )-dit->getEC() ) != diff_tmp_582_ ) && ( static_cast<uint8_t>( ( optorx.getLV1() & 0xFF )-dit->getEC() ) < 128 ) )
              diff_tmp_582_ = static_cast<int>( optorx.getLV1() & 0xFF )-( static_cast<unsigned int>( dit->getEC() ) & 0xFF );
            if ( diff_tmp_582_ != 1 && diff_tmp_582_ != -500 && diff_tmp_582_ < 128 && diff_tmp_582_ > -128 )
              if (verbosity_) edm::LogProblem("CTPPSDiamondDQMSource")  << "FED 852: ECError at EV: 0x"<< std::hex << optorx.getLV1()
                                                                        << "\t\tVFAT EC: 0x"<< static_cast<unsigned int>( dit->getEC() )
                                                                        << "\twith ID: " << std::dec << detId
                                                                        << "\tdiff: " <<  diff_tmp_582_;
          }
          else if ( detId.arm() == 0 && optorx.getFEDId()==583 ) {
            potPlots_[detId_pot].ec_check->Fill((int)((optorx.getLV1()& 0xFF)-dit->getEC()) & 0xFF);
            if ( ( static_cast<int>( ( optorx.getLV1() & 0xFF )-dit->getEC() ) != diff_tmp_583_ ) && ( static_cast<uint8_t>( ( optorx.getLV1() & 0xFF )-dit->getEC() ) < 128 ) )
              diff_tmp_583_ = static_cast<int>( optorx.getLV1() & 0xFF )-( static_cast<unsigned int>( dit->getEC() ) & 0xFF );
            if ( diff_tmp_583_ != 1 && diff_tmp_583_ != -500 && diff_tmp_583_ < 128 && diff_tmp_583_ > -128 )
              if (verbosity_) edm::LogProblem("CTPPSDiamondDQMSource")  << "FED 853: ECError at EV: 0x"<< std::hex << optorx.getLV1()
                                                                        << "\t\tVFAT EC: 0x"<< static_cast<unsigned int>( dit->getEC() )
                                                                        << "\twith ID: " << std::dec << detId
                                                                        << "\tdiff: " <<  diff_tmp_583_;
          }
        }
      }
    }
  }
  
  //Using CTPPSDiamondRecHit
  std::map<unsigned int, std::set<unsigned int> > planes;

  for ( edm::DetSetVector<CTPPSDiamondRecHit>::const_iterator it=hits->begin(); it != hits->end(); ++it ) {
    CTPPSDiamondDetId detId_pot(it->detId());
    detId_pot.setPlane( 0 );
    detId_pot.setChannel( 0 );
    const CTPPSDiamondDetId detId( it->detId() );

    for ( edm::DetSet<CTPPSDiamondRecHit>::const_iterator hitIt=it->begin(); hitIt != it->end(); ++hitIt ) {
      if ( excludeMultipleHits_ && hitIt->getMultipleHits() > 0 ) continue;
      planes[detId_pot].insert( detId.plane() );
      if ( potPlots_.find( detId_pot ) != potPlots_.end() ) {
	const double rnd = ( 2. * randomGen.Rndm() ) - 1.;

        potPlots_[detId_pot].hitDistribution2d->Fill( detId.plane(), hitIt->getX() + 0.5*rnd*hitIt->getXWidth() );
        potPlots_[detId_pot].hitDistribution2dOOT->Fill( detId.plane() + 0.25 * hitIt->getOOTIndex(), hitIt->getX() + 0.5*rnd*hitIt->getXWidth() );

        potPlots_[detId_pot].leadingEdgeCumulativePot->Fill( hitIt->getT() );
        potPlots_[detId_pot].timeOverThresholdCumulativePot->Fill( hitIt->getToT() );
        potPlots_[detId_pot].leadingTrailingCorrelationPot->Fill( hitIt->getT(), hitIt->getT() + hitIt->getToT() );

        switch ( hitIt->getOOTIndex() ) {
          case 0: {
            potPlots_[detId_pot].activity_per_bx_minus1->Fill( event.bunchCrossing() );
            potPlots_[detId_pot].activity_per_bx_short_minus1->Fill( event.bunchCrossing() );
          } break;
	  case 1: {
            potPlots_[detId_pot].activity_per_bx->Fill( event.bunchCrossing() );
            potPlots_[detId_pot].activity_per_bx_short->Fill( event.bunchCrossing() );
          } break;
	  case 2: {
            potPlots_[detId_pot].activity_per_bx_plus1->Fill( event.bunchCrossing() );
            potPlots_[detId_pot].activity_per_bx_short_plus1->Fill( event.bunchCrossing() );
          } break;
	}
      }
      
    }
  }
  
  for ( auto& plt : potPlots_ ) {
    plt.second.activePlanes->Fill( planes[plt.first].size() );
  }
  
  // Using CTPPSDiamondLocalTrack
  for ( auto &it : *localTracks ) {
    CTPPSDiamondDetId detId_pot( it.detId() );
    detId_pot.setPlane( 0 );
    detId_pot.setChannel( 0 );
    const CTPPSDiamondDetId detId(it.detId());

    for ( auto &track : it ) {
      if ( ! track.isValid() ) continue;
      if ( excludeMultipleHits_ && track.getMultipleHits() > 0 ) continue;
      if ( potPlots_.find( detId_pot ) != potPlots_.end() ) {
        const double rnd = ( 2. * randomGen.Rndm() ) - 1.;
        if ( track.getOOTIndex() == 1 ) potPlots_[detId_pot].trackDistribution->Fill( track.getX0()+rnd*track.getX0Sigma() );
	potPlots_[detId_pot].trackDistributionOOT->Fill( track.getOOTIndex(), track.getX0()+rnd*track.getX0Sigma() );
      }
    }
  }
  
  // Tomography of diamonds using strips
  for ( edm::DetSetVector<CTPPSDiamondRecHit>::const_iterator it = hits->begin(); it != hits->end(); ++it ) {
    CTPPSDiamondDetId detId_pot( it->detId() );
    detId_pot.setPlane( 0 );
    detId_pot.setChannel( 0 );
    const CTPPSDiamondDetId detId( it->detId() );

    for ( edm::DetSet<CTPPSDiamondRecHit>::const_iterator hitIt = it->begin(); hitIt != it->end(); ++hitIt ) {
      if ( excludeMultipleHits_ && hitIt->getMultipleHits() > 0 ) continue;
      if ( !StripTracks.isValid() ) continue;
      if ( potPlots_.find( detId_pot ) == potPlots_.end() ) continue;

      for ( auto& ds2 : *StripTracks ) {
        for ( auto& striplt : ds2 ) {
          if ( !striplt.isValid() ) continue;
          if ( striplt.getTx() > minimumStripAngleForTomography_ || striplt.getTy() > minimumStripAngleForTomography_) continue; 
          const unsigned int arm = ( striplt.getZ0() > 0 ) ? 1 : 0;
          if ( arm != detId_pot.arm() ) continue;
          if ( fabs( striplt.getZ0() ) > 207e3 ) {
            switch ( hitIt->getOOTIndex() ) {
              case 0: {
                potPlots_[detId_pot].stripTomographyAllFar_minus1->Fill( striplt.getX0() + 50*detId.plane(), striplt.getY0() );
              } break;
              case 1: {
                potPlots_[detId_pot].stripTomographyAllFar->Fill( striplt.getX0() + 50*detId.plane(), striplt.getY0() );
              } break;
              case 2: {
                potPlots_[detId_pot].stripTomographyAllFar_plus1->Fill( striplt.getX0() + 50*detId.plane(), striplt.getY0() );
              } break;
            }
          }
          else {
            switch ( hitIt->getOOTIndex() ) {
              case 0: {
                potPlots_[detId_pot].stripTomographyAllNear_minus1->Fill( striplt.getX0() + 50*detId.plane(), striplt.getY0() );
              } break;
              case 1: {
                potPlots_[detId_pot].stripTomographyAllNear->Fill( striplt.getX0() + 50*detId.plane(), striplt.getY0() );
              } break;
              case 2: {
                potPlots_[detId_pot].stripTomographyAllNear_plus1->Fill( striplt.getX0() + 50*detId.plane(), striplt.getY0() );
              } break;
            }
          }
        }
      }
    }
  }
  
  
  //------------------------------
  // Clock Plots
  //------------------------------
  
  for ( edm::DetSetVector<CTPPSDiamondDigi>::const_iterator it = digi->begin(); it != digi->end(); ++it ) {
    const CTPPSDiamondDetId detId( it->detId() );
    if ( detId.channel() == 30 ) {
      CTPPSDiamondDetId detId_pot( it->detId() );
      detId_pot.setPlane( 0 );
      detId_pot.setChannel( 0 );
      for ( edm::DetSet<CTPPSDiamondDigi>::const_iterator dit = it->begin(); dit != it->end(); ++dit ) {
        if ( detId.plane() == 1 ) {
          potPlots_[detId_pot].clock_Digi1_le->Fill( dit->getLeadingEdge() );
          potPlots_[detId_pot].clock_Digi1_te->Fill( dit->getTrailingEdge() );
        }
        if ( detId.plane() == 3 ) {
          potPlots_[detId_pot].clock_Digi3_le->Fill( dit->getLeadingEdge() );
          potPlots_[detId_pot].clock_Digi3_te->Fill( dit->getTrailingEdge() );
        }     
      }
    }
  }
  

  //------------------------------
  // Plane Plots
  //------------------------------

  // Using CTPPSDiamondDigi
  std::map<unsigned int, unsigned int> channelsPerPlane;
  for ( edm::DetSetVector<CTPPSDiamondDigi>::const_iterator it = digi->begin(); it != digi->end(); ++it ) {
    const CTPPSDiamondDetId detId( it->detId() );
    for ( edm::DetSet<CTPPSDiamondDigi>::const_iterator dit = it->begin(); dit != it->end(); ++dit ) {
      CTPPSDiamondDetId detId_plane( it->detId() );
      detId_plane.setChannel( 0 );
      if ( planePlots_.find( detId_plane ) != planePlots_.end() ) {
        planePlots_[detId_plane].threshold_voltage->Fill( detId.channel(), dit->getThresholdVoltage() );

        if ( dit->getLeadingEdge() != 0 ) {
          planePlots_[detId_plane].digi_profile_cumulative->Fill( detId.channel() );
          if ( channelsPerPlane.find(detId_plane) !=  channelsPerPlane.end() ) channelsPerPlane[detId_plane]++;
          else channelsPerPlane[detId_plane]=0;
        }
      }     
    }
  }

  for ( auto& plt : channelsPerPlane ) {
    planePlots_[plt.first].hit_multiplicity->Fill( plt.second );
  }
  
  // Using CTPPSDiamondRecHit
  for ( edm::DetSetVector<CTPPSDiamondRecHit>::const_iterator it=hits->begin(); it != hits->end(); ++it ) {
    CTPPSDiamondDetId detId_plane( it->detId() );
    detId_plane.setChannel( 0 );
    for ( edm::DetSet<CTPPSDiamondRecHit>::const_iterator hitIt=it->begin(); hitIt != it->end(); ++hitIt ) {
      if ( excludeMultipleHits_ && hitIt->getMultipleHits() > 0 ) continue;
      if ( planePlots_.find( detId_plane ) != planePlots_.end() ) {
	double rnd = (2.*randomGen.Rndm()) - 1.;
	if (hitIt->getOOTIndex() == 1) planePlots_[detId_plane].hitProfile->Fill( hitIt->getX()+.5*rnd*hitIt->getXWidth() );
      }
    }
  }

  // Tomography of diamonds using strips
  for ( edm::DetSetVector<CTPPSDiamondRecHit>::const_iterator it=hits->begin(); it != hits->end(); ++it ) {
    CTPPSDiamondDetId detId_plane( it->detId() );
    detId_plane.setChannel( 0 );
    for ( edm::DetSet<CTPPSDiamondRecHit>::const_iterator hitIt=it->begin(); hitIt != it->end(); ++hitIt ) {
      if ( excludeMultipleHits_ && hitIt->getMultipleHits() > 0 ) continue;
      if ( StripTracks.isValid() ) {
        if (planePlots_.find(detId_plane) != planePlots_.end()) {
          for (auto &ds2 : *StripTracks) {
            for (auto &striplt : ds2) {
              if (! striplt.isValid()) continue;
              if (striplt.getTx() > minimumStripAngleForTomography_ || striplt.getTy() > minimumStripAngleForTomography_) continue;
              const unsigned int arm = striplt.getZ0()>0 ? 1 : 0;
              if ( arm != detId_plane.arm() ) continue;
              if ( abs( striplt.getZ0() ) > 207e3 ) {
                planePlots_[detId_plane].stripTomography_far->Fill( striplt.getX0(), striplt.getY0() + 50*( hitIt->getOOTIndex()-1 ) );
              }
              else {
                planePlots_[detId_plane].stripTomography_near->Fill( striplt.getX0(), striplt.getY0() + 50*( hitIt->getOOTIndex()-1 ) );
              }
            }
          }
        }
      }
    }
  }

  //------------------------------
  // Channel Plots
  //------------------------------

  //Check Event Number
  for ( edm::DetSetVector<TotemVFATStatus>::const_iterator it = status->begin(); it != status->end(); ++it ) {
    const CTPPSDiamondDetId detId(it->detId());
    for ( edm::DetSet<TotemVFATStatus>::const_iterator dit = it->begin(); dit != it->end(); ++dit ) {
      if ( channelPlots_.find(detId) != channelPlots_.end() ) {
        for ( auto& optorx : *fedInfo ) {
          if ( ( detId.arm() == 1 && optorx.getFEDId() == 582 ) || ( detId.arm() == 0 && optorx.getFEDId()==583 ) ) {
            channelPlots_[detId].channel_ec_check->Fill((int)((optorx.getLV1()& 0xFF)-((unsigned int) dit->getEC() & 0xFF)) & 0xFF);
          }
        }
      }
    }
  }

  // digi profile cumulative
  for ( edm::DetSetVector<CTPPSDiamondDigi>::const_iterator it = digi->begin(); it != digi->end(); ++it ) {
    const CTPPSDiamondDetId detId( it->detId() );
    for ( edm::DetSet<CTPPSDiamondDigi>::const_iterator dit = it->begin(); dit != it->end(); ++dit ) {
      if ( channelPlots_.find( detId ) != channelPlots_.end() ) {
        if ( dit->getLeadingEdge() != 0 ) {
          channelPlots_[detId].leading_edge_cumulative->Fill(dit->getLeadingEdge() * HPTDC_BIN_WIDTH);
          if (dit->getTrailingEdge() != 0) {
            channelPlots_[detId].time_over_threshold_cumulative->Fill( HPTDC_BIN_WIDTH * ( dit->getLeadingEdge() - dit->getTrailingEdge() ) );
            channelPlots_[detId].leading_trailing_correlation->Fill( dit->getTrailingEdge() * HPTDC_BIN_WIDTH, dit->getLeadingEdge() * HPTDC_BIN_WIDTH );    
            ++(channelPlots_[detId].hitsCounterPerLumisection);
          }
        }

        // HPTDC Errors
        const HPTDCErrorFlags hptdcErrors = dit->getHPTDCErrorFlags();
        for ( unsigned short hptdcErrorIndex = 1; hptdcErrorIndex < 16; ++hptdcErrorIndex )
          if ( hptdcErrors.getErrorId( hptdcErrorIndex-1 ) ) channelPlots_[detId].error_flags->Fill( hptdcErrorIndex );
        if ( dit->getMultipleHit() ) channelPlots_[detId].error_flags->Fill( 16 );

        // Check dropped trailing edges
        if      ( dit->getLeadingEdge() == 0 && dit->getTrailingEdge() == 0 ) channelPlots_[detId].leading_without_trailing->Fill( 1 );
        else if ( dit->getLeadingEdge() != 0 && dit->getTrailingEdge() == 0 ) channelPlots_[detId].leading_without_trailing->Fill( 2 );
        else if ( dit->getLeadingEdge() == 0 && dit->getTrailingEdge() != 0 ) channelPlots_[detId].leading_without_trailing->Fill( 3 );
        else if ( dit->getLeadingEdge() != 0 && dit->getTrailingEdge() != 0 ) channelPlots_[detId].leading_without_trailing->Fill( 4 );
      }
    }
  }
  
  // Tomography of diamonds using strips
  for ( edm::DetSetVector<CTPPSDiamondRecHit>::const_iterator it=hits->begin(); it != hits->end(); ++it ) {
    CTPPSDiamondDetId detId( it->detId() );
    for ( edm::DetSet<CTPPSDiamondRecHit>::const_iterator hitIt=it->begin(); hitIt != it->end(); ++hitIt ) {
      if ( excludeMultipleHits_ && hitIt->getMultipleHits() > 0 ) continue;
      if ( StripTracks.isValid() ) {
        if (channelPlots_.find(detId) != channelPlots_.end()) {
          for (auto &ds2 : *StripTracks) {
            for (auto &striplt : ds2) {
              if (! striplt.isValid()) continue;
              if (striplt.getTx() > minimumStripAngleForTomography_ || striplt.getTy() > minimumStripAngleForTomography_) continue;
              const unsigned int arm = striplt.getZ0()>0 ? 1 : 0;
              if ( arm != detId.arm() ) continue;
              if ( abs( striplt.getZ0() ) > 207e3 ) {
                channelPlots_[detId].stripTomography_far->Fill( striplt.getX0(), striplt.getY0() + 50*( hitIt->getOOTIndex()-1 ) );
              }
              else {
                channelPlots_[detId].stripTomography_near->Fill( striplt.getX0(), striplt.getY0() + 50*( hitIt->getOOTIndex()-1 ) );
              }
            }
          }
        }
      }
    }
  }
  
}

//----------------------------------------------------------------------------------------------------

void
CTPPSDiamondDQMSource::endLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& ) 
{}

//----------------------------------------------------------------------------------------------------

void
CTPPSDiamondDQMSource::endRun( const edm::Run&, const edm::EventSetup& )
{}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE( CTPPSDiamondDQMSource );
