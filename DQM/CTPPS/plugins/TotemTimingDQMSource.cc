/****************************************************************************
*
* This is a part of CTPPS offline software.
* Authors:
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
#include "FWCore/Framework/interface/Run.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/CTPPSDigi/interface/TotemFEDInfo.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"

#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class TotemTimingDQMSource : public DQMEDAnalyzer
{
  public:
    TotemTimingDQMSource( const edm::ParameterSet& );
    ~TotemTimingDQMSource() override;

  protected:
    void dqmBeginRun( const edm::Run&, const edm::EventSetup& ) override;
    void bookHistograms( DQMStore::IBooker&, const edm::Run&, const edm::EventSetup& ) override;
    void analyze( const edm::Event&, const edm::EventSetup& ) override;
    void beginLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& ) override;
    void endLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& ) override;
    void endRun( const edm::Run&, const edm::EventSetup& ) override;

    uint16_t timestampAConverter( const uint16_t& binary) const;

  private:
    // Constants
    static const double SEC_PER_LUMI_SECTION;                   // Number of seconds per lumisection: used to compute hit rates in Hz
    static const double LHC_CLOCK_PERIOD_NS;
    static const double DQM_FRACTION_OF_EVENTS;                 // approximate fraction of events sent to DQM stream
    static const double HIT_RATE_FACTOR;                        // factor to have real rate in Hz
    static const double DISPLAY_RESOLUTION_FOR_HITS_MM;         // Bin width of histograms showing hits and tracks (in mm)
    static const double INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
    static const double SAMPIC_SAMPLING_PERIOD_NS;                        // ns per HPTDC bin
    static const double SAMPIC_MAX_NUMBER_OF_SAMPLES;
    static const double SAMPIC_ADC_V;
    static const int CTPPS_NUM_OF_ARMS;
    static const int TOTEM_TIMING_STATION_ID;
    static const int TOTEM_STATION_210;
    static const int TOTEM_STATION_220;
    static const int TOTEM_TIMING_MIN_RP_ID;
    static const int TOTEM_TIMING_MAX_RP_ID;
    static const int TOTEM_STRIP_MIN_RP_ID;
    static const int TOTEM_STRIP_MAX_RP_ID;
    static const int CTPPS_NEAR_RP_ID;
    static const int CTPPS_FAR_RP_ID;
    static const int TOTEM_TIMING_NUM_OF_PLANES;
    static const int TOTEM_TIMING_NUM_OF_CHANNELS;
    static const int TOTEM_TIMING_FED_ID_45;
    static const int TOTEM_TIMING_FED_ID_56;

    edm::EDGetTokenT< edm::DetSetVector<TotemRPLocalTrack> > tokenLocalTrack_;
    edm::EDGetTokenT< edm::DetSetVector<TotemTimingDigi> > tokenDigi_;
    edm::EDGetTokenT< edm::DetSetVector<CTPPSDiamondRecHit> > tokenDiamondHit_;
    edm::EDGetTokenT< edm::DetSetVector<CTPPSDiamondLocalTrack> > tokenDiamondTrack_;
    edm::EDGetTokenT< std::vector<TotemFEDInfo> > tokenFEDInfo_;

    double minimumStripAngleForTomography_;
    double maximumStripAngleForTomography_;
    unsigned int samplesForNoise_;
    unsigned int verbosity_;
    edm::TimeValue_t timeOfPreviousEvent_;

    /// plots related to the whole system
    struct GlobalPlots
    {
      MonitorElement* h_trackCorr_hor = nullptr;

      GlobalPlots() {}
      GlobalPlots( DQMStore::IBooker& ibooker );
    };

    GlobalPlots globalPlot_;

    /// plots related to one Diamond detector package
    struct PotPlots
    {
      MonitorElement* activityPerBX = nullptr;

      MonitorElement* tirggerCellTime = nullptr;

      MonitorElement* dataSamplesRaw = nullptr;

      MonitorElement* hitDistribution2d = nullptr;
      MonitorElement* hitDistribution2d_lumisection = nullptr;

      MonitorElement* digiDistribution = nullptr;
      MonitorElement* noiseRMS = nullptr;
      MonitorElement* baseline = nullptr;
      MonitorElement* meanAmplitude = nullptr;

      MonitorElement* digiSent = nullptr;
      MonitorElement* digiAll = nullptr;
      MonitorElement* digiSentPercentage = nullptr;

      MonitorElement* hitRate = nullptr;

      MonitorElement* activePlanes = nullptr;

      MonitorElement* trackDistribution = nullptr;

      MonitorElement* stripTomography210 = nullptr;
      MonitorElement* stripTomography220 = nullptr;

      MonitorElement* leadingEdge = nullptr;
      MonitorElement* amplitude = nullptr;

      std::set<unsigned int> planesWithHits;

      PotPlots() {};
      PotPlots( DQMStore::IBooker& ibooker, unsigned int id );
    };

    std::unordered_map<unsigned int, PotPlots> potPlots_;

    /// plots related to one Diamond plane
    struct PlanePlots
    {
      MonitorElement* digiDistribution = nullptr;

      MonitorElement* hitProfile = nullptr;
      MonitorElement* hitMultiplicity = nullptr;

      PlanePlots() {}
      PlanePlots( DQMStore::IBooker& ibooker, unsigned int id );
    };

    std::unordered_map<unsigned int, PlanePlots> planePlots_;

    /// plots related to one Diamond channel
    struct ChannelPlots
    {
      MonitorElement* activityPerBX = nullptr;

      MonitorElement* tirggerCellTime = nullptr;

      MonitorElement* dataSamplesRaw = nullptr;

      MonitorElement* timestampA = nullptr;
      MonitorElement* timestampB = nullptr;
      MonitorElement* timestampDiff = nullptr;

      MonitorElement* leadingEdge = nullptr;
      MonitorElement* amplitude = nullptr;
      MonitorElement* noiseSamples = nullptr;

      MonitorElement* hitTime = nullptr;
      MonitorElement* hitRate = nullptr;
      unsigned long hitsCounterPerLumisection;

      ChannelPlots() : hitsCounterPerLumisection( 0 ) {}
      ChannelPlots( DQMStore::IBooker &ibooker, unsigned int id );
    };

    std::unordered_map<unsigned int, ChannelPlots> channelPlots_;
};

//----------------------------------------------------------------------------------------------------

// Values for all constants
const double    TotemTimingDQMSource::SEC_PER_LUMI_SECTION = 23.31;
const double    TotemTimingDQMSource::LHC_CLOCK_PERIOD_NS = 24.95;
const double    TotemTimingDQMSource::DQM_FRACTION_OF_EVENTS = 1.;
const double    TotemTimingDQMSource::HIT_RATE_FACTOR = DQM_FRACTION_OF_EVENTS/SEC_PER_LUMI_SECTION;
const double    TotemTimingDQMSource::DISPLAY_RESOLUTION_FOR_HITS_MM = 0.1;
const double    TotemTimingDQMSource::INV_DISPLAY_RESOLUTION_FOR_HITS_MM = 1./DISPLAY_RESOLUTION_FOR_HITS_MM;
const double    TotemTimingDQMSource::SAMPIC_SAMPLING_PERIOD_NS = 1./7.8e9;
const double    TotemTimingDQMSource::SAMPIC_MAX_NUMBER_OF_SAMPLES = 64;
const double    TotemTimingDQMSource::SAMPIC_ADC_V = 1./256;
const int       TotemTimingDQMSource::CTPPS_NUM_OF_ARMS = 2;
const int       TotemTimingDQMSource::TOTEM_TIMING_STATION_ID = 2;
const int       TotemTimingDQMSource::TOTEM_STATION_210 = 0;
const int       TotemTimingDQMSource::TOTEM_STATION_220 = 1;
const int       TotemTimingDQMSource::TOTEM_TIMING_MIN_RP_ID = 0;
const int       TotemTimingDQMSource::TOTEM_TIMING_MAX_RP_ID = 1;
const int       TotemTimingDQMSource::TOTEM_STRIP_MIN_RP_ID = 4;
const int       TotemTimingDQMSource::TOTEM_STRIP_MAX_RP_ID = 5;
const int       TotemTimingDQMSource::CTPPS_NEAR_RP_ID = 2;
const int       TotemTimingDQMSource::CTPPS_FAR_RP_ID = 3;
const int       TotemTimingDQMSource::TOTEM_TIMING_NUM_OF_PLANES = 4;
const int       TotemTimingDQMSource::TOTEM_TIMING_NUM_OF_CHANNELS = 12;
const int       TotemTimingDQMSource::TOTEM_TIMING_FED_ID_45 = FEDNumbering::MAXTotemRPTimingVerticalFEDID;
const int       TotemTimingDQMSource::TOTEM_TIMING_FED_ID_56 = FEDNumbering::MINTotemRPTimingVerticalFEDID;

//----------------------------------------------------------------------------------------------------

TotemTimingDQMSource::GlobalPlots::GlobalPlots( DQMStore::IBooker& ibooker )
{
  ibooker.setCurrentFolder( "CTPPS" );

  h_trackCorr_hor = ibooker.book2D( "track correlation all vertical", "rp, all, ver", 12, -0.5, 11.5, 12, -0.5, 11.5 );
  TH2F* hist = h_trackCorr_hor->getTH2F();
  TAxis* xa = hist->GetXaxis(), *ya = hist->GetYaxis();
  xa->SetBinLabel( 12, "45, 210, far, top" ); ya->SetBinLabel( 1, "45, 210, far, top" );
  xa->SetBinLabel( 11, "45, 220, timing, top" );  ya->SetBinLabel( 2, "45, 220, timing, top" );
  xa->SetBinLabel( 10, "45, 220, far, top" );  ya->SetBinLabel( 3, "45, 220, far, top" );
  xa->SetBinLabel( 9, "56, 210, far, top" ); ya->SetBinLabel( 4, "56, 210, far, top" );
  xa->SetBinLabel( 8, "56, 220, timing, top" );  ya->SetBinLabel( 5, "56, 220, timing, top" );
  xa->SetBinLabel( 7, "56, 220, far, top" );  ya->SetBinLabel( 6, "56, 220, far, top" );

  xa->SetBinLabel( 6, "45, 210, far, bot" ); ya->SetBinLabel( 7, "45, 210, far, bot" );
  xa->SetBinLabel( 5, "45, 220, timing, bot" );  ya->SetBinLabel( 8, "45, 220, timing, bot" );
  xa->SetBinLabel( 4, "45, 220, far, bot" );  ya->SetBinLabel( 9, "45, 220, far, bot" );
  xa->SetBinLabel( 3, "56, 210, far, bot" ); ya->SetBinLabel( 10, "56, 210, far, bot" );
  xa->SetBinLabel( 2, "56, 220, timing, bot" );  ya->SetBinLabel( 11, "56, 220, timing, bot" );
  xa->SetBinLabel( 1, "56, 220, far, bot" );  ya->SetBinLabel( 12, "56, 220, far, bot" );
}

//----------------------------------------------------------------------------------------------------


TotemTimingDQMSource::PotPlots::PotPlots( DQMStore::IBooker& ibooker, unsigned int id )
{
  std::string path, title;
  TotemTimingDetId( id ).rpName( path, TotemTimingDetId::nPath );
  ibooker.setCurrentFolder( path );

  TotemTimingDetId( id ).rpName( title, TotemTimingDetId::nFull );

  activityPerBX = ibooker.book1D( "activity per BX CMS", title+" Activity per BX;Event.BX", 3600, -1.5, 3598. + 0.5 );

  tirggerCellTime = ibooker.book1D( "trigger Cell Time", title+" Trigger Cell Time; t (ns)", 500, -25, 25 );

  dataSamplesRaw = ibooker.book1D( "raw Samples", title+" Raw Samples; ADC", 255, 0, 255 );

//   hitDistribution2d = ibooker.book2D( "hits in planes", title+" hits in planes;plane number;x (mm)", 10, -0.5, 4.5, 19.*INV_DISPLAY_RESOLUTION_FOR_HITS_MM, -1, 18 );    //TODO needs RecHits
//   hitDistribution2d_lumisection = ibooker.book2D( "hits in planes lumisection", title+" hits in planes in the last lumisection;plane number;x (mm)", 10, -0.5, 4.5, 19.*INV_DISPLAY_RESOLUTION_FOR_HITS_MM, -1, 18 );    //TODO needs RecHits, uncomment Reset in beginLuminosityBlock

  digiDistribution = ibooker.book2D( "digi distribution", title+" digi distribution;plane;channel", 10, -0.5, 4.5, 12, 0, 12);
  digiSent = ibooker.book2D( "digis sent", title+" digi sent (sampic);board + 0.5 sampic;channel", 14, -0.5, 6.5, 16, 0, 16);
  digiAll = ibooker.book2D( "all digis", title+" all digis(sampic);board + 0.5 sampic;channel", 14, -0.5, 6.5, 16, 0, 16);
  digiSentPercentage = ibooker.book2D( "sent digis percentage", title+" sent digis percentage(sampic);board + 0.5 sampic;channel", 14, -0.5, 6.5, 16, 0, 16);

  noiseRMS = ibooker.book2D( "noise RMS", title+" noise RMS (V);plane;channel", 10, -0.5, 4.5, 12, 0, 12);
  baseline = ibooker.book2D( "baseline", title+" baseline (V);plane;channel", 10, -0.5, 4.5, 12, 0, 12);
  meanAmplitude = ibooker.book2D( "mean Amplitude", title+" Mean Amplitude (V);plane;channel", 10, -0.5, 4.5, 12, 0, 12);
  hitRate = ibooker.book2D( "hit rate", title+" hit rate (Hz);plane;channel", 10, -0.5, 4.5, 12, 0, 12);

  activePlanes = ibooker.book1D( "active planes", title+" active planes (per event);number of active planes", 6, -0.5, 5.5 );

//   trackDistribution = ibooker.book1D( "tracks", title+" tracks;x (mm)", 19.*INV_DISPLAY_RESOLUTION_FOR_HITS_MM, -1, 18 );    //TODO needs tracks

  stripTomography210 = ibooker.book2D( "tomography 210", title+" tomography with strips 210 (all planes);x + 25*plane(mm);y (mm)", 300, -15, 15, 105, -15, 90 );
  stripTomography220 = ibooker.book2D( "tomography 220", title+" tomography with strips 220 (all planes);x + 25*plane(mm);y (mm)", 300, -15, 15, 105, -15, 90 );

//   leadingEdge = ibooker.book1D( "leading edge", title+" leading edge; leading edge (ns)", 125, 0, 125 );    //TODO needs RecHits
  amplitude = ibooker.book1D( "amplitude", title+" amplitude p-p; amplitude (V)", 100, 0, 1 );
}

//----------------------------------------------------------------------------------------------------

TotemTimingDQMSource::PlanePlots::PlanePlots( DQMStore::IBooker& ibooker, unsigned int id )
{
  std::string path, title;
  TotemTimingDetId( id ).planeName( path, TotemTimingDetId::nPath );
  ibooker.setCurrentFolder( path );

  TotemTimingDetId( id ).planeName( title, TotemTimingDetId::nFull );

  digiDistribution = ibooker.book1D( "digi distribution", title+" digi distribution;channel", 12, 0, 12);

//   hitProfile = ibooker.book1D( "hit profile", title+" hit profile;x (mm)", 19.*INV_DISPLAY_RESOLUTION_FOR_HITS_MM, -1, 18 );    //TODO needs RecHits
  hitMultiplicity = ibooker.book1D( "channels per plane", title+" channels per plane; ch per plane", 13, -0.5, 12.5 );
}

//----------------------------------------------------------------------------------------------------

TotemTimingDQMSource::ChannelPlots::ChannelPlots( DQMStore::IBooker& ibooker, unsigned int id ) : hitsCounterPerLumisection(0)
{
  std::string path, title;
  TotemTimingDetId( id ).channelName( path, TotemTimingDetId::nPath );
  ibooker.setCurrentFolder( path );

  TotemTimingDetId( id ).channelName( title, TotemTimingDetId::nFull );


  activityPerBX = ibooker.book1D( "activity per BX", title+" Activity per BX;Event.BX", 1000, -1.5, 998. + 0.5 );

  tirggerCellTime = ibooker.book1D( "trigger Cell Time", title+" Trigger Cell Time; t (ns)", 500, -25, 25 );

  dataSamplesRaw = ibooker.book1D( "raw Samples", title+" Raw Samples; ADC", 255, 0, 255 );
  amplitude = ibooker.book1D( "amplitude", title+" amplitude p-p; amplitude (V)", 100, 0, 1 );
  noiseSamples = ibooker.book1D( "noise samples", title+" noise samples; V", 100, 0, 1 );

  timestampA = ibooker.book1D( "timestampA", title+" TimestampA; Decimal", 4096, 0, 4096 );
  timestampB = ibooker.book1D( "timestampB", title+" TimestampB; Decimal", 4096, 0, 4096 );
  timestampDiff = ibooker.book1D( "timestampDiff", title+" timestampDiff; Decimal", 10, -5, 5 );

  hitTime = ibooker.book1D( "hit time", title+"hit time;t - t_previous (us)", 100, 0, 10000);
  hitRate = ibooker.book1D( "hit rate", title+"hit rate;rate (Hz)", 100, 0, 10000);
}

//----------------------------------------------------------------------------------------------------

TotemTimingDQMSource::TotemTimingDQMSource( const edm::ParameterSet& ps ) :
  tokenLocalTrack_  ( consumes< edm::DetSetVector<TotemRPLocalTrack> >          ( ps.getParameter<edm::InputTag>( "tagLocalTrack" ) ) ),
  tokenDigi_        ( consumes< edm::DetSetVector<TotemTimingDigi> >            ( ps.getParameter<edm::InputTag>( "tagDigi" ) ) ),
  tokenDiamondHit_  ( consumes< edm::DetSetVector<CTPPSDiamondRecHit> >         ( ps.getParameter<edm::InputTag>( "tagDiamondRecHits" ) ) ),
  tokenDiamondTrack_( consumes< edm::DetSetVector<CTPPSDiamondLocalTrack> >     ( ps.getParameter<edm::InputTag>( "tagDiamondLocalTracks" ) ) ),
  tokenFEDInfo_     ( consumes< std::vector<TotemFEDInfo> >                     ( ps.getParameter<edm::InputTag>( "tagFEDInfo" ) ) ),
  minimumStripAngleForTomography_       ( ps.getParameter<double>( "minimumStripAngleForTomography" ) ),
  maximumStripAngleForTomography_       ( ps.getParameter<double>( "maximumStripAngleForTomography" ) ),
  samplesForNoise_                      ( ps.getUntrackedParameter<unsigned int>( "samplesForNoise", 5 ) ),
  verbosity_                            ( ps.getUntrackedParameter<unsigned int>( "verbosity", 0 ) ),
  timeOfPreviousEvent_                  ( 0 )
{
}

//----------------------------------------------------------------------------------------------------

TotemTimingDQMSource::~TotemTimingDQMSource()
{}

//----------------------------------------------------------------------------------------------------

void
TotemTimingDQMSource::dqmBeginRun( const edm::Run& iRun, const edm::EventSetup& )
{
}


//----------------------------------------------------------------------------------------------------

void
TotemTimingDQMSource::bookHistograms( DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup& )
{
  ibooker.cd();
  ibooker.setCurrentFolder( "CTPPS" );

  globalPlot_= GlobalPlots( ibooker );

  for ( unsigned short arm = 0; arm < CTPPS_NUM_OF_ARMS; ++arm ) {
    for (unsigned short rp = TOTEM_TIMING_MIN_RP_ID; rp <= TOTEM_TIMING_MAX_RP_ID; ++rp) {
      const TotemTimingDetId rpId( arm, TOTEM_TIMING_STATION_ID, rp );
      potPlots_[rpId] = PotPlots( ibooker, rpId );
      for ( unsigned short pl = 0; pl < TOTEM_TIMING_NUM_OF_PLANES; ++pl ) {
        const TotemTimingDetId plId( arm, TOTEM_TIMING_STATION_ID, rp, pl );
        planePlots_[plId] = PlanePlots( ibooker, plId);
        for ( unsigned short ch = 0; ch < TOTEM_TIMING_NUM_OF_CHANNELS; ++ch ) {
          const TotemTimingDetId chId( arm, TOTEM_TIMING_STATION_ID, rp, pl, ch );
          channelPlots_[chId] = ChannelPlots( ibooker, chId );
        }
      }
    }
  }
}


//----------------------------------------------------------------------------------------------------

void
TotemTimingDQMSource::beginLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& )
{
}

//----------------------------------------------------------------------------------------------------

void
TotemTimingDQMSource::analyze( const edm::Event& event, const edm::EventSetup& eventSetup )
{
  // get event setup data
  edm::ESHandle<CTPPSGeometry> geometry;
  eventSetup.get<VeryForwardRealGeometryRecord>().get(geometry);

  // get event data
  edm::Handle< edm::DetSetVector<TotemRPLocalTrack> > stripTracks;
  event.getByToken( tokenLocalTrack_, stripTracks );

  edm::Handle< edm::DetSetVector<TotemTimingDigi> > timingDigis;
  event.getByToken( tokenDigi_, timingDigis );

  edm::Handle< std::vector<TotemFEDInfo> > fedInfo;
  event.getByToken( tokenFEDInfo_, fedInfo );

  edm::Handle< edm::DetSetVector<CTPPSDiamondRecHit> > diamondRecHits;
  event.getByToken( tokenDiamondHit_, diamondRecHits );

  edm::Handle< edm::DetSetVector<CTPPSDiamondLocalTrack> > diamondLocalTracks;
  event.getByToken( tokenDiamondTrack_, diamondLocalTracks );

  // check validity
  bool valid = true;
  valid &= timingDigis.isValid();
  valid &= fedInfo.isValid();

  if ( !valid ) {
    if ( verbosity_ ) {
      edm::LogProblem("TotemTimingDQMSource")
        << "ERROR in TotemDQMModuleRP::analyze > some of the required inputs are not valid. Skipping this event.\n"
        << "    timingDigis.isValid = " << timingDigis.isValid() << "\n"
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
  //TODO

  // Using TotemTimingDigi
  std::set<uint8_t> boardSet;
  std::unordered_map<unsigned int, unsigned int> channelsPerPlane;

  for ( const auto& digis : *timingDigis ) {
    const TotemTimingDetId detId( digis.detId() );
    TotemTimingDetId detId_pot( digis.detId() );
    detId_pot.setPlane( 0 );
    detId_pot.setChannel( 0 );
    TotemTimingDetId detId_plane( digis.detId() );
    detId_plane.setChannel( 0 );

    for ( const auto& digi : digis ) {


      // Time of samples
      unsigned int cell0TimeClock;
      double cell0TimeInstant;  // Time of first cell
      double triggerCellTimeInstant;    // Time of triggered cell

      uint16_t timestampA = timestampAConverter( digi.getTimestampA() );
      unsigned int timestamp = digi.getCellInfo() <= 32 ? timestampA : digi.getTimestampB();

      cell0TimeClock =  timestamp + ( ( digi.getFPGATimestamp() - timestamp ) & 0xFFFFFFF000 ) - digi.getEventInfo().getL1ATimestamp() + digi.getEventInfo().getL1ALatency();
      cell0TimeInstant = SAMPIC_MAX_NUMBER_OF_SAMPLES * SAMPIC_SAMPLING_PERIOD_NS * cell0TimeClock;

      if ( digi.getCellInfo() < digi.getEventInfo().getOffsetOfSamples() )
        triggerCellTimeInstant = cell0TimeInstant + digi.getCellInfo() * SAMPIC_SAMPLING_PERIOD_NS;
      else
        triggerCellTimeInstant = cell0TimeInstant - ( digi.getNumberOfSamples() - digi.getCellInfo() ) * SAMPIC_SAMPLING_PERIOD_NS;
      // End time of samples

      // Pot Plots
      if ( potPlots_.find( detId_pot ) != potPlots_.end() )
      {
        potPlots_[detId_pot].digiDistribution->Fill( detId.plane(), detId.channel() );
        float boardId =  digi.getEventInfo().getHardwareBoardId() + 0.5*digi.getEventInfo().getHardwareSampicId();
        potPlots_[detId_pot].digiSent->Fill( boardId, digi.getHardwareChannelId() );
        if ( boardSet.find( digi.getEventInfo().getHardwareId() ) == boardSet.end() )
        {
          // This guarantees that every board is counted only once
          boardSet.insert( digi.getEventInfo().getHardwareId() );
          std::bitset<16> chMap( digi.getEventInfo().getChannelMap() );
          for ( int i=0; i<16; ++i)
          {
            if ( chMap.test(i) ) potPlots_[detId_pot].digiAll->Fill( boardId, i );
          }
        }

        potPlots_[detId_pot].activityPerBX->Fill( event.bunchCrossing() );

        potPlots_[detId_pot].tirggerCellTime->Fill( triggerCellTimeInstant );

        potPlots_[detId_pot].planesWithHits.insert( detId.plane() );

        for ( auto it = digi.getSamplesBegin(); it != digi.getSamplesEnd(); ++it )
          potPlots_[detId_pot].dataSamplesRaw->Fill( *it );

        potPlots_[detId_pot].amplitude->Fill( SAMPIC_ADC_V * ( *( std::max_element( digi.getSamplesBegin(), digi.getSamplesEnd() ) ) - *( std::min_element( digi.getSamplesBegin(), digi.getSamplesEnd() ) ) ) );


        // Tomography of timing using strips
        if ( stripTracks.isValid() )
        {
          for ( const auto& ds : *stripTracks ) {
            const CTPPSDetId stripId( ds.detId() );
            // mean position of U and V planes
            TotemRPDetId plId_V(stripId); plId_V.setPlane(0);
            TotemRPDetId plId_U(stripId); plId_U.setPlane(1);

            double rp_x = ( geometry->getSensor(plId_V)->translation().x() +
                            geometry->getSensor(plId_U)->translation().x() ) / 2.;
            double rp_y = ( geometry->getSensor(plId_V)->translation().y() +
                            geometry->getSensor(plId_U)->translation().y() ) / 2.;

            for ( const auto& striplt : ds ) {
              if ( striplt.isValid() && stripId.arm() == detId_pot.arm() )
              {
                if ( striplt.getTx() > maximumStripAngleForTomography_ || striplt.getTy() > maximumStripAngleForTomography_) continue;
                if ( striplt.getTx() < minimumStripAngleForTomography_ || striplt.getTy() < minimumStripAngleForTomography_) continue;
                if ( stripId.rp() - detId_pot.rp() == ( TOTEM_STRIP_MAX_RP_ID - TOTEM_TIMING_MAX_RP_ID ) )
                {
                  double x = striplt.getX0() - rp_x;
                  double y = striplt.getY0() - rp_y;
                  if ( stripId.station() == TOTEM_STATION_210 )
                    potPlots_[detId_pot].stripTomography210->Fill( x + 25*detId.plane(), y );
                  else if ( stripId.station() == TOTEM_STATION_220 )
                    potPlots_[detId_pot].stripTomography220->Fill( x + 25*detId.plane(), y );
                }
              }
            }
          }
        }
      }

      // Plane Plots
      if ( planePlots_.find( detId_plane ) != planePlots_.end() )
      {
        planePlots_[detId_plane].digiDistribution->Fill( detId.channel() );

        if ( channelsPerPlane.find(detId_plane) != channelsPerPlane.end() ) channelsPerPlane[detId_plane]++;
        else channelsPerPlane[detId_plane] = 0;
      }

      // Channel Plots
      if ( channelPlots_.find( detId ) != channelPlots_.end() )
      {
        channelPlots_[ detId ].activityPerBX->Fill( event.bunchCrossing() );
        channelPlots_[ detId ].tirggerCellTime->Fill( triggerCellTimeInstant );
        for ( auto it = digi.getSamplesBegin(); it != digi.getSamplesEnd(); ++it )
          channelPlots_[ detId ].dataSamplesRaw->Fill( *it );
        for ( unsigned short i=0; i<samplesForNoise_; ++i )
          channelPlots_[ detId ].noiseSamples->Fill( SAMPIC_ADC_V * digi.getSampleAt( i ) );
        channelPlots_[ detId ].amplitude->Fill( SAMPIC_ADC_V * ( *( std::max_element( digi.getSamplesBegin(), digi.getSamplesEnd() ) ) - *( std::min_element( digi.getSamplesBegin(), digi.getSamplesEnd() ) ) ) );

        channelPlots_[ detId ].timestampA->Fill( timestampA );
        channelPlots_[ detId ].timestampB->Fill( digi.getTimestampB() );
        channelPlots_[ detId ].timestampDiff->Fill( timestampA - digi.getTimestampB() );

        if ( timeOfPreviousEvent_ != 0 ) channelPlots_[ detId ].hitTime->Fill( 1e-3 * LHC_CLOCK_PERIOD_NS * ( event.time().value() - timeOfPreviousEvent_ ) );
        ++( channelPlots_[ detId ].hitsCounterPerLumisection );
      }
    }
  }

  for ( auto& plt : potPlots_ ) {
    plt.second.activePlanes->Fill( plt.second.planesWithHits.size() );
    plt.second.planesWithHits.clear();
  }

  for ( const auto& plt : channelsPerPlane ) {
    planePlots_[plt.first].hitMultiplicity->Fill( plt.second );
  }



  timeOfPreviousEvent_ = event.time().value();

}

//----------------------------------------------------------------------------------------------------

void
TotemTimingDQMSource::endLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& )
{
  // Efficeincy of Data Transmission from SAMPIC
  for ( auto& plot : potPlots_ ) {
    TH2F *hitHistoTmp = plot.second.digiSentPercentage->getTH2F();
    TH2F *histoSent = plot.second.digiSent->getTH2F();
    TH2F *histoAll = plot.second.digiAll->getTH2F();

    hitHistoTmp->Divide( histoSent, histoAll );
    hitHistoTmp->Scale(100);

    plot.second.noiseRMS->Reset();
    plot.second.baseline->Reset();
    plot.second.meanAmplitude->Reset();
    TotemTimingDetId rpId(plot.first);
    for ( auto& chPlot : channelPlots_ ) {
      TotemTimingDetId chId(chPlot.first);
      if ( chId.arm() == rpId.arm() && chId.rp() == rpId.rp() ) {
       plot.second.baseline->Fill( chId.plane(), chId.channel(), chPlot.second.noiseSamples->getTH1F()->GetMean() );
       plot.second.noiseRMS->Fill( chId.plane(), chId.channel(), chPlot.second.noiseSamples->getTH1F()->GetRMS() );
       plot.second.meanAmplitude->Fill( chId.plane(), chId.channel(), chPlot.second.amplitude->getTH1F()->GetMean() );

       plot.second.hitRate->Fill( chId.plane(), chId.channel(), (double) chPlot.second.hitsCounterPerLumisection * HIT_RATE_FACTOR );
      }
    }
  }

  for ( auto& plot : channelPlots_ ) {
    if ( plot.second.hitsCounterPerLumisection != 0 ) {
      plot.second.hitRate->Fill( (double) plot.second.hitsCounterPerLumisection * HIT_RATE_FACTOR );
    }
    plot.second.hitsCounterPerLumisection = 0;
  }
}

//----------------------------------------------------------------------------------------------------

void
TotemTimingDQMSource::endRun( const edm::Run&, const edm::EventSetup& )
{}

//----------------------------------------------------------------------------------------------------

uint16_t TotemTimingDQMSource::timestampAConverter(const uint16_t& binary) const
{
  uint16_t gray;
  gray = binary & 0x800;
  for (unsigned short int i = 1; i < 12; ++i)
    gray |= ( binary ^ ( binary >> 1 ) ) & (0x0001 << ( 11 - i ) );

  gray = 0xFFF - gray;

  uint16_t binaryOut = 0;
  binaryOut = gray & 0x800;
  for (unsigned short int i = 1; i < 12; ++i)
    binaryOut |= ( gray ^ ( binaryOut >> 1 ) ) & (0x0001 << ( 11 - i ) );

  return binaryOut;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE( TotemTimingDQMSource );

