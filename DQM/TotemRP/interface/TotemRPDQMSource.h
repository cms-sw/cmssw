/****************************************************************************
*
* This is a part of TotemDQM and TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*   Rafał Leszko (rafal.leszko@gmail.com)
*
****************************************************************************/

#ifndef TotemRPDQMSource_H
#define TotemRPDQMSource_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/TotemDigi/interface/TotemRPDigi.h"
#include "DataFormats/TotemRPReco/interface/TotemRPCluster.h"
#include "DataFormats/TotemRPReco/interface/TotemRPRecHit.h"
#include "RecoTotemRP/RPRecoDataFormats/interface/RPRecognizedPatternsCollection.h"
#include "RecoTotemRP/RPRecoDataFormats/interface/RPFittedTrack.h"
#include "RecoTotemRP/RPRecoDataFormats/interface/RPFittedTrackCollection.h"
#include "RecoTotemRP/RPRecoDataFormats/interface/RPTrackCandidateCollection.h"
#include "RecoTotemRP/RPRecoDataFormats/interface/RPMulFittedTrackCollection.h"
#include "Geometry/TotemRPDetTopology/interface/RPTopology.h"

#include "DQM/TotemRP/interface/CorrelationPlotsSelector.h"

//----------------------------------------------------------------------------------------------------
 
class TotemRPDQMSource: public DQMEDAnalyzer
{
  public:
    TotemRPDQMSource(const edm::ParameterSet& ps);
    virtual ~TotemRPDQMSource();
  
  protected:
    void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
    void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
    void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
    void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
    void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  private:
    edm::EDGetTokenT< edm::DetSetVector<TotemRPDigi> > tokenStripDigi;
    edm::EDGetTokenT< edm::DetSetVector<TotemRPCluster> > tokenDigiCluster;
    edm::EDGetTokenT< edm::DetSetVector<TotemRPRecHit> > tokenRecoHit;
    edm::EDGetTokenT< RPRecognizedPatternsCollection > tokenPatternColl;
    edm::EDGetTokenT< RPTrackCandidateCollection > tokenTrackCandColl;
    edm::EDGetTokenT< RPFittedTrackCollection > tokenTrackColl;
    edm::EDGetTokenT< RPMulFittedTrackCollection > tokenMultiTrackColl;

    bool buildCorrelationPlots;                           ///< decides wheather the correlation plots are created
    unsigned int correlationPlotsLimit;                   ///< maximum number of created correlation plots
    CorrelationPlotsSelector correlationPlotsSelector;

    /// plots related to one (anti)diagonal
    struct DiagonalPlots
    {
      int id;

      MonitorElement *h_lrc_x_d=NULL, *h_lrc_x_n=NULL, *h_lrc_x_f=NULL;
      MonitorElement *h_lrc_y_d=NULL, *h_lrc_y_n=NULL, *h_lrc_y_f=NULL;

      DiagonalPlots() {}

      DiagonalPlots(DQMStore::IBooker &ibooker, int _id);
    };

    std::map<unsigned int, DiagonalPlots> diagonalPlots;

    /// plots related to one arm
    struct ArmPlots
    {
      int id;

      MonitorElement *h_numRPWithTrack_top=NULL, *h_numRPWithTrack_hor=NULL, *h_numRPWithTrack_bot=NULL;
      MonitorElement *h_trackCorr=NULL, *h_trackCorr_overlap=NULL;

      ArmPlots(){}

      ArmPlots(DQMStore::IBooker &ibooker, int _id);
    };

    std::map<unsigned int, ArmPlots> armPlots;

    /// plots related to one station
    struct StationPlots
    {
      int id;

      std::map<int, std::map<int, MonitorElement*> > hist;

      StationPlots() {}
      StationPlots(DQMStore::IBooker &ibooker, int _id, std::set<unsigned int> planes, bool allocateCorrelationPlots, 
        CorrelationPlotsSelector *correlationPlotsSelector, int limit = -1);

      void Add(DQMStore::IBooker &ibooker, std::set<unsigned int> planes, CorrelationPlotsSelector *correlationPlotsSelector, int limit = -1);
    };

    std::map<unsigned int, StationPlots> stationPlots;

    /// plots related to one RP
    struct PotPlots
    {
      MonitorElement *activity=NULL, *activity_u=NULL, *activity_v=NULL;
      MonitorElement *hit_plane_hist=NULL;
      MonitorElement *patterns_u=NULL, *patterns_v=NULL;
      MonitorElement *h_planes_fit_u=NULL, *h_planes_fit_v=NULL;
      MonitorElement *event_category=NULL;
      MonitorElement *trackHitsCumulativeHist=NULL;
      MonitorElement *track_u_profile=NULL, *track_v_profile=NULL;

      PotPlots() {}
      PotPlots(DQMStore::IBooker &ibooker, unsigned int id);
    };

    std::map<unsigned int, PotPlots> potPlots;

    /// plots related to one RP plane
    struct PlanePlots
    {
      MonitorElement *digi_profile_cumulative = NULL;
      MonitorElement *cluster_profile_cumulative = NULL;
      MonitorElement *hit_multiplicity = NULL;
      MonitorElement *cluster_size = NULL;

      PlanePlots() {}
      PlanePlots(DQMStore::IBooker &ibooker, unsigned int id);
    };

    std::map<unsigned int, PlanePlots> planePlots;
};

#endif
