/*
 * Package:     CommonAlignmentProducer
 * Class  :     AlignmentMonitorMuonVsCurvature
 *
 * Original Author:  Jim Pivarski
 *         Created:  Fri Feb 19 21:45:02 CET 2010
 *
 * $Id:$
 */

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include <sstream>
#include "TProfile.h"
#include "TH2F.h"
#include "TH1F.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"


class AlignmentMonitorMuonVsCurvature: public AlignmentMonitorBase
{
public:
  AlignmentMonitorMuonVsCurvature(const edm::ParameterSet& cfg);
  virtual ~AlignmentMonitorMuonVsCurvature() {}

  void book() override;

  void event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks) override;
  void processMuonResidualsFromTrack(MuonResidualsFromTrack &mrft, const Trajectory* traj = NULL);

private:
  
  edm::InputTag m_muonCollectionTag;
  double m_minTrackPt;
  double m_minTrackP;
  int m_minTrackerHits;
  double m_maxTrackerRedChi2;
  bool m_allowTIDTEC;
  bool m_minNCrossedChambers;
  double m_maxDxy;
  int m_minDT13Hits;
  int m_minDT2Hits;
  int m_minCSCHits;
  int m_layer;
  std::string m_propagator;
  bool m_doDT;
  bool m_doCSC;

  enum {
    kDeltaX = 0,
    kDeltaDxDz,
    kNumComponents
  };

  // DT [wheel][station][sector]
  TH2F *th2f_wheel_st_sector[5][4][14][kNumComponents];
  TProfile *tprofile_wheel_st_sector[5][4][14][kNumComponents];
  
  // CSC [endcap*station][ring][chamber]
  TH2F *th2f_st_ring_chamber[8][3][36][kNumComponents];
  TProfile *tprofile_st_ring_chamber[8][3][36][kNumComponents];

  TH1F *th1f_trackerRedChi2;
  TH1F *th1f_trackerRedChi2Diff;
};


AlignmentMonitorMuonVsCurvature::AlignmentMonitorMuonVsCurvature(const edm::ParameterSet& cfg)
: AlignmentMonitorBase(cfg, "AlignmentMonitorMuonVsCurvature")
, m_muonCollectionTag(cfg.getParameter<edm::InputTag>("muonCollectionTag"))
, m_minTrackPt(cfg.getParameter<double>("minTrackPt"))
, m_minTrackP(cfg.getParameter<double>("minTrackP"))
, m_minTrackerHits(cfg.getParameter<int>("minTrackerHits"))
, m_maxTrackerRedChi2(cfg.getParameter<double>("maxTrackerRedChi2"))
, m_allowTIDTEC(cfg.getParameter<bool>("allowTIDTEC"))
, m_minNCrossedChambers(cfg.getParameter<int>("minNCrossedChambers"))
, m_maxDxy(cfg.getParameter<double>("maxDxy"))
, m_minDT13Hits(cfg.getParameter<int>("minDT13Hits"))
, m_minDT2Hits(cfg.getParameter<int>("minDT2Hits"))
, m_minCSCHits(cfg.getParameter<int>("minCSCHits"))
, m_layer(cfg.getParameter<int>("layer"))
, m_propagator(cfg.getParameter<std::string>("propagator"))
, m_doDT(cfg.getParameter<bool>("doDT"))
, m_doCSC(cfg.getParameter<bool>("doCSC"))
{}


void AlignmentMonitorMuonVsCurvature::book()
{
  // DT
  std::string wheelname[5] = {"wheelm2_", "wheelm1_", "wheelz_", "wheelp1_", "wheelp2_"};
  if (m_doDT)
  for (int wheel = -2;  wheel <=2 ;  wheel++)
  for (int station = 1; station <= 4; station++)
  for (int sector = 1;  sector <= 14;  sector++)
  {
    if (station != 4 && sector > 12) continue;
    
    char stationname[20];
    sprintf(stationname,"st%d_", station);

    char sectorname[20];
    sprintf(sectorname,"sector%02d_", sector);

    for (int component = 0;  component < kNumComponents;  component++)
    {
      std::stringstream th2f_name, tprofile_name;
      th2f_name << "th2f_" << wheelname[wheel+2] <<stationname<< sectorname;
      tprofile_name << "tprofile_" << wheelname[wheel+2] <<stationname<< sectorname;

      double yminmax = 50., xminmax = 0.05;
      if (m_minTrackPt>0.) xminmax = 1./m_minTrackPt;
      int ynbins = 50;
      if (component == kDeltaX) {
        th2f_name << "deltax";
        tprofile_name << "deltax";
      }
      else if (component == kDeltaDxDz) {
        th2f_name << "deltadxdz";
        tprofile_name << "deltadxdz";
      }

      th2f_wheel_st_sector[wheel+2][station-1][sector-1][component] =
          book2D("/iterN/", th2f_name.str().c_str(), "", 30, -xminmax, xminmax, ynbins, -yminmax, yminmax);
      tprofile_wheel_st_sector[wheel+2][station-1][sector-1][component] =
          bookProfile("/iterN/", tprofile_name.str().c_str(), "", 30,  -xminmax, xminmax);
    }
  }

  // CSC
  std::string stname[8] = {"Ep_S1_", "Ep_S2_", "Ep_S3_", "Ep_S4_", "Em_S1_", "Em_S2_", "Em_S3_", "Em_S4_"};
  if (m_doCSC)
  for (int station = 0;  station < 8;  station++)
  for (int ring = 1;  ring <= 3;  ring++)
  for (int chamber = 1;  chamber <= 36;  chamber++)
  {
    int st = station%4+1;
    if (st > 1 && ring > 2) continue; // only station 1 has more then 2 rings
    if (st > 1 && ring == 1 && chamber > 18) continue; // ring 1 stations 1,2,3 have 18 chambers

    char ringname[20];
    sprintf(ringname,"R%d_", ring);

    char chname[20];
    sprintf(chname,"C%02d_", chamber);

    for (int component = 0;  component < kNumComponents;  component++)
    {
      std::stringstream componentname;
      double yminmax = 50., xminmax = 0.05;
      if (m_minTrackPt>0.) xminmax = 1./m_minTrackPt;
      if (ring == 1) xminmax *= 0.5;
      if (component == kDeltaX) {
	componentname << "deltax";
      }
      else if (component == kDeltaDxDz) {
	componentname << "deltadxdz";
      }
      
      std::stringstream th2f_name, tprofile_name;
      th2f_name << "th2f_" << stname[station] << ringname << chname << componentname.str();
      tprofile_name << "tprofile_" << stname[station] << ringname << chname << componentname.str();

      th2f_st_ring_chamber[station][ring-1][chamber-1][component] =
          book2D("/iterN/", th2f_name.str().c_str(), "", 30, -xminmax, xminmax, 100, -yminmax, yminmax);
      tprofile_st_ring_chamber[station][ring-1][chamber-1][component] =
          bookProfile("/iterN/", tprofile_name.str().c_str(), "", 30, -xminmax, xminmax);
    }
  }

  th1f_trackerRedChi2 = book1D("/iterN/", "trackerRedChi2", "Refit tracker reduced chi^2", 100, 0., 30.);
  th1f_trackerRedChi2Diff = book1D("/iterN/", "trackerRedChi2Diff", "Fit-minus-refit tracker reduced chi^2", 100, -5., 5.);
}


void AlignmentMonitorMuonVsCurvature::event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& trajtracks)
{
  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByLabel(m_beamSpotTag, beamSpot);

  edm::ESHandle<DetIdAssociator> muonDetIdAssociator_;
  iSetup.get<DetIdAssociatorRecord>().get("MuonDetIdAssociator", muonDetIdAssociator_);

  edm::ESHandle<Propagator> prop;
  iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny",prop);
  
  edm::ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);

  if (m_muonCollectionTag.label().empty()) // use trajectories
  {
    for (ConstTrajTrackPairCollection::const_iterator trajtrack = trajtracks.begin();  trajtrack != trajtracks.end();  ++trajtrack)
    {
      const Trajectory* traj = (*trajtrack).first;
      const reco::Track* track = (*trajtrack).second;

      if (track->pt() > m_minTrackPt  && track->p() > m_minTrackP  &&  fabs(track->dxy(beamSpot->position())) < m_maxDxy )
      {
        MuonResidualsFromTrack muonResidualsFromTrack(iSetup, magneticField, globalGeometry, muonDetIdAssociator_, prop, traj, track, pNavigator(), 1000.);
        processMuonResidualsFromTrack(muonResidualsFromTrack, traj );
      } // end if track pT is within range
    } // end loop over tracks
  }
  else
  {
    edm::Handle<reco::MuonCollection> muons;
    iEvent.getByLabel(m_muonCollectionTag, muons);

    for (reco::MuonCollection::const_iterator muon = muons->begin();  muon != muons->end();  ++muon)
    {
      if ( !(muon->isTrackerMuon() && muon->innerTrack().isNonnull() ) ) continue;

      if (m_minTrackPt < muon->pt()  &&  m_minTrackP < muon->p()  &&  fabs(muon->innerTrack()->dxy(beamSpot->position())) < m_maxDxy)
      {
        MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, &(*muon), pNavigator(), 100.);
        processMuonResidualsFromTrack(muonResidualsFromTrack);
      }
    }
  }
}


void AlignmentMonitorMuonVsCurvature::processMuonResidualsFromTrack(MuonResidualsFromTrack &mrft, const Trajectory* traj)
{
  if (mrft.trackerNumHits() < m_minTrackerHits) return;
  if (!m_allowTIDTEC  && mrft.contains_TIDTEC()) return;
  
  int nMuChambers = 0;
  std::vector<DetId> chamberIds = mrft.chamberIds();
  for (unsigned ch=0; ch < chamberIds.size(); ch++)  if (chamberIds[ch].det() == DetId::Muon)  nMuChambers++;
  if (nMuChambers < m_minNCrossedChambers ) return;
  
  th1f_trackerRedChi2->Fill(mrft.trackerRedChi2());
  th1f_trackerRedChi2Diff->Fill(mrft.getTrack()->normalizedChi2() - mrft.trackerRedChi2());

  if (mrft.normalizedChi2() > m_maxTrackerRedChi2) return;

  double qoverpt = mrft.getTrack()->charge() / mrft.getTrack()->pt();
  double qoverpz = 0.;
  if (fabs(mrft.getTrack()->pz()) > 0.01) qoverpz = mrft.getTrack()->charge() / fabs(mrft.getTrack()->pz());

  for (std::vector<DetId>::const_iterator chamberId = chamberIds.begin();  chamberId != chamberIds.end();  ++chamberId)
  {
    if (chamberId->det() != DetId::Muon  ) continue;
    
    if (m_doDT  &&  chamberId->subdetId() == MuonSubdetId::DT)
    {
      DTChamberId dtid(chamberId->rawId());
      MuonChamberResidual *dt13 = mrft.chamberResidual(*chamberId, MuonChamberResidual::kDT13);
      
      if (dt13 != NULL  &&  dt13->numHits() >= m_minDT13Hits)
      {
        int wheel = dtid.wheel() + 2;
        int station = dtid.station() -1;
        int sector = dtid.sector() - 1;

        double resid_x = 10. * dt13->global_residual();
        double resid_dxdz = 1000. * dt13->global_resslope();

        if (fabs(resid_x) < 100. && fabs(resid_dxdz) < 100.)
        {
          th2f_wheel_st_sector[wheel][station][sector][kDeltaX]->Fill(qoverpt, resid_x);
          tprofile_wheel_st_sector[wheel][station][sector][kDeltaX]->Fill(qoverpt, resid_x);
          th2f_wheel_st_sector[wheel][station][sector][kDeltaDxDz]->Fill(qoverpt, resid_dxdz);
          tprofile_wheel_st_sector[wheel][station][sector][kDeltaDxDz]->Fill(qoverpt, resid_dxdz);
        }
      } // if it's a good segment
    } // if DT

    if (m_doCSC  &&  chamberId->subdetId() == MuonSubdetId::CSC)
    {
      CSCDetId cscid(chamberId->rawId());
      MuonChamberResidual *csc = mrft.chamberResidual(*chamberId, MuonChamberResidual::kCSC);

      if (csc != NULL  &&  csc->numHits() >= m_minCSCHits)
      {
        int station = 4*cscid.endcap() + cscid.station() - 5;
        int ring = cscid.ring() - 1;
        if (cscid.station()==1 && cscid.ring()==4) ring = 0; // join ME1/a to ME1/b
        int chamber = cscid.chamber() - 1;

        double resid_x = 10. * csc->global_residual();
        double resid_dxdz = 1000. * csc->global_resslope();

        if (fabs(resid_x) < 100. && fabs(resid_dxdz) < 100.)
        {
          th2f_st_ring_chamber[station][ring][chamber][kDeltaX]->Fill(qoverpz, resid_x);
          tprofile_st_ring_chamber[station][ring][chamber][kDeltaX]->Fill(qoverpz, resid_x);
          th2f_st_ring_chamber[station][ring][chamber][kDeltaDxDz]->Fill(qoverpz, resid_dxdz);
          tprofile_st_ring_chamber[station][ring][chamber][kDeltaDxDz]->Fill(qoverpz, resid_dxdz);
        }
      } // if it's a good segment
    } // if CSC

  } // end loop over chamberIds
}

DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorMuonVsCurvature, "AlignmentMonitorMuonVsCurvature");
