/*
 * Package:     CommonAlignmentProducer
 * Class  :     AlignmentMonitorSegmentDifferences
 *
 * Original Author:  Jim Pivarski
 *         Created:  Mon Nov 12 13:30:14 CST 2007
 *
 * $Id: AlignmentMonitorSegmentDifferences.cc,v 1.5 2011/04/15 23:09:38 khotilov Exp $
 */

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsPositionFitter.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsAngleFitter.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsTwoBin.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include <sstream>

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"


class AlignmentMonitorSegmentDifferences: public AlignmentMonitorBase
{
public:
  AlignmentMonitorSegmentDifferences(const edm::ParameterSet& cfg);
  ~AlignmentMonitorSegmentDifferences() {}

  void book() override;

  void event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks) override;
  void processMuonResidualsFromTrack(MuonResidualsFromTrack &mrft);

private:

  edm::InputTag m_muonCollectionTag;
  double m_minTrackPt;
  double m_minTrackP;
  double m_maxDxy;
  int m_minTrackerHits;
  double m_maxTrackerRedChi2;
  bool m_allowTIDTEC;
  bool m_minNCrossedChambers;
  int m_minDT13Hits;
  int m_minDT2Hits;
  int m_minCSCHits;
  bool m_doDT;
  bool m_doCSC;

  // wheel, sector, stationdiff
  TProfile *m_dt13_resid[5][12][3];
  TProfile *m_dt13_slope[5][12][3];
  TProfile *m_dt2_resid[5][12][2];
  TProfile *m_dt2_slope[5][12][2];
  TH1F *m_posdt13_resid[5][12][3];
  TH1F *m_posdt13_slope[5][12][3];
  TH1F *m_posdt2_resid[5][12][2];
  TH1F *m_posdt2_slope[5][12][2];
  TH1F *m_negdt13_resid[5][12][3];
  TH1F *m_negdt13_slope[5][12][3];
  TH1F *m_negdt2_resid[5][12][2];
  TH1F *m_negdt2_slope[5][12][2];

  // endcap, chamber, stationdiff
  TProfile *m_cscouter_resid[2][36][2];
  TProfile *m_cscouter_slope[2][36][2];
  TProfile *m_cscinner_resid[2][18][3];
  TProfile *m_cscinner_slope[2][18][3];
  TH1F *m_poscscouter_resid[2][36][2];
  TH1F *m_poscscouter_slope[2][36][2];
  TH1F *m_poscscinner_resid[2][18][3];
  TH1F *m_poscscinner_slope[2][18][3];
  TH1F *m_negcscouter_resid[2][36][2];
  TH1F *m_negcscouter_slope[2][36][2];
  TH1F *m_negcscinner_resid[2][18][3];
  TH1F *m_negcscinner_slope[2][18][3];

  // cross-system segdiffs: // [endcap][dtsector]
  TH1F *m_x_pos_dt1_csc1_resid[2][12]; // tracks going through DT W+-2 St1 and CSC St1 Ring3
  TH1F *m_x_pos_dt1_csc2_resid[2][12]; // tracks going through DT W+-2 St1 and CSC St2 Ring2
  TH1F *m_x_pos_dt2_csc1_resid[2][12]; // tracks going through DT W+-2 St2 and CSC St1 Ring3
  TH1F *m_x_neg_dt1_csc1_resid[2][12];
  TH1F *m_x_neg_dt1_csc2_resid[2][12];
  TH1F *m_x_neg_dt2_csc1_resid[2][12];
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// member functions
//

AlignmentMonitorSegmentDifferences::AlignmentMonitorSegmentDifferences(const edm::ParameterSet& cfg)
   : AlignmentMonitorBase(cfg, "AlignmentMonitorSegmentDifferences")
   , m_muonCollectionTag(cfg.getParameter<edm::InputTag>("muonCollectionTag"))
   , m_minTrackPt(cfg.getParameter<double>("minTrackPt"))
   , m_minTrackP(cfg.getParameter<double>("minTrackP"))
   , m_maxDxy(cfg.getParameter<double>("maxDxy"))
   , m_minTrackerHits(cfg.getParameter<int>("minTrackerHits"))
   , m_maxTrackerRedChi2(cfg.getParameter<double>("maxTrackerRedChi2"))
   , m_allowTIDTEC(cfg.getParameter<bool>("allowTIDTEC"))
   , m_minNCrossedChambers(cfg.getParameter<int>("minNCrossedChambers"))
   , m_minDT13Hits(cfg.getParameter<int>("minDT13Hits"))
   , m_minDT2Hits(cfg.getParameter<int>("minDT2Hits"))
   , m_minCSCHits(cfg.getParameter<int>("minCSCHits"))
   , m_doDT(cfg.getParameter<bool>("doDT"))
   , m_doCSC(cfg.getParameter<bool>("doCSC"))
{
}

void AlignmentMonitorSegmentDifferences::book() 
{
  char name[222], pos[222], neg[222];

  double max_curv = 1./m_minTrackPt;

  if (m_doDT) for (int wheel = -2;  wheel <= +2;  wheel++) 
  {
    char wheel_label[][2]={"A","B","C","D","E"};
    for (int sector = 1;  sector <= 12;  sector++) 
    {
      char wheel_sector[50];
      sprintf(wheel_sector,"%s_%02d", wheel_label[wheel+2], sector );

      int nb = 100;
      double wnd = 25.;

      sprintf(name, "dt13_resid_%s_12", wheel_sector);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_dt13_resid[wheel+2][sector-1][0] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_posdt13_resid[wheel+2][sector-1][0] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negdt13_resid[wheel+2][sector-1][0] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
      
      sprintf(name, "dt13_resid_%s_23", wheel_sector);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_dt13_resid[wheel+2][sector-1][1] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_posdt13_resid[wheel+2][sector-1][1] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negdt13_resid[wheel+2][sector-1][1] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
      
      sprintf(name, "dt13_resid_%s_34", wheel_sector);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_dt13_resid[wheel+2][sector-1][2] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_posdt13_resid[wheel+2][sector-1][2] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negdt13_resid[wheel+2][sector-1][2] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);

      sprintf(name, "dt2_resid_%s_12", wheel_sector);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_dt2_resid[wheel+2][sector-1][0] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -200., 200., " ");
      m_posdt2_resid[wheel+2][sector-1][0] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negdt2_resid[wheel+2][sector-1][0] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
      
      sprintf(name, "dt2_resid_%s_23", wheel_sector);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_dt2_resid[wheel+2][sector-1][1] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -200., 200., " ");
      m_posdt2_resid[wheel+2][sector-1][1] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negdt2_resid[wheel+2][sector-1][1] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);

      sprintf(name, "dt13_slope_%s_12", wheel_sector);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_dt13_slope[wheel+2][sector-1][0] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_posdt13_slope[wheel+2][sector-1][0] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negdt13_slope[wheel+2][sector-1][0] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);

      sprintf(name, "dt13_slope_%s_23", wheel_sector);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_dt13_slope[wheel+2][sector-1][1] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_posdt13_slope[wheel+2][sector-1][1] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negdt13_slope[wheel+2][sector-1][1] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
      
      sprintf(name, "dt13_slope_%s_34", wheel_sector);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_dt13_slope[wheel+2][sector-1][2] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_posdt13_slope[wheel+2][sector-1][2] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negdt13_slope[wheel+2][sector-1][2] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
      
      sprintf(name, "dt2_slope_%s_12", wheel_sector);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_dt2_slope[wheel+2][sector-1][0] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -1000., 1000., " ");
      m_posdt2_slope[wheel+2][sector-1][0] = book1D("/iterN/", pos, pos, nb, -100., 100.);
      m_negdt2_slope[wheel+2][sector-1][0] = book1D("/iterN/", neg, neg, nb, -100., 100.);
      
      sprintf(name, "dt2_slope_%s_23", wheel_sector);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_dt2_slope[wheel+2][sector-1][1] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -1000., 1000., " ");
      m_posdt2_slope[wheel+2][sector-1][1] = book1D("/iterN/", pos, pos, nb, -100., 100.);
      m_negdt2_slope[wheel+2][sector-1][1] = book1D("/iterN/", neg, neg, nb, -100., 100.);
    }
  }

  if (m_doCSC) for (int endcap = 1;  endcap <= 2;  endcap++)
  {
    std::string endcapletter;
    if (endcap == 1) endcapletter = "p";
    else if (endcap == 2) endcapletter = "m";

    for (int chamber = 1;  chamber <= 36;  chamber++)
    {
      char ec_chamber[50];
      sprintf(ec_chamber,"%s_%02d", endcapletter.c_str(), chamber );

      int nb = 100;
      double wnd = 60.;

      sprintf(name, "cscouter_resid_%s_12",ec_chamber);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_cscouter_resid[endcap-1][chamber-1][0] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_poscscouter_resid[endcap-1][chamber-1][0] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negcscouter_resid[endcap-1][chamber-1][0] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
      
      sprintf(name, "cscouter_resid_%s_23",ec_chamber);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_cscouter_resid[endcap-1][chamber-1][1] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_poscscouter_resid[endcap-1][chamber-1][1] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negcscouter_resid[endcap-1][chamber-1][1] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
      
      sprintf(name, "cscouter_slope_%s_12",ec_chamber);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_cscouter_slope[endcap-1][chamber-1][0] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_poscscouter_slope[endcap-1][chamber-1][0] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negcscouter_slope[endcap-1][chamber-1][0] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
      
      sprintf(name, "cscouter_slope_%s_23",ec_chamber);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_cscouter_slope[endcap-1][chamber-1][1] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_poscscouter_slope[endcap-1][chamber-1][1] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negcscouter_slope[endcap-1][chamber-1][1] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
    }

    for (int chamber = 1;  chamber <= 18;  chamber++) 
    {
      char ec_chamber[50];
      sprintf(ec_chamber,"%s_%02d", endcapletter.c_str(), chamber );

      int nb = 100;
      double wnd = 40.;
      
      sprintf(name, "cscinner_resid_%s_12",ec_chamber);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_cscinner_resid[endcap-1][chamber-1][0] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_poscscinner_resid[endcap-1][chamber-1][0] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negcscinner_resid[endcap-1][chamber-1][0] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
      
      sprintf(name, "cscinner_resid_%s_23",ec_chamber);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_cscinner_resid[endcap-1][chamber-1][1] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_poscscinner_resid[endcap-1][chamber-1][1] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negcscinner_resid[endcap-1][chamber-1][1] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
      
      sprintf(name, "cscinner_resid_%s_34",ec_chamber);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_cscinner_resid[endcap-1][chamber-1][2] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_poscscinner_resid[endcap-1][chamber-1][2] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negcscinner_resid[endcap-1][chamber-1][2] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);

      sprintf(name, "cscinner_slope_%s_12",ec_chamber);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_cscinner_slope[endcap-1][chamber-1][0] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_poscscinner_slope[endcap-1][chamber-1][0] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negcscinner_slope[endcap-1][chamber-1][0] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);

      sprintf(name, "cscinner_slope_%s_23",ec_chamber);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_cscinner_slope[endcap-1][chamber-1][1] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_poscscinner_slope[endcap-1][chamber-1][1] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negcscinner_slope[endcap-1][chamber-1][1] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);

      sprintf(name, "cscinner_slope_%s_34",ec_chamber);
      sprintf(pos,"pos%s", name);
      sprintf(neg,"neg%s", name);
      m_cscinner_slope[endcap-1][chamber-1][2] = bookProfile("/iterN/", name, name, 20, -max_curv, max_curv, 1, -100., 100., " ");
      m_poscscinner_slope[endcap-1][chamber-1][2] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
      m_negcscinner_slope[endcap-1][chamber-1][2] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
    }
  }

  // cross-system
  for (int e = 1; e<=2; e++)
  for (int s = 1; s <= 12; s++)
  {
    char endcap_sector[50];
    if (e == 1) sprintf(endcap_sector,"Wp2S%02d", s);
    if (e == 2) sprintf(endcap_sector,"Wm2S%02d", s);

    int nb = 200;
    double wnd = 100.;

    sprintf(pos,"pos_x_dt1_csc1_%s", endcap_sector);
    sprintf(neg,"neg_x_dt1_csc1_%s", endcap_sector);
    m_x_pos_dt1_csc1_resid[e-1][s-1] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
    m_x_neg_dt1_csc1_resid[e-1][s-1] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);

    sprintf(pos,"pos_x_dt1_csc2_%s", endcap_sector);
    sprintf(neg,"neg_x_dt1_csc2_%s", endcap_sector);
    m_x_pos_dt1_csc2_resid[e-1][s-1] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
    m_x_neg_dt1_csc2_resid[e-1][s-1] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);

    sprintf(pos,"pos_x_dt2_csc1_%s", endcap_sector);
    sprintf(neg,"neg_x_dt2_csc1_%s", endcap_sector);
    m_x_pos_dt2_csc1_resid[e-1][s-1] = book1D("/iterN/", pos, pos, nb, -wnd, wnd);
    m_x_neg_dt2_csc1_resid[e-1][s-1] = book1D("/iterN/", neg, neg, nb, -wnd, wnd);
  }

}


void AlignmentMonitorSegmentDifferences::event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& trajtracks)
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

      if (track->pt() > m_minTrackPt && track->p() > m_minTrackP && fabs(track->dxy(beamSpot->position())) < m_maxDxy )
      {
        MuonResidualsFromTrack muonResidualsFromTrack(iSetup, magneticField, globalGeometry,muonDetIdAssociator_, prop, traj, track, pNavigator(), 1000.);
        processMuonResidualsFromTrack(muonResidualsFromTrack);
      }
    } // end loop over tracks
  }
  else
  {
    edm::Handle<reco::MuonCollection> muons;
    iEvent.getByLabel(m_muonCollectionTag, muons);

    for (reco::MuonCollection::const_iterator muon = muons->begin();  muon != muons->end();  ++muon)
    {
      if ( !(muon->isTrackerMuon() && muon->innerTrack().isNonnull() ) ) continue;

      if (m_minTrackPt < muon->pt()  &&  m_minTrackP < muon->p() && fabs(muon->innerTrack()->dxy(beamSpot->position())) < m_maxDxy)
      {
        MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, &(*muon), pNavigator(), 100.);
        processMuonResidualsFromTrack(muonResidualsFromTrack);
      }
    }
  }
}


void AlignmentMonitorSegmentDifferences::processMuonResidualsFromTrack(MuonResidualsFromTrack &mrft)
{
  if (mrft.trackerNumHits() < m_minTrackerHits) return;
  if (!m_allowTIDTEC  && mrft.contains_TIDTEC()) return;
  if (mrft.normalizedChi2() > m_maxTrackerRedChi2) return;

  int nMuChambers = 0;
  std::vector<DetId> chamberIds = mrft.chamberIds();
  for (unsigned ch=0; ch < chamberIds.size(); ch++)  if (chamberIds[ch].det() == DetId::Muon)  nMuChambers++;
  if (nMuChambers < m_minNCrossedChambers ) return;

  double qoverpt = (mrft.getTrack()->charge() > 0 ? 1. : -1.) / mrft.getTrack()->pt();
  double qoverpz = 0.;
  if (fabs(mrft.getTrack()->pz()) > 0.01) qoverpz = mrft.getTrack()->charge() / fabs(mrft.getTrack()->pz());

  for (std::vector<DetId>::const_iterator chamberId = chamberIds.begin();  chamberId != chamberIds.end();  ++chamberId)
  {
    if (chamberId->det() != DetId::Muon  ) continue;

    // **************** DT ****************
    if (m_doDT  &&  chamberId->subdetId() == MuonSubdetId::DT)
    {
      MuonChamberResidual *dt13 = mrft.chamberResidual(*chamberId, MuonChamberResidual::kDT13);
      MuonChamberResidual *dt2 = mrft.chamberResidual(*chamberId, MuonChamberResidual::kDT2);
      
      if (dt13 != NULL  &&  dt13->numHits() >= m_minDT13Hits)
      {
        DTChamberId thisid(chamberId->rawId());
        for (std::vector<DetId>::const_iterator otherId = chamberIds.begin();  otherId != chamberIds.end();  ++otherId)
        {
          if (otherId->det() == DetId::Muon  &&  otherId->subdetId() == MuonSubdetId::DT)
          {
            DTChamberId thatid(otherId->rawId());
            if (thisid.rawId() != thatid.rawId()  &&  thisid.wheel() == thatid.wheel()  &&  thisid.sector() == thatid.sector())
            {
              MuonChamberResidual *dt13other = mrft.chamberResidual(*otherId, MuonChamberResidual::kDT13);
              if (dt13other != NULL  &&  dt13other->numHits() >= m_minDT13Hits)
              {
                double slopediff = 1000. * (dt13->global_resslope() - dt13other->global_resslope());
                //double length = dt13->chamberAlignable()->surface().toGlobal(align::LocalPoint(0,0,0)).perp() -
                //                dt13other->chamberAlignable()->surface().toGlobal(align::LocalPoint(0,0,0)).perp();
                //double residdiff = 10. * (dt13->global_residual() + length*dt13->global_resslope() - dt13other->global_residual());
                double residdiff = 10. * (dt13->global_residual() - dt13other->global_residual());

                int st = 0;
                if (thatid.station() - thisid.station() == 1) st = thisid.station();
                if (st>0)
                {
                  m_dt13_resid[thisid.wheel()+2][thisid.sector()-1][st-1]->Fill(qoverpt, residdiff);
                  m_dt13_slope[thisid.wheel()+2][thisid.sector()-1][st-1]->Fill(qoverpt, slopediff);
                  if (qoverpt > 0)
                  {
                    m_posdt13_resid[thisid.wheel()+2][thisid.sector()-1][st-1]->Fill(residdiff);
                    m_posdt13_slope[thisid.wheel()+2][thisid.sector()-1][st-1]->Fill(slopediff);
                  }
                  else
                  {
                    m_negdt13_resid[thisid.wheel()+2][thisid.sector()-1][st-1]->Fill(residdiff);
                    m_negdt13_slope[thisid.wheel()+2][thisid.sector()-1][st-1]->Fill(slopediff);
                  }
                }
              } // end other numhits
            } // end this near other
          } // end other is DT

          // cross-system: other is CSC
          // only do it for DT stubs in W+-2 St1&2:
          if ( !(abs(thisid.wheel()) == 2 && (thisid.station() == 1 || thisid.station() == 2)) ) continue;
          if (otherId->det() == DetId::Muon  &&  otherId->subdetId() == MuonSubdetId::CSC)
          {
            CSCDetId thatid(otherId->rawId());
            //only do it for CSC stubs in St1R3 or St2R2:
            if ( !( (thatid.station()==1 && thatid.ring()==3) || (thatid.station()==2 && thatid.ring()==2) ) ) continue;

            MuonChamberResidual *cscother = mrft.chamberResidual(*otherId, MuonChamberResidual::kCSC);
            if (cscother != NULL  &&  cscother->numHits() >= m_minCSCHits)
            {
              // scale to adjust the csc residual size to be comparabe to dt's one
              double csc_scale = dt13->chamberAlignable()->surface().toGlobal(align::LocalPoint(dt13->trackx(), dt13->tracky(),0)).perp() /
                                 cscother->chamberAlignable()->surface().toGlobal(align::LocalPoint(cscother->trackx(), cscother->tracky(),0)).perp();
              double residdiff = 10. * (dt13->global_residual() - cscother->global_residual() * csc_scale);
              if (thisid.station() == 1 && thatid.station()==1)
              {
                if (qoverpt > 0) m_x_pos_dt1_csc1_resid[thatid.endcap()-1][thisid.sector()-1]->Fill(residdiff);
                else             m_x_neg_dt1_csc1_resid[thatid.endcap()-1][thisid.sector()-1]->Fill(residdiff);
              }
              else if (thisid.station() == 1 && thatid.station()==2)
              {
                if (qoverpt > 0) m_x_pos_dt1_csc2_resid[thatid.endcap()-1][thisid.sector()-1]->Fill(residdiff);
                else             m_x_neg_dt1_csc2_resid[thatid.endcap()-1][thisid.sector()-1]->Fill(residdiff);
              }
              else if (thisid.station() == 2 && thatid.station()==1)
              {
                if (qoverpt > 0) m_x_pos_dt2_csc1_resid[thatid.endcap()-1][thisid.sector()-1]->Fill(residdiff);
                else             m_x_neg_dt2_csc1_resid[thatid.endcap()-1][thisid.sector()-1]->Fill(residdiff);
              }
            }
          } // end other is CSC
        } // end loop over other
      } // end if DT13

      // z-direction
      if (dt2 != NULL  &&  dt2->numHits() >= m_minDT2Hits && (dt2->chi2() / double(dt2->ndof())) < 2.0)
      {
        DTChamberId thisid(chamberId->rawId());
        for (std::vector<DetId>::const_iterator otherId = chamberIds.begin();  otherId != chamberIds.end();  ++otherId)
        {
          if (otherId->det() == DetId::Muon  &&  otherId->subdetId() == MuonSubdetId::DT)
          {
            DTChamberId thatid(otherId->rawId());
            if (thisid.rawId() != thatid.rawId()  &&  thisid.wheel() == thatid.wheel()  &&  thisid.sector() == thatid.sector())
            {
              MuonChamberResidual *dt2other = mrft.chamberResidual(*otherId, MuonChamberResidual::kDT2);
              if (dt2other != NULL  &&  dt2other->numHits() >= m_minDT2Hits)
              {
                double slopediff = 1000. * (dt2->global_resslope() - dt2other->global_resslope());
                //double length = dt2->chamberAlignable()->surface().toGlobal(align::LocalPoint(0,0,0)).perp() -
                //                dt2other->chamberAlignable()->surface().toGlobal(align::LocalPoint(0,0,0)).perp();
                //double residdiff = 10. * (dt2->global_residual() + length*dt2->global_resslope() - dt2other->global_residual());
                double residdiff = 10. * (dt2->global_residual() - dt2other->global_residual());

                int st = 0;
                if (thatid.station() - thisid.station() == 1) st = thisid.station();
                if (st>0)
                {
                  m_dt2_resid[thisid.wheel()+2][thisid.sector()-1][st-1]->Fill(qoverpt, residdiff);
                  m_dt2_slope[thisid.wheel()+2][thisid.sector()-1][st-1]->Fill(qoverpt, slopediff);
                  if (qoverpt > 0)
                  {
                    m_posdt2_resid[thisid.wheel()+2][thisid.sector()-1][st-1]->Fill(residdiff);
                    m_posdt2_slope[thisid.wheel()+2][thisid.sector()-1][st-1]->Fill(slopediff);
                  }
                  else
                  {
                    m_negdt2_resid[thisid.wheel()+2][thisid.sector()-1][st-1]->Fill(residdiff);
                    m_negdt2_slope[thisid.wheel()+2][thisid.sector()-1][st-1]->Fill(slopediff);
                  }
                }
              } // end other numhits
            } // end this near other
          } // end other is DT
        } // end loop over other
      } // end if DT2
    } // end if DT

    // **************** CSC ****************
    else if (m_doCSC  &&  chamberId->subdetId() == MuonSubdetId::CSC)
    {
      MuonChamberResidual *csc = mrft.chamberResidual(*chamberId, MuonChamberResidual::kCSC);
      if (csc->numHits() >= m_minCSCHits)
      {
        CSCDetId thisid(chamberId->rawId());
        for (std::vector<DetId>::const_iterator otherId = chamberIds.begin();  otherId != chamberIds.end();  ++otherId)
        {
          if (otherId->det() == DetId::Muon  &&  otherId->subdetId() == MuonSubdetId::CSC)
          {
            CSCDetId thatid(otherId->rawId());
            if (thisid.rawId() != thatid.rawId()  &&  thisid.endcap() == thatid.endcap())
            {
              MuonChamberResidual *cscother = mrft.chamberResidual(*otherId, MuonChamberResidual::kCSC);
              if (cscother != NULL  &&  cscother->numHits() >= m_minCSCHits)
              {
                double slopediff = 1000. * (csc->global_resslope() - cscother->global_resslope());
                //double length = csc->chamberAlignable()->surface().toGlobal(align::LocalPoint(0,0,0)).z() -
                //                cscother->chamberAlignable()->surface().toGlobal(align::LocalPoint(0,0,0)).z();
                //double residdiff = 10. * (csc->global_residual() + length*csc->global_resslope() - cscother->global_residual());
                double residdiff = 10. * (csc->global_residual() - cscother->global_residual());

                int thischamber = thisid.chamber();
                int thisring = thisid.ring();
                if (thisid.station() == 1  &&  (thisring == 1  ||  thisring == 4))
                {
                  thischamber = (thischamber - 1) / 2 + 1;
                  thisring = 1;
                }

                if (thisring == thatid.ring()  &&  thischamber == thatid.chamber())
                {
                  bool inner = (thisring == 1);
                  bool outer = (thisring == 2);
                  int st = 0;
                  if (thatid.station() - thisid.station() == 1 && (inner || thisid.station()<3) ) st = thisid.station();

                  if (outer && st>0)
                  {
                    m_cscouter_resid[thisid.endcap()-1][thischamber-1][st-1]->Fill(qoverpz, residdiff);
                    m_cscouter_slope[thisid.endcap()-1][thischamber-1][st-1]->Fill(qoverpz, slopediff);
                    if (qoverpz > 0)
                    {
                      m_poscscouter_resid[thisid.endcap()-1][thischamber-1][st-1]->Fill(residdiff);
                      m_poscscouter_slope[thisid.endcap()-1][thischamber-1][st-1]->Fill(slopediff);
                    }
                    else
                    {
                      m_negcscouter_resid[thisid.endcap()-1][thischamber-1][st-1]->Fill(residdiff);
                      m_negcscouter_slope[thisid.endcap()-1][thischamber-1][st-1]->Fill(slopediff);
                    }
                  }
                  if (inner && st>0)
                  {
                    m_cscinner_resid[thisid.endcap()-1][thischamber-1][st-1]->Fill(qoverpz, residdiff);
                    m_cscinner_slope[thisid.endcap()-1][thischamber-1][st-1]->Fill(qoverpz, slopediff);
                    if (qoverpz > 0)
                    {
                      m_poscscinner_resid[thisid.endcap()-1][thischamber-1][st-1]->Fill(residdiff);
                      m_poscscinner_slope[thisid.endcap()-1][thischamber-1][st-1]->Fill(slopediff);
                    }
                    else
                    {
                      m_negcscinner_resid[thisid.endcap()-1][thischamber-1][st-1]->Fill(residdiff);
                      m_negcscinner_slope[thisid.endcap()-1][thischamber-1][st-1]->Fill(slopediff);
                    }
                  }
                } // end of same ring&chamber
              } // end other min numhits
            } // end this near other
          } // end other is CSC
        } // end loop over other
          
      } // end if this min numhits
    } // end if CSC

  } // end loop over chamberIds
}


DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorSegmentDifferences, "AlignmentMonitorSegmentDifferences");
