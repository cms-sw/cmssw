/*
 * Package:     CommonAlignmentProducer
 * Class  :     AlignmentMonitorMuonSystemMap1D
 *
 * Original Author:  Jim Pivarski
 *         Created:  Mon Nov 12 13:30:14 CST 2007
 *
 * $Id: AlignmentMonitorMuonSystemMap1D.cc,v 1.6 2011/10/12 22:59:47 khotilov Exp $
 */

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "TH1F.h"
#include "TH2F.h"


class AlignmentMonitorMuonSystemMap1D: public AlignmentMonitorBase
{
public:
  AlignmentMonitorMuonSystemMap1D(const edm::ParameterSet& cfg);
  virtual ~AlignmentMonitorMuonSystemMap1D() {}

  void book();

  void event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks);
  void processMuonResidualsFromTrack(MuonResidualsFromTrack &mrft, const edm::Event &iEvent);

  void afterAlignment(const edm::EventSetup &iSetup);

private:

  // parameters
  edm::InputTag m_muonCollectionTag;
  double m_minTrackPt;
  double m_maxTrackPt;
  double m_minTrackP;
  double m_maxTrackP;
  double m_maxDxy;
  int m_minTrackerHits;
  double m_maxTrackerRedChi2;
  bool m_allowTIDTEC;
  int m_minNCrossedChambers;
  int m_minDT13Hits;
  int m_minDT2Hits;
  int m_minCSCHits;
  bool m_doDT;
  bool m_doCSC;
  bool m_useStubPosition;
  bool m_createNtuple;

  // counter
  long m_counter_event;
  long m_counter_track;
  long m_counter_trackmoment;
  long m_counter_trackdxy;
  long m_counter_trackokay;
  long m_counter_dt;
  long m_counter_13numhits;
  long m_counter_2numhits;
  long m_counter_csc;
  long m_counter_cscnumhits;

  // histogram helper
  class MuonSystemMapPlot1D
  {
  public:
    MuonSystemMapPlot1D(std::string name, AlignmentMonitorMuonSystemMap1D *module, int bins, double low, double high, bool xy, bool add_1d);

    void fill_x_1d(double residx, double chi2, int dof);
    void fill_x(char charge, double abscissa, double residx, double chi2, int dof);
    void fill_y(char charge, double abscissa, double residy, double chi2, int dof);
    void fill_dxdz(char charge, double abscissa, double slopex, double chi2, int dof);
    void fill_dydz(char charge, double abscissa, double slopey, double chi2, int dof);

  private:
    std::string m_name;
    int m_bins;
    bool m_xy;
    bool m_1d;
    TH1F *m_x_1d;
    TH2F *m_x_2d, *m_y_2d, *m_dxdz_2d, *m_dydz_2d;
  };

  MuonSystemMapPlot1D *m_DTvsz_station[4][14]; // [station][sector]
  MuonSystemMapPlot1D *m_CSCvsr_me[2][4][36];  // [endcap][station][chamber]
  MuonSystemMapPlot1D *m_DTvsphi_station[4][5];// [station][wheel]
  MuonSystemMapPlot1D *m_CSCvsphi_me[2][4][3]; // [endcap][station][ring]

  std::vector<MuonSystemMapPlot1D*> m_plots;

  std::string num02d(int num);

  // optional debug ntuple
  TTree *m_cscnt;

  struct MyCSCDetId
  {
    void init(CSCDetId &id)
    {
      e = id.endcap();
      s = id.station();
      r = id.ring();
      c = id.chamber();
      t = id.iChamberType();
    }
    Short_t e, s, r, c;
    Short_t t; // type 1-10: ME1/a,1/b,1/2,1/3,2/1...4/2
  };
  MyCSCDetId m_id;

  struct MyTrack
  {
    Int_t q;
    Float_t pt, pz;
  };
  MyTrack m_tr;

  struct MyResidual
  {
    Float_t res, slope, rho, phi, z;
  };
  MyResidual m_re;

  UInt_t m_run;
};



AlignmentMonitorMuonSystemMap1D::AlignmentMonitorMuonSystemMap1D(const edm::ParameterSet& cfg)
   : AlignmentMonitorBase(cfg, "AlignmentMonitorMuonSystemMap1D")
   , m_muonCollectionTag(cfg.getParameter<edm::InputTag>("muonCollectionTag"))
   , m_minTrackPt(cfg.getParameter<double>("minTrackPt"))
   , m_maxTrackPt(cfg.getParameter<double>("maxTrackPt"))
   , m_minTrackP(cfg.getParameter<double>("minTrackP"))
   , m_maxTrackP(cfg.getParameter<double>("maxTrackP"))
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
   , m_useStubPosition(cfg.getParameter<bool>("useStubPosition"))
   , m_createNtuple(cfg.getParameter<bool>("createNtuple"))
{
  if (m_createNtuple)
  {
    edm::Service<TFileService> fs;
    m_cscnt = fs->make<TTree>("mualNtuple", "mualNtuple");
    m_cscnt->Branch("id", &m_id.e,"e/S:s:r:c:t");
    m_cscnt->Branch("tr", &m_tr.q,"q/I:pt/F:pz");
    m_cscnt->Branch("re", &m_re.res, "res/F:slope:rho:phi:z");
    m_cscnt->Branch("run", &m_run, "run/i");
  }
}


std::string AlignmentMonitorMuonSystemMap1D::num02d(int num)
{
  assert(num>=0 && num <100);
  char tmp[4];
  sprintf(tmp, "%02d", num);
  return std::string(tmp);
}


void AlignmentMonitorMuonSystemMap1D::book()
{
  std::string wheel_label[5]={"A","B","C","D","E"};

  for (int station = 1; station<=4; station++)
  {
    char c_station[4];
    sprintf(c_station, "%d", station);
    std::string s_station(c_station);
    
    bool do_y = true;
    if (station==4) do_y = false;

    // *** DT ***
    if (m_doDT) for (int sector = 1;  sector <= 14;  sector++)
    {
      if ((station<4 && sector <= 12) || station==4)
      {
	m_DTvsz_station[station-1][sector-1] = 
	  new MuonSystemMapPlot1D("DTvsz_st" + s_station + "sec" + num02d(sector), this, 60, -660., 660., do_y,false);
	m_plots.push_back(m_DTvsz_station[station-1][sector-1]);
      }
    }

    if (m_doDT) for (int wheel = -2;  wheel <= 2;  wheel++)
    {
      m_DTvsphi_station[station-1][wheel+2] = 
	new MuonSystemMapPlot1D("DTvsphi_st" + s_station + "wh" + wheel_label[wheel+2], this, 180, -M_PI, M_PI, do_y, false);
      m_plots.push_back(m_DTvsphi_station[station-1][wheel+2]);
    }

    // *** CSC ***
    if (m_doCSC) for (int endcap = 1;  endcap <= 2;  endcap++)
    {
      std::string s_endcap("m");
      if (endcap == 1) s_endcap = "p";

      for (int chamber = 1;  chamber <= 36;  chamber++)
      {
        m_CSCvsr_me[endcap-1][station-1][chamber-1] =
          new MuonSystemMapPlot1D("CSCvsr_me" + s_endcap + s_station +"ch" + num02d(chamber), this, 60, 100., 700., false, false);
        m_plots.push_back(m_CSCvsr_me[endcap-1][station-1][chamber-1]);
      }
      
      for (int ring = 1; ring <= 3; ring++) // the ME1/a (ring4) is not independent from ME1/b (ring1)
      {
	char c_ring[4];
	sprintf(c_ring, "%d", ring);
	std::string s_ring(c_ring);
	if ( (station>1 && ring<=2) || station==1)
        {
	  m_CSCvsphi_me[endcap-1][station-1][ring-1] = 
	    new MuonSystemMapPlot1D("CSCvsphi_me" + s_endcap + s_station + s_ring, this, 180, -M_PI/180.*5., M_PI*(2.-5./180.), false, true);
	  m_plots.push_back(m_CSCvsphi_me[endcap-1][station-1][ring-1]);
	}
      }
    } // endcaps
  } // stations

  m_counter_event = 0;
  m_counter_track = 0;
  m_counter_trackmoment = 0;
  m_counter_trackdxy = 0;
  m_counter_trackokay = 0;
  m_counter_dt = 0;
  m_counter_13numhits = 0;
  m_counter_2numhits = 0;
  m_counter_csc = 0;
  m_counter_cscnumhits = 0;
}


void AlignmentMonitorMuonSystemMap1D::event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& trajtracks)
{
  m_counter_event++;

  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByLabel(m_beamSpotTag, beamSpot);

  if (m_muonCollectionTag.label().empty()) // use trajectories
  {
    for (ConstTrajTrackPairCollection::const_iterator trajtrack = trajtracks.begin();  trajtrack != trajtracks.end();  ++trajtrack)
    {
      m_counter_track++;
      const Trajectory* traj = (*trajtrack).first;
      const reco::Track* track = (*trajtrack).second;

      if (m_minTrackPt < track->pt()  &&  track->pt() < m_maxTrackPt && m_minTrackP < track->p()  &&  track->p() < m_maxTrackP)
      {
        m_counter_trackmoment++;
        if ( fabs(track->dxy(beamSpot->position())) < m_maxDxy )
        {
          m_counter_trackdxy++;

          MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, traj, track, pNavigator(), 1000.);
          processMuonResidualsFromTrack(muonResidualsFromTrack, iEvent);
        }
      } // end if track has acceptable momentum
    } // end loop over tracks
  }
  else
  {
    edm::Handle<reco::MuonCollection> muons;
    iEvent.getByLabel(m_muonCollectionTag, muons);

    for (reco::MuonCollection::const_iterator muon = muons->begin();  muon != muons->end();  ++muon)
    {
      if ( !(muon->isTrackerMuon() && muon->innerTrack().isNonnull() ) ) continue;

      m_counter_track++;

      if (m_minTrackPt < muon->pt()  &&  muon->pt() < m_maxTrackPt && m_minTrackP < muon->p()  &&  muon->p() < m_maxTrackP )
      {
        m_counter_trackmoment++;
        if (fabs(muon->innerTrack()->dxy(beamSpot->position())) < m_maxDxy)
        {
          m_counter_trackdxy++;

          MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, &(*muon), pNavigator(), 100.);
          processMuonResidualsFromTrack(muonResidualsFromTrack, iEvent);
        }
      }
    }
  }
}


void AlignmentMonitorMuonSystemMap1D::processMuonResidualsFromTrack(MuonResidualsFromTrack &mrft, const edm::Event &iEvent)
{
  if (mrft.trackerNumHits() < m_minTrackerHits) return;
  if (!m_allowTIDTEC  && mrft.contains_TIDTEC()) return;
  if (mrft.normalizedChi2() > m_maxTrackerRedChi2) return;

  int nMuChambers = 0;
  std::vector<DetId> chamberIds = mrft.chamberIds();
  for (unsigned ch=0; ch < chamberIds.size(); ch++)  if (chamberIds[ch].det() == DetId::Muon)  nMuChambers++;
  if (nMuChambers < m_minNCrossedChambers ) return;

  char charge = (mrft.getTrack()->charge() > 0 ? 1 : -1);
  // double qoverpt = track->charge() / track->pt();
  // double qoverpz = track->charge() / track->pz();

  m_counter_trackokay++;

  for (std::vector<DetId>::const_iterator chamberId = chamberIds.begin();  chamberId != chamberIds.end();  ++chamberId)
  {
    if (chamberId->det() != DetId::Muon  ) continue;

    if (m_doDT  &&  chamberId->subdetId() == MuonSubdetId::DT)
    {
      MuonChamberResidual *dt13 = mrft.chamberResidual(*chamberId, MuonChamberResidual::kDT13);
      MuonChamberResidual *dt2 = mrft.chamberResidual(*chamberId, MuonChamberResidual::kDT2);
      DTChamberId id(chamberId->rawId());

      m_counter_dt++;

      if (id.station() < 4 && dt13 != NULL  &&  dt13->numHits() >= m_minDT13Hits && dt2 != NULL  &&  dt2->numHits() >= m_minDT2Hits)
      {
        m_counter_13numhits++;

        double residual = dt13->global_residual();
        double resslope = dt13->global_resslope();
        double chi2 = dt13->chi2();
        int dof = dt13->ndof();

        align::GlobalPoint gpos;
        if (m_useStubPosition) gpos = dt13->global_stubpos();
        else gpos = dt13->global_trackpos();
        double phi = atan2(gpos.y(), gpos.x());
        double z = gpos.z();

        assert(1 <= id.sector()  &&  id.sector() <= 14);

        m_DTvsz_station[id.station()-1][id.sector()-1]->fill_x(charge, z, residual, chi2, dof);
        m_DTvsz_station[id.station()-1][id.sector()-1]->fill_dxdz(charge, z, resslope, chi2, dof);
        m_DTvsphi_station[id.station()-1][id.wheel()+2]->fill_x(charge, phi, residual, chi2, dof);
        m_DTvsphi_station[id.station()-1][id.wheel()+2]->fill_dxdz(charge, phi, resslope, chi2, dof);

        m_counter_2numhits++;

        residual = dt2->global_residual();
        resslope = dt2->global_resslope();
        chi2 = dt2->chi2();
        dof = dt2->ndof();

        if (m_useStubPosition) gpos = dt2->global_stubpos();
        else gpos = dt2->global_trackpos();
        phi = atan2(gpos.y(), gpos.x());
        z = gpos.z();

        assert(1 <= id.sector()  &&  id.sector() <= 14);

        m_DTvsz_station[id.station()-1][id.sector()-1]->fill_y(charge, z, residual, chi2, dof);
        m_DTvsz_station[id.station()-1][id.sector()-1]->fill_dydz(charge, z, resslope, chi2, dof);
        m_DTvsphi_station[id.station()-1][id.wheel()+2]->fill_y(charge, phi, residual, chi2, dof);
        m_DTvsphi_station[id.station()-1][id.wheel()+2]->fill_dydz(charge, phi, resslope, chi2, dof);
      }

      if (id.station() == 4 && dt13 != NULL  &&  dt13->numHits() >= m_minDT13Hits)
      {
        m_counter_13numhits++;

        double residual = dt13->global_residual();
        double resslope = dt13->global_resslope();
        double chi2 = dt13->chi2();
        int dof = dt13->ndof();

        align::GlobalPoint gpos;
        if (m_useStubPosition) gpos = dt13->global_stubpos();
        else gpos = dt13->global_trackpos();
        double phi = atan2(gpos.y(), gpos.x());
        double z = gpos.z();

        assert(1 <= id.sector()  &&  id.sector() <= 14);

        m_DTvsz_station[id.station()-1][id.sector()-1]->fill_x(charge, z, residual, chi2, dof);
        m_DTvsz_station[id.station()-1][id.sector()-1]->fill_dxdz(charge, z, resslope, chi2, dof);
        m_DTvsphi_station[id.station()-1][id.wheel()+2]->fill_x(charge, phi, residual, chi2, dof);
        m_DTvsphi_station[id.station()-1][id.wheel()+2]->fill_dxdz(charge, phi, resslope, chi2, dof);
      }
    }

    else if (m_doCSC  &&  chamberId->subdetId() == MuonSubdetId::CSC)
    {
      MuonChamberResidual *csc = mrft.chamberResidual(*chamberId, MuonChamberResidual::kCSC);
      CSCDetId id(chamberId->rawId());

      int ring = id.ring();
      if (id.ring()==4) ring = 1; // combine ME1/a + ME1/b

      m_counter_csc++;

      if (csc != NULL  &&  csc->numHits() >= m_minCSCHits)
      {
        m_counter_cscnumhits++;

        double residual = csc->global_residual();
        double resslope = csc->global_resslope();
        double chi2 = csc->chi2();
        int dof = csc->ndof();

        align::GlobalPoint gpos;
        if (m_useStubPosition) gpos = csc->global_stubpos();
        else gpos = csc->global_trackpos();
        double phi = atan2(gpos.y(), gpos.x());
        // start phi from -5deg
        if (phi<-M_PI/180.*5.) phi += 2.*M_PI;
        double R = sqrt(pow(gpos.x(), 2) + pow(gpos.y(), 2));

        int chamber = id.chamber() - 1;
        if (id.station() > 1  &&  ring == 1) chamber *= 2;

        assert(1 <= id.endcap()  &&  id.endcap() <= 2  &&  0 <= chamber  &&  chamber <= 35);

        if (R>0.) m_CSCvsphi_me[id.endcap()-1][id.station()-1][ring-1]->fill_x_1d(residual/R, chi2, dof);

        m_CSCvsr_me[id.endcap()-1][id.station()-1][chamber]->fill_x(charge, R, residual, chi2, dof);
        m_CSCvsr_me[id.endcap()-1][id.station()-1][chamber]->fill_dxdz(charge, R, resslope, chi2, dof);
        m_CSCvsphi_me[id.endcap()-1][id.station()-1][ring-1]->fill_x(charge, phi, residual, chi2, dof);
        m_CSCvsphi_me[id.endcap()-1][id.station()-1][ring-1]->fill_dxdz(charge, phi, resslope, chi2, dof);

        if (m_createNtuple && chi2 > 0.)//  &&  TMath::Prob(chi2, dof) < 0.95)
        {
          m_id.init(id);
          m_tr.q = charge;
          m_tr.pt = mrft.getTrack()->pt();
          m_tr.pz = mrft.getTrack()->pz();
          m_re.res = residual;
          m_re.slope = resslope;
          m_re.rho = R;
          m_re.phi = phi;
          m_re.z = gpos.z();
          m_run = iEvent.id().run();
          m_cscnt->Fill();
        }

      }
    }

    //else { assert(false); }
  } // end loop over chambers
}


void AlignmentMonitorMuonSystemMap1D::afterAlignment(const edm::EventSetup &iSetup)
{
  std::cout << "AlignmentMonitorMuonSystemMap1D counters:"<<std::endl;
  std::cout << " monitor m_counter_event      = " << m_counter_event << std::endl;
  std::cout << " monitor m_counter_track      = " << m_counter_track << std::endl;
  std::cout << " monitor m_counter_trackppt   = " << m_counter_trackmoment << std::endl;
  std::cout << " monitor m_counter_trackdxy   = " << m_counter_trackdxy  << std::endl;
  std::cout << " monitor m_counter_trackokay  = " << m_counter_trackokay << std::endl;
  std::cout << " monitor m_counter_dt         = " << m_counter_dt << std::endl;
  std::cout << " monitor m_counter_13numhits  = " << m_counter_13numhits << std::endl;
  std::cout << " monitor m_counter_2numhits   = " << m_counter_2numhits << std::endl;
  std::cout << " monitor m_counter_csc        = " << m_counter_csc << std::endl;
  std::cout << " monitor m_counter_cscnumhits = " << m_counter_cscnumhits << std::endl;
}


AlignmentMonitorMuonSystemMap1D::MuonSystemMapPlot1D::MuonSystemMapPlot1D(std::string name, AlignmentMonitorMuonSystemMap1D *module, int bins, double low, double high, bool xy, bool add_1d)
   : m_name(name), m_bins(bins), m_xy(xy), m_1d(add_1d)
{
  m_x_2d = m_y_2d = m_dxdz_2d = m_dydz_2d = NULL;
  std::stringstream name_x_2d, name_y_2d, name_dxdz_2d, name_dydz_2d;
  name_x_2d << m_name << "_x_2d";
  name_y_2d << m_name << "_y_2d";
  name_dxdz_2d << m_name << "_dxdz_2d";
  name_dydz_2d << m_name << "_dydz_2d";

  const int nbins = 200;
  const double window = 100.;

  m_x_2d = module->book2D("/iterN/", name_x_2d.str().c_str(), "", m_bins, low, high, nbins, -window, window);
  if (m_xy) m_y_2d = module->book2D("/iterN/", name_y_2d.str().c_str(), "", m_bins, low, high, nbins, -window, window);
  m_dxdz_2d = module->book2D("/iterN/", name_dxdz_2d.str().c_str(), "", m_bins, low, high, nbins, -window, window);
  if (m_xy) m_dydz_2d = module->book2D("/iterN/", name_dydz_2d.str().c_str(), "", m_bins, low, high, nbins, -window, window);

  m_x_1d = NULL;
  if (m_1d) {
    std::stringstream name_x_1d;//, name_y_1d, name_dxdz_1d, name_dydz_1d;
    name_x_1d << m_name << "_x_1d";
    m_x_1d = module->book1D("/iterN/", name_x_1d.str().c_str(), "", nbins, -window, window);
  }
}


void AlignmentMonitorMuonSystemMap1D::MuonSystemMapPlot1D::fill_x_1d(double residx, double chi2, int dof)
{
  if (m_1d && chi2 > 0.) {
    // assume that residx was in radians
    double residual = residx * 1000.;
    m_x_1d->Fill(residual);
  }
}


void AlignmentMonitorMuonSystemMap1D::MuonSystemMapPlot1D::fill_x(char charge, double abscissa, double residx, double chi2, int dof)
{
  if (chi2 > 0.) {
    double residual = residx * 10.;
    //double weight = dof / chi2;
    m_x_2d->Fill(abscissa, residual);
  }
}


void AlignmentMonitorMuonSystemMap1D::MuonSystemMapPlot1D::fill_y(char charge, double abscissa, double residy, double chi2, int dof)
{
  if (m_xy  &&  chi2 > 0.) {
    double residual = residy * 10.;
    //double weight = dof / chi2;
    m_y_2d->Fill(abscissa, residual);
  }
}


void AlignmentMonitorMuonSystemMap1D::MuonSystemMapPlot1D::fill_dxdz(char charge, double abscissa, double slopex, double chi2, int dof)
{
  if (chi2 > 0.) {
    double residual = slopex * 1000.;
    //double weight = dof / chi2;
    m_dxdz_2d->Fill(abscissa, residual);
  }
}


void AlignmentMonitorMuonSystemMap1D::MuonSystemMapPlot1D::fill_dydz(char charge, double abscissa, double slopey, double chi2, int dof)
{
  if (m_xy  &&  chi2 > 0.) {
    double residual = slopey * 1000.;
    //double weight = dof / chi2;
    m_dydz_2d->Fill(abscissa, residual);
  }
}


DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorMuonSystemMap1D, "AlignmentMonitorMuonSystemMap1D");
