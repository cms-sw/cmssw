// -*- C++ -*-
//
// Package:     CommonAlignmentProducer
// Class  :     AlignmentMonitorMuonSystemMap1D
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Mon Nov 12 13:30:14 CST 2007
// $Id$

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorMuonSystemMap1D.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"


AlignmentMonitorMuonSystemMap1D::AlignmentMonitorMuonSystemMap1D(const edm::ParameterSet& cfg)
   : AlignmentMonitorBase(cfg, "AlignmentMonitorMuonSystemMap1D")
   , m_minTrackPt(cfg.getParameter<double>("minTrackPt"))
   , m_maxTrackPt(cfg.getParameter<double>("maxTrackPt"))
   , m_minTrackerHits(cfg.getParameter<int>("minTrackerHits"))
   , m_maxTrackerRedChi2(cfg.getParameter<double>("maxTrackerRedChi2"))
   , m_allowTIDTEC(cfg.getParameter<bool>("allowTIDTEC"))
   , m_minDT13Hits(cfg.getParameter<int>("minDT13Hits"))
   , m_minDT2Hits(cfg.getParameter<int>("minDT2Hits"))
   , m_minCSCHits(cfg.getParameter<int>("minCSCHits"))
   , m_doDT(cfg.getParameter<bool>("doDT"))
   , m_doCSC(cfg.getParameter<bool>("doCSC"))
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
  m_counter_trackpt = 0;
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

  for (ConstTrajTrackPairCollection::const_iterator trajtrack = trajtracks.begin();  trajtrack != trajtracks.end();  ++trajtrack)
  {
    const Trajectory* traj = (*trajtrack).first;
    const reco::Track* track = (*trajtrack).second;

    m_counter_track++;

    if (m_minTrackPt < track->pt()  &&  track->pt() < m_maxTrackPt)
    {
      char charge = (track->charge() > 0 ? 1 : -1);
      // double qoverpt = track->charge() / track->pt();
      // double qoverpz = track->charge() / track->pz();
      MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, traj, pNavigator(), 1000.);

      m_counter_trackpt++;

      if (muonResidualsFromTrack.trackerNumHits() >= m_minTrackerHits  &&
          muonResidualsFromTrack.trackerRedChi2() < m_maxTrackerRedChi2  &&
          (m_allowTIDTEC  ||  !muonResidualsFromTrack.contains_TIDTEC()) ) 
      {
	std::vector<DetId> chamberIds = muonResidualsFromTrack.chamberIds();

	m_counter_trackokay++;

	for (std::vector<DetId>::const_iterator chamberId = chamberIds.begin();  chamberId != chamberIds.end();  ++chamberId)
        {
          if (m_doDT && chamberId->det() == DetId::Muon  &&  chamberId->subdetId() == MuonSubdetId::DT)
          {
	    MuonChamberResidual *dt13 = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kDT13);
	    MuonChamberResidual *dt2 = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kDT2);
	    DTChamberId id(chamberId->rawId());

	    m_counter_dt++;

            if (dt13 != NULL  &&  dt13->numHits() >= m_minDT13Hits)
            {
	       m_counter_13numhits++;

	      double residual = dt13->global_residual();
	      double resslope = dt13->global_resslope();
	      double chi2 = dt13->chi2();
	      int dof = dt13->ndof();

	      GlobalPoint trackpos = dt13->global_trackpos();
	      double phi = atan2(trackpos.y(), trackpos.x());
	      double z = trackpos.z();

	      assert(1 <= id.sector()  &&  id.sector() <= 14);

              m_DTvsz_station[id.station()-1][id.sector()-1]->fill_x(charge, z, residual, chi2, dof);
              m_DTvsz_station[id.station()-1][id.sector()-1]->fill_dxdz(charge, z, resslope, chi2, dof);
              m_DTvsphi_station[id.station()-1][id.wheel()+2]->fill_x(charge, phi, residual, chi2, dof);
              m_DTvsphi_station[id.station()-1][id.wheel()+2]->fill_dxdz(charge, phi, resslope, chi2, dof);
	    }

	    if (dt2 != NULL  &&  dt2->numHits() >= m_minDT2Hits)
            {
              m_counter_2numhits++;

	      double residual = dt2->global_residual();
	      double resslope = dt2->global_resslope();
	      double chi2 = dt2->chi2();
	      int dof = dt2->ndof();

	      GlobalPoint trackpos = dt2->global_trackpos();
	      double phi = atan2(trackpos.y(), trackpos.x());
	      double z = trackpos.z();

	      assert(1 <= id.sector()  &&  id.sector() <= 14);

              m_DTvsz_station[id.station()-1][id.sector()-1]->fill_y(charge, z, residual, chi2, dof);
              m_DTvsz_station[id.station()-1][id.sector()-1]->fill_dydz(charge, z, resslope, chi2, dof);
              m_DTvsphi_station[id.station()-1][id.wheel()+2]->fill_y(charge, phi, residual, chi2, dof);
              m_DTvsphi_station[id.station()-1][id.wheel()+2]->fill_dydz(charge, phi, resslope, chi2, dof);
	    }
	  }

          else if (m_doCSC && chamberId->det() == DetId::Muon  &&  chamberId->subdetId() == MuonSubdetId::CSC)
          {
	    MuonChamberResidual *csc = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kCSC);
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

	      GlobalPoint trackpos = csc->global_trackpos();
	      double phi = atan2(trackpos.y(), trackpos.x());
              // start phi from -5deg
              if (phi<-M_PI/180.*5.) phi += 2.*M_PI;
	      double R = sqrt(pow(trackpos.x(), 2) + pow(trackpos.y(), 2));

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
		m_tr.pt = track->pt();
		m_tr.pz = track->pz();
		m_re.res = residual;
		m_re.slope = resslope;
		m_re.rho = R;
		m_re.phi = phi;
		m_re.z = trackpos.z();
		m_run = iEvent.id().run();
		m_cscnt->Fill();
	      }

	    }
	  }

	  //else { assert(false); }

	} // end loop over chambers
      } // end if track has enough tracker hits
    } // end if track has acceptable momentum
  } // end loop over tracks
}


void AlignmentMonitorMuonSystemMap1D::afterAlignment(const edm::EventSetup &iSetup)
{
  std::cout << "AlignmentMonitorMuonSystemMap1D counters:"<<std::endl;
  std::cout << " monitor m_counter_event      = " << m_counter_event << std::endl;
  std::cout << " monitor m_counter_track      = " << m_counter_track << std::endl;
  std::cout << " monitor m_counter_trackpt    = " << m_counter_trackpt << std::endl;
  std::cout << " monitor m_counter_trackokay  = " << m_counter_trackokay << std::endl;
  std::cout << " monitor m_counter_dt         = " << m_counter_dt << std::endl;
  std::cout << " monitor m_counter_13numhits  = " << m_counter_13numhits << std::endl;
  std::cout << " monitor m_counter_2numhits   = " << m_counter_2numhits << std::endl;
  std::cout << " monitor m_counter_csc        = " << m_counter_csc << std::endl;
  std::cout << " monitor m_counter_cscnumhits = " << m_counter_cscnumhits << std::endl;
}

//
// constructors and destructor
//

// AlignmentMonitorMuonSystemMap1D::AlignmentMonitorMuonSystemMap1D(const AlignmentMonitorMuonSystemMap1D& rhs)
// {
//    // do actual copying here;
// }

//
// assignment operators
//
// const AlignmentMonitorMuonSystemMap1D& AlignmentMonitorMuonSystemMap1D::operator=(const AlignmentMonitorMuonSystemMap1D& rhs)
// {
//   //An exception safe implementation is
//   AlignmentMonitorMuonSystemMap1D temp(rhs);
//   swap(rhs);
//
//   return *this;
// }


//
// SEAL definitions
//

DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorMuonSystemMap1D, "AlignmentMonitorMuonSystemMap1D");
