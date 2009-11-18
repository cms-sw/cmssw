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
//

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorMuonSystemMap1D.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// member functions
//

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
   , m_writeTemporaryFile(cfg.getParameter<std::string>("writeTemporaryFile"))
   , m_readTemporaryFiles(cfg.getParameter<std::vector<std::string> >("readTemporaryFiles"))
   , m_doFits(cfg.getParameter<bool>("doFits"))
   , m_minFitHits(cfg.getParameter<int>("minFitHits"))
{}

std::string AlignmentMonitorMuonSystemMap1D::num02d(int num) {
  int tens = num / 10;
  int ones = num % 10;
  
  std::string s_tens, s_ones;
  if (tens == 0) s_tens = std::string("0");
  if (tens == 1) s_tens = std::string("1");
  if (tens == 2) s_tens = std::string("2");
  if (tens == 3) s_tens = std::string("3");
  if (tens == 4) s_tens = std::string("4");
  if (tens == 5) s_tens = std::string("5");
  if (tens == 6) s_tens = std::string("6");
  if (tens == 7) s_tens = std::string("7");
  if (tens == 8) s_tens = std::string("8");
  if (tens == 9) s_tens = std::string("9");

  if (ones == 0) s_ones = std::string("0");
  if (ones == 1) s_ones = std::string("1");
  if (ones == 2) s_ones = std::string("2");
  if (ones == 3) s_ones = std::string("3");
  if (ones == 4) s_ones = std::string("4");
  if (ones == 5) s_ones = std::string("5");
  if (ones == 6) s_ones = std::string("6");
  if (ones == 7) s_ones = std::string("7");
  if (ones == 8) s_ones = std::string("8");
  if (ones == 9) s_ones = std::string("9");

  return s_tens + s_ones;
}

void AlignmentMonitorMuonSystemMap1D::book() {
  for (int sector = 1;  sector <= 14;  sector++) {
    if (sector <= 12) {
      m_DTvsz_station1[sector-1] = new MuonSystemMapPlot1D(std::string("DTvsz_st1sec") + num02d(sector), this, 60, -660., 660., m_minFitHits);  m_plots.push_back(m_DTvsz_station1[sector-1]);
      m_DTvsz_station2[sector-1] = new MuonSystemMapPlot1D(std::string("DTvsz_st2sec") + num02d(sector), this, 60, -660., 660., m_minFitHits);  m_plots.push_back(m_DTvsz_station2[sector-1]);
      m_DTvsz_station3[sector-1] = new MuonSystemMapPlot1D(std::string("DTvsz_st3sec") + num02d(sector), this, 60, -660., 660., m_minFitHits);  m_plots.push_back(m_DTvsz_station3[sector-1]);
    }
    m_DTvsz_station4[sector-1] = new MuonSystemMapPlot1D(std::string("DTvsz_st4sec") + num02d(sector), this, 60, -660., 660., m_minFitHits);  m_plots.push_back(m_DTvsz_station4[sector-1]);
  }

  for (int endcap = 1;  endcap <= 2;  endcap++) {
    for (int chamber = 1;  chamber <= 36;  chamber++) {
      m_CSCvsr_me1[endcap-1][chamber-1] = new MuonSystemMapPlot1D(std::string("CSCvsr_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("1ch") + num02d(chamber), this, 60, 100., 700., m_minFitHits);  m_plots.push_back(m_CSCvsr_me1[endcap-1][chamber-1]);
      m_CSCvsr_me2[endcap-1][chamber-1] = new MuonSystemMapPlot1D(std::string("CSCvsr_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("2ch") + num02d(chamber), this, 60, 100., 700., m_minFitHits);  m_plots.push_back(m_CSCvsr_me2[endcap-1][chamber-1]);
      m_CSCvsr_me3[endcap-1][chamber-1] = new MuonSystemMapPlot1D(std::string("CSCvsr_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("3ch") + num02d(chamber), this, 60, 100., 700., m_minFitHits);  m_plots.push_back(m_CSCvsr_me3[endcap-1][chamber-1]);
      m_CSCvsr_me4[endcap-1][chamber-1] = new MuonSystemMapPlot1D(std::string("CSCvsr_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("4ch") + num02d(chamber), this, 60, 100., 700., m_minFitHits);  m_plots.push_back(m_CSCvsr_me4[endcap-1][chamber-1]);
    }
  }

  for (int wheel = -2;  wheel <= 2;  wheel++) {
    std::string s_wheel;
    if (wheel == -2)      s_wheel = std::string("A");
    else if (wheel == -1) s_wheel = std::string("B");
    else if (wheel == 0)  s_wheel = std::string("C");
    else if (wheel == +1) s_wheel = std::string("D");
    else if (wheel == +2) s_wheel = std::string("E");

    m_DTvsphi_station1[wheel+2] = new MuonSystemMapPlot1D(std::string("DTvsphi_st1wh") + s_wheel, this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_DTvsphi_station1[wheel+2]);
    m_DTvsphi_station2[wheel+2] = new MuonSystemMapPlot1D(std::string("DTvsphi_st2wh") + s_wheel, this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_DTvsphi_station2[wheel+2]);
    m_DTvsphi_station3[wheel+2] = new MuonSystemMapPlot1D(std::string("DTvsphi_st3wh") + s_wheel, this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_DTvsphi_station3[wheel+2]);
    m_DTvsphi_station4[wheel+2] = new MuonSystemMapPlot1D(std::string("DTvsphi_st4wh") + s_wheel, this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_DTvsphi_station4[wheel+2]);
  }

  for (int endcap = 1;  endcap <= 2;  endcap++) {
    m_CSCvsphi_me11[endcap-1] = new MuonSystemMapPlot1D(std::string("CSCvsphi_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("11"), this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_CSCvsphi_me11[endcap-1]);
    m_CSCvsphi_me12[endcap-1] = new MuonSystemMapPlot1D(std::string("CSCvsphi_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("12"), this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_CSCvsphi_me12[endcap-1]);
    m_CSCvsphi_me13[endcap-1] = new MuonSystemMapPlot1D(std::string("CSCvsphi_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("13"), this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_CSCvsphi_me13[endcap-1]);
    m_CSCvsphi_me14[endcap-1] = new MuonSystemMapPlot1D(std::string("CSCvsphi_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("14"), this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_CSCvsphi_me14[endcap-1]);
    m_CSCvsphi_me21[endcap-1] = new MuonSystemMapPlot1D(std::string("CSCvsphi_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("21"), this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_CSCvsphi_me21[endcap-1]);
    m_CSCvsphi_me22[endcap-1] = new MuonSystemMapPlot1D(std::string("CSCvsphi_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("22"), this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_CSCvsphi_me22[endcap-1]);
    m_CSCvsphi_me31[endcap-1] = new MuonSystemMapPlot1D(std::string("CSCvsphi_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("31"), this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_CSCvsphi_me31[endcap-1]);
    m_CSCvsphi_me32[endcap-1] = new MuonSystemMapPlot1D(std::string("CSCvsphi_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("32"), this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_CSCvsphi_me32[endcap-1]);
    m_CSCvsphi_me41[endcap-1] = new MuonSystemMapPlot1D(std::string("CSCvsphi_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("41"), this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_CSCvsphi_me41[endcap-1]);
    m_CSCvsphi_me42[endcap-1] = new MuonSystemMapPlot1D(std::string("CSCvsphi_me") + (endcap == 1 ? std::string("p") : std::string("m")) + std::string("42"), this, 180, -M_PI, M_PI, m_minFitHits);  m_plots.push_back(m_CSCvsphi_me42[endcap-1]);
  }
}

void AlignmentMonitorMuonSystemMap1D::event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& trajtracks) {
  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

  for (ConstTrajTrackPairCollection::const_iterator trajtrack = trajtracks.begin();  trajtrack != trajtracks.end();  ++trajtrack) {
    const Trajectory* traj = (*trajtrack).first;
    const reco::Track* track = (*trajtrack).second;

    if (m_minTrackPt < track->pt()  &&  track->pt() < m_maxTrackPt) {
      char charge = (track->charge() > 0 ? 1 : -1);
      // double qoverpt = track->charge() / track->pt();
      // double qoverpz = track->charge() / track->pz();
      MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, traj, pNavigator(), 1000.);

      if (muonResidualsFromTrack.trackerNumHits() >= m_minTrackerHits  &&  muonResidualsFromTrack.trackerRedChi2() < m_maxTrackerRedChi2  &&  (m_allowTIDTEC  ||  !muonResidualsFromTrack.contains_TIDTEC())) {
	std::vector<DetId> chamberIds = muonResidualsFromTrack.chamberIds();

	for (std::vector<DetId>::const_iterator chamberId = chamberIds.begin();  chamberId != chamberIds.end();  ++chamberId) {

	  if (chamberId->det() == DetId::Muon  &&  chamberId->subdetId() == MuonSubdetId::DT) {
	    MuonChamberResidual *dt13 = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kDT13);
	    MuonChamberResidual *dt2 = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kDT2);
	    DTChamberId id(chamberId->rawId());

	    if (dt13 != NULL  &&  dt13->numHits() >= m_minDT13Hits) {
	      double residual = dt13->global_residual();
	      double resslope = dt13->global_resslope();
	      double redchi2 = dt13->chi2() / double(dt13->ndof());

	      GlobalPoint trackpos = dt13->global_trackpos();
	      double phi = atan2(trackpos.y(), trackpos.x());
	      double z = trackpos.z();

	      assert(1 <= id.sector()  &&  id.sector() <= 14);

	      if (id.station() == 1) m_DTvsz_station1[id.sector()-1]->fill_x(charge, z, residual, redchi2);
	      if (id.station() == 2) m_DTvsz_station2[id.sector()-1]->fill_x(charge, z, residual, redchi2);
	      if (id.station() == 3) m_DTvsz_station3[id.sector()-1]->fill_x(charge, z, residual, redchi2);
	      if (id.station() == 4) m_DTvsz_station4[id.sector()-1]->fill_x(charge, z, residual, redchi2);

	      if (id.station() == 1) m_DTvsz_station1[id.sector()-1]->fill_dxdz(charge, z, resslope, redchi2);
	      if (id.station() == 2) m_DTvsz_station2[id.sector()-1]->fill_dxdz(charge, z, resslope, redchi2);
	      if (id.station() == 3) m_DTvsz_station3[id.sector()-1]->fill_dxdz(charge, z, resslope, redchi2);
	      if (id.station() == 4) m_DTvsz_station4[id.sector()-1]->fill_dxdz(charge, z, resslope, redchi2);

	      if (id.station() == 1) m_DTvsphi_station1[id.wheel()+2]->fill_x(charge, phi, residual, redchi2);
	      if (id.station() == 2) m_DTvsphi_station2[id.wheel()+2]->fill_x(charge, phi, residual, redchi2);
	      if (id.station() == 3) m_DTvsphi_station3[id.wheel()+2]->fill_x(charge, phi, residual, redchi2);
	      if (id.station() == 4) m_DTvsphi_station4[id.wheel()+2]->fill_x(charge, phi, residual, redchi2);

	      if (id.station() == 1) m_DTvsphi_station1[id.wheel()+2]->fill_dxdz(charge, phi, resslope, redchi2);
	      if (id.station() == 2) m_DTvsphi_station2[id.wheel()+2]->fill_dxdz(charge, phi, resslope, redchi2);
	      if (id.station() == 3) m_DTvsphi_station3[id.wheel()+2]->fill_dxdz(charge, phi, resslope, redchi2);
	      if (id.station() == 4) m_DTvsphi_station4[id.wheel()+2]->fill_dxdz(charge, phi, resslope, redchi2);
	    }

	    if (dt2 != NULL  &&  dt2->numHits() >= m_minDT2Hits) {
	      double residual = dt2->global_residual();
	      double resslope = dt2->global_resslope();
	      double redchi2 = dt2->chi2() / double(dt2->ndof());

	      GlobalPoint trackpos = dt2->global_trackpos();
	      double phi = atan2(trackpos.y(), trackpos.x());
	      double z = trackpos.z();

	      assert(1 <= id.sector()  &&  id.sector() <= 14);

	      if (id.station() == 1) m_DTvsz_station1[id.sector()-1]->fill_y(charge, z, residual, redchi2);
	      if (id.station() == 2) m_DTvsz_station2[id.sector()-1]->fill_y(charge, z, residual, redchi2);
	      if (id.station() == 3) m_DTvsz_station3[id.sector()-1]->fill_y(charge, z, residual, redchi2);
	      if (id.station() == 4) m_DTvsz_station4[id.sector()-1]->fill_y(charge, z, residual, redchi2);

	      if (id.station() == 1) m_DTvsz_station1[id.sector()-1]->fill_dydz(charge, z, resslope, redchi2);
	      if (id.station() == 2) m_DTvsz_station2[id.sector()-1]->fill_dydz(charge, z, resslope, redchi2);
	      if (id.station() == 3) m_DTvsz_station3[id.sector()-1]->fill_dydz(charge, z, resslope, redchi2);
	      if (id.station() == 4) m_DTvsz_station4[id.sector()-1]->fill_dydz(charge, z, resslope, redchi2);

	      if (id.station() == 1) m_DTvsphi_station1[id.wheel()+2]->fill_y(charge, phi, residual, redchi2);
	      if (id.station() == 2) m_DTvsphi_station2[id.wheel()+2]->fill_y(charge, phi, residual, redchi2);
	      if (id.station() == 3) m_DTvsphi_station3[id.wheel()+2]->fill_y(charge, phi, residual, redchi2);
	      if (id.station() == 4) m_DTvsphi_station4[id.wheel()+2]->fill_y(charge, phi, residual, redchi2);

	      if (id.station() == 1) m_DTvsphi_station1[id.wheel()+2]->fill_dydz(charge, phi, resslope, redchi2);
	      if (id.station() == 2) m_DTvsphi_station2[id.wheel()+2]->fill_dydz(charge, phi, resslope, redchi2);
	      if (id.station() == 3) m_DTvsphi_station3[id.wheel()+2]->fill_dydz(charge, phi, resslope, redchi2);
	      if (id.station() == 4) m_DTvsphi_station4[id.wheel()+2]->fill_dydz(charge, phi, resslope, redchi2);
	    }
	  }

	  else if (chamberId->det() == DetId::Muon  &&  chamberId->subdetId() == MuonSubdetId::CSC) {
	    MuonChamberResidual *csc = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kCSC);
	    CSCDetId id(chamberId->rawId());

	    if (csc != NULL  &&  csc->numHits() >= m_minCSCHits) {
	      double residual = csc->global_residual();
	      double resslope = csc->global_resslope();
	      double redchi2 = csc->chi2() / double(csc->ndof());

	      GlobalPoint trackpos = csc->global_trackpos();
	      double phi = atan2(trackpos.y(), trackpos.x());
	      double R = sqrt(pow(trackpos.x(), 2) + pow(trackpos.y(), 2));

	      int chamber = id.chamber() - 1;
	      if (id.station() > 1  &&  id.ring() == 1) chamber *= 2;

	      assert(1 <= id.endcap()  &&  id.endcap() <= 2  &&  0 <= chamber  &&  chamber <= 35);

	      if (id.station() == 1) m_CSCvsr_me1[id.endcap()-1][chamber]->fill_x(charge, R, residual, redchi2);
	      if (id.station() == 2) m_CSCvsr_me2[id.endcap()-1][chamber]->fill_x(charge, R, residual, redchi2);
	      if (id.station() == 3) m_CSCvsr_me3[id.endcap()-1][chamber]->fill_x(charge, R, residual, redchi2);
	      if (id.station() == 4) m_CSCvsr_me4[id.endcap()-1][chamber]->fill_x(charge, R, residual, redchi2);

	      if (id.station() == 1) m_CSCvsr_me1[id.endcap()-1][chamber]->fill_dxdz(charge, R, resslope, redchi2);
	      if (id.station() == 2) m_CSCvsr_me2[id.endcap()-1][chamber]->fill_dxdz(charge, R, resslope, redchi2);
	      if (id.station() == 3) m_CSCvsr_me3[id.endcap()-1][chamber]->fill_dxdz(charge, R, resslope, redchi2);
	      if (id.station() == 4) m_CSCvsr_me4[id.endcap()-1][chamber]->fill_dxdz(charge, R, resslope, redchi2);

	      if (id.station() == 1  &&  id.ring() == 1) m_CSCvsphi_me11[id.endcap()-1]->fill_x(charge, R, residual, redchi2);
	      if (id.station() == 1  &&  id.ring() == 2) m_CSCvsphi_me12[id.endcap()-1]->fill_x(charge, R, residual, redchi2);
	      if (id.station() == 1  &&  id.ring() == 3) m_CSCvsphi_me13[id.endcap()-1]->fill_x(charge, R, residual, redchi2);
	      if (id.station() == 1  &&  id.ring() == 4) m_CSCvsphi_me14[id.endcap()-1]->fill_x(charge, R, residual, redchi2);
	      if (id.station() == 2  &&  id.ring() == 1) m_CSCvsphi_me21[id.endcap()-1]->fill_x(charge, R, residual, redchi2);
	      if (id.station() == 2  &&  id.ring() == 2) m_CSCvsphi_me22[id.endcap()-1]->fill_x(charge, R, residual, redchi2);
	      if (id.station() == 3  &&  id.ring() == 1) m_CSCvsphi_me31[id.endcap()-1]->fill_x(charge, R, residual, redchi2);
	      if (id.station() == 3  &&  id.ring() == 2) m_CSCvsphi_me32[id.endcap()-1]->fill_x(charge, R, residual, redchi2);
	      if (id.station() == 4  &&  id.ring() == 1) m_CSCvsphi_me41[id.endcap()-1]->fill_x(charge, R, residual, redchi2);
	      if (id.station() == 4  &&  id.ring() == 2) m_CSCvsphi_me42[id.endcap()-1]->fill_x(charge, R, residual, redchi2);

	      if (id.station() == 1  &&  id.ring() == 1) m_CSCvsphi_me11[id.endcap()-1]->fill_dxdz(charge, phi, resslope, redchi2);
	      if (id.station() == 1  &&  id.ring() == 2) m_CSCvsphi_me12[id.endcap()-1]->fill_dxdz(charge, phi, resslope, redchi2);
	      if (id.station() == 1  &&  id.ring() == 3) m_CSCvsphi_me13[id.endcap()-1]->fill_dxdz(charge, phi, resslope, redchi2);
	      if (id.station() == 1  &&  id.ring() == 4) m_CSCvsphi_me14[id.endcap()-1]->fill_dxdz(charge, phi, resslope, redchi2);
	      if (id.station() == 2  &&  id.ring() == 1) m_CSCvsphi_me21[id.endcap()-1]->fill_dxdz(charge, phi, resslope, redchi2);
	      if (id.station() == 2  &&  id.ring() == 2) m_CSCvsphi_me22[id.endcap()-1]->fill_dxdz(charge, phi, resslope, redchi2);
	      if (id.station() == 3  &&  id.ring() == 1) m_CSCvsphi_me31[id.endcap()-1]->fill_dxdz(charge, phi, resslope, redchi2);
	      if (id.station() == 3  &&  id.ring() == 2) m_CSCvsphi_me32[id.endcap()-1]->fill_dxdz(charge, phi, resslope, redchi2);
	      if (id.station() == 4  &&  id.ring() == 1) m_CSCvsphi_me41[id.endcap()-1]->fill_dxdz(charge, phi, resslope, redchi2);
	      if (id.station() == 4  &&  id.ring() == 2) m_CSCvsphi_me42[id.endcap()-1]->fill_dxdz(charge, phi, resslope, redchi2);
	    }
	  }

	  else { assert(false); }

	} // end loop over chambers
      } // end if track has enough tracker hits
    } // end if track has acceptable momentum
  } // end loop over tracks
}

void AlignmentMonitorMuonSystemMap1D::afterAlignment(const edm::EventSetup &iSetup) {
  // collect temporary files
  if (m_readTemporaryFiles.size() != 0) {
    for (std::vector<std::string>::const_iterator fileName = m_readTemporaryFiles.begin();  fileName != m_readTemporaryFiles.end();  ++fileName) {
      FILE *file;
      int size;
      file = fopen(fileName->c_str(), "r");
      fread(&size, sizeof(int), 1, file);
      if (int(m_plots.size()) != size) throw cms::Exception("AlignmentMonitorMuonSystemMap1D") << "file \"" << *fileName << "\" has " << size << " plots, but this job has " << m_plots.size() << " plots (probably corresponds to the wrong job)" << std::endl;

      int which = 0;
      for (std::vector<MuonSystemMapPlot1D*>::const_iterator plot = m_plots.begin();  plot != m_plots.end();  ++plot) {
	(*plot)->read(file, which);
      }

      fclose(file);
    }
  }

  if (m_doFits) {
    for (std::vector<MuonSystemMapPlot1D*>::const_iterator plot = m_plots.begin();  plot != m_plots.end();  ++plot) {
      (*plot)->fit();
    }
  }

  // always do this
  for (std::vector<MuonSystemMapPlot1D*>::const_iterator plot = m_plots.begin();  plot != m_plots.end();  ++plot) {
    (*plot)->fill_profs();
  }

  // write out the pseudontuples for a later job to collect
  if (m_writeTemporaryFile != std::string("")) {
    FILE *file;
    file = fopen(m_writeTemporaryFile.c_str(), "w");
    int size = m_plots.size();
    fwrite(&size, sizeof(int), 1, file);

    int which = 0;
    for (std::vector<MuonSystemMapPlot1D*>::const_iterator plot = m_plots.begin();  plot != m_plots.end();  ++plot) {
      (*plot)->write(file, which);
    }

    fclose(file);
  }
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
// const member functions
//

//
// static member functions
//

//
// SEAL definitions
//

DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorMuonSystemMap1D, "AlignmentMonitorMuonSystemMap1D");
