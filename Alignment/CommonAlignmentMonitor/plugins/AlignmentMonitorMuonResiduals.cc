// -*- C++ -*-
//
// Package:     CommonAlignmentProducer
// Class  :     AlignmentMonitorMuonResiduals
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Mon Nov 12 13:30:14 CST 2007
//

// system include files
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "TH1F.h"
#include "TTree.h"

// user include files

// 
// class definition
// 

class AlignmentMonitorMuonResiduals: public AlignmentMonitorBase {
   public:
      AlignmentMonitorMuonResiduals(const edm::ParameterSet& cfg);
      ~AlignmentMonitorMuonResiduals() {};

      void book() override;
      void event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks) override;
      void afterAlignment() override;

   private:
      std::map<int, int> m_numx;
      std::map<int, double> m_x_w;
      std::map<int, double> m_x_ww;
      std::map<int, double> m_x_wx;
      std::map<int, double> m_x_wxx;
      std::map<int, int> m_numy;
      std::map<int, double> m_y_w;
      std::map<int, double> m_y_ww;
      std::map<int, double> m_y_wy;
      std::map<int, double> m_y_wyy;

      TH1F *m_sumnumx, *m_sumnumy, *m_xsummary, *m_ysummary;

      TH1F *m_xresid, *m_xresid_mb, *m_xresid_me,
	 *m_xresid_mb1, *m_xresid_mb2, *m_xresid_mb3, *m_xresid_mb4,
	 *m_xresid_minus2, *m_xresid_minus1, *m_xresid_zero, *m_xresid_plus1, *m_xresid_plus2,
	 *m_xresid_mep11, *m_xresid_mep12, *m_xresid_mep13, *m_xresid_mep14,
	 *m_xresid_mep21, *m_xresid_mep22, *m_xresid_mep31, *m_xresid_mep32, *m_xresid_mep41,
	 *m_xresid_mem11, *m_xresid_mem12, *m_xresid_mem13, *m_xresid_mem14,
	 *m_xresid_mem21, *m_xresid_mem22, *m_xresid_mem31, *m_xresid_mem32, *m_xresid_mem41,
	 *m_xresid_me11, *m_xresid_me12, *m_xresid_me13, *m_xresid_me14,
	 *m_xresid_me21, *m_xresid_me22, *m_xresid_me31, *m_xresid_me32, *m_xresid_me41;

      TH1F *m_xmean, *m_xmean_mb, *m_xmean_me,
	 *m_xmean_mb1, *m_xmean_mb2, *m_xmean_mb3, *m_xmean_mb4,
	 *m_xmean_minus2, *m_xmean_minus1, *m_xmean_zero, *m_xmean_plus1, *m_xmean_plus2,
	 *m_xmean_mep11, *m_xmean_mep12, *m_xmean_mep13, *m_xmean_mep14,
	 *m_xmean_mep21, *m_xmean_mep22, *m_xmean_mep31, *m_xmean_mep32, *m_xmean_mep41,
	 *m_xmean_mem11, *m_xmean_mem12, *m_xmean_mem13, *m_xmean_mem14,
	 *m_xmean_mem21, *m_xmean_mem22, *m_xmean_mem31, *m_xmean_mem32, *m_xmean_mem41,
	 *m_xmean_me11, *m_xmean_me12, *m_xmean_me13, *m_xmean_me14,
	 *m_xmean_me21, *m_xmean_me22, *m_xmean_me31, *m_xmean_me32, *m_xmean_me41;

      TH1F *m_xstdev, *m_xstdev_mb, *m_xstdev_me,
	 *m_xstdev_mb1, *m_xstdev_mb2, *m_xstdev_mb3, *m_xstdev_mb4,
	 *m_xstdev_minus2, *m_xstdev_minus1, *m_xstdev_zero, *m_xstdev_plus1, *m_xstdev_plus2,
	 *m_xstdev_mep11, *m_xstdev_mep12, *m_xstdev_mep13, *m_xstdev_mep14,
	 *m_xstdev_mep21, *m_xstdev_mep22, *m_xstdev_mep31, *m_xstdev_mep32, *m_xstdev_mep41,
	 *m_xstdev_mem11, *m_xstdev_mem12, *m_xstdev_mem13, *m_xstdev_mem14,
	 *m_xstdev_mem21, *m_xstdev_mem22, *m_xstdev_mem31, *m_xstdev_mem32, *m_xstdev_mem41,
	 *m_xstdev_me11, *m_xstdev_me12, *m_xstdev_me13, *m_xstdev_me14,
	 *m_xstdev_me21, *m_xstdev_me22, *m_xstdev_me31, *m_xstdev_me32, *m_xstdev_me41;

      TH1F *m_xerronmean, *m_xerronmean_mb, *m_xerronmean_me,
	 *m_xerronmean_mb1, *m_xerronmean_mb2, *m_xerronmean_mb3, *m_xerronmean_mb4,
	 *m_xerronmean_minus2, *m_xerronmean_minus1, *m_xerronmean_zero, *m_xerronmean_plus1, *m_xerronmean_plus2,
	 *m_xerronmean_mep11, *m_xerronmean_mep12, *m_xerronmean_mep13, *m_xerronmean_mep14,
	 *m_xerronmean_mep21, *m_xerronmean_mep22, *m_xerronmean_mep31, *m_xerronmean_mep32, *m_xerronmean_mep41,
	 *m_xerronmean_mem11, *m_xerronmean_mem12, *m_xerronmean_mem13, *m_xerronmean_mem14,
	 *m_xerronmean_mem21, *m_xerronmean_mem22, *m_xerronmean_mem31, *m_xerronmean_mem32, *m_xerronmean_mem41,
	 *m_xerronmean_me11, *m_xerronmean_me12, *m_xerronmean_me13, *m_xerronmean_me14,
	 *m_xerronmean_me21, *m_xerronmean_me22, *m_xerronmean_me31, *m_xerronmean_me32, *m_xerronmean_me41;

      TH1F *m_yresid, *m_yresid_mb, *m_yresid_me,
	 *m_yresid_mb1, *m_yresid_mb2, *m_yresid_mb3, *m_yresid_mb4,
	 *m_yresid_minus2, *m_yresid_minus1, *m_yresid_zero, *m_yresid_plus1, *m_yresid_plus2,
	 *m_yresid_mep11, *m_yresid_mep12, *m_yresid_mep13, *m_yresid_mep14,
	 *m_yresid_mep21, *m_yresid_mep22, *m_yresid_mep31, *m_yresid_mep32, *m_yresid_mep41,
	 *m_yresid_mem11, *m_yresid_mem12, *m_yresid_mem13, *m_yresid_mem14,
	 *m_yresid_mem21, *m_yresid_mem22, *m_yresid_mem31, *m_yresid_mem32, *m_yresid_mem41,
	 *m_yresid_me11, *m_yresid_me12, *m_yresid_me13, *m_yresid_me14,
	 *m_yresid_me21, *m_yresid_me22, *m_yresid_me31, *m_yresid_me32, *m_yresid_me41;

      TH1F *m_ymean, *m_ymean_mb, *m_ymean_me,
	 *m_ymean_mb1, *m_ymean_mb2, *m_ymean_mb3, *m_ymean_mb4,
	 *m_ymean_minus2, *m_ymean_minus1, *m_ymean_zero, *m_ymean_plus1, *m_ymean_plus2,
	 *m_ymean_mep11, *m_ymean_mep12, *m_ymean_mep13, *m_ymean_mep14,
	 *m_ymean_mep21, *m_ymean_mep22, *m_ymean_mep31, *m_ymean_mep32, *m_ymean_mep41,
	 *m_ymean_mem11, *m_ymean_mem12, *m_ymean_mem13, *m_ymean_mem14,
	 *m_ymean_mem21, *m_ymean_mem22, *m_ymean_mem31, *m_ymean_mem32, *m_ymean_mem41,
	 *m_ymean_me11, *m_ymean_me12, *m_ymean_me13, *m_ymean_me14,
	 *m_ymean_me21, *m_ymean_me22, *m_ymean_me31, *m_ymean_me32, *m_ymean_me41;

      TH1F *m_ystdev, *m_ystdev_mb, *m_ystdev_me,
	 *m_ystdev_mb1, *m_ystdev_mb2, *m_ystdev_mb3, *m_ystdev_mb4,
	 *m_ystdev_minus2, *m_ystdev_minus1, *m_ystdev_zero, *m_ystdev_plus1, *m_ystdev_plus2,
	 *m_ystdev_mep11, *m_ystdev_mep12, *m_ystdev_mep13, *m_ystdev_mep14,
	 *m_ystdev_mep21, *m_ystdev_mep22, *m_ystdev_mep31, *m_ystdev_mep32, *m_ystdev_mep41,
	 *m_ystdev_mem11, *m_ystdev_mem12, *m_ystdev_mem13, *m_ystdev_mem14,
	 *m_ystdev_mem21, *m_ystdev_mem22, *m_ystdev_mem31, *m_ystdev_mem32, *m_ystdev_mem41,
	 *m_ystdev_me11, *m_ystdev_me12, *m_ystdev_me13, *m_ystdev_me14,
	 *m_ystdev_me21, *m_ystdev_me22, *m_ystdev_me31, *m_ystdev_me32, *m_ystdev_me41;

      TH1F *m_yerronmean, *m_yerronmean_mb, *m_yerronmean_me,
	 *m_yerronmean_mb1, *m_yerronmean_mb2, *m_yerronmean_mb3, *m_yerronmean_mb4,
	 *m_yerronmean_minus2, *m_yerronmean_minus1, *m_yerronmean_zero, *m_yerronmean_plus1, *m_yerronmean_plus2,
	 *m_yerronmean_mep11, *m_yerronmean_mep12, *m_yerronmean_mep13, *m_yerronmean_mep14,
	 *m_yerronmean_mep21, *m_yerronmean_mep22, *m_yerronmean_mep31, *m_yerronmean_mep32, *m_yerronmean_mep41,
	 *m_yerronmean_mem11, *m_yerronmean_mem12, *m_yerronmean_mem13, *m_yerronmean_mem14,
	 *m_yerronmean_mem21, *m_yerronmean_mem22, *m_yerronmean_mem31, *m_yerronmean_mem32, *m_yerronmean_mem41,
	 *m_yerronmean_me11, *m_yerronmean_me12, *m_yerronmean_me13, *m_yerronmean_me14,
	 *m_yerronmean_me21, *m_yerronmean_me22, *m_yerronmean_me31, *m_yerronmean_me32, *m_yerronmean_me41;

      TTree *m_chambers;
      Int_t m_chambers_rawid, m_chambers_endcap, m_chambers_wheel, m_chambers_station, m_chambers_sector, m_chambers_ring, m_chambers_chamber;
      Int_t m_chambers_numx, m_chambers_numy;
      Float_t m_chambers_x_w, m_chambers_x_ww, m_chambers_x_wx, m_chambers_x_wxx;
      Float_t m_chambers_y_w, m_chambers_y_ww, m_chambers_y_wy, m_chambers_y_wyy;

      unsigned int xresid_bins, xmean_bins, xstdev_bins, xerronmean_bins, yresid_bins, ymean_bins, ystdev_bins, yerronmean_bins;
      double xresid_low, xresid_high, xmean_low, xmean_high, xstdev_low, xstdev_high, xerronmean_low, xerronmean_high, yresid_low, yresid_high, ymean_low, ymean_high, ystdev_low, ystdev_high, yerronmean_low, yerronmean_high;

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

AlignmentMonitorMuonResiduals::AlignmentMonitorMuonResiduals(const edm::ParameterSet& cfg)
   : AlignmentMonitorBase(cfg, "AlignmentMonitorMuonResiduals")
{
   xresid_bins = cfg.getParameter<unsigned int>("xresid_bins");
   xmean_bins = cfg.getParameter<unsigned int>("xmean_bins");
   xstdev_bins = cfg.getParameter<unsigned int>("xstdev_bins");
   xerronmean_bins = cfg.getParameter<unsigned int>("xerronmean_bins");
   yresid_bins = cfg.getParameter<unsigned int>("yresid_bins");
   ymean_bins = cfg.getParameter<unsigned int>("ymean_bins");
   ystdev_bins = cfg.getParameter<unsigned int>("ystdev_bins");
   yerronmean_bins = cfg.getParameter<unsigned int>("yerronmean_bins");
   xresid_low = cfg.getParameter<double>("xresid_low");
   xresid_high = cfg.getParameter<double>("xresid_high");
   xmean_low = cfg.getParameter<double>("xmean_low");
   xmean_high = cfg.getParameter<double>("xmean_high");
   xstdev_low = cfg.getParameter<double>("xstdev_low");
   xstdev_high = cfg.getParameter<double>("xstdev_high");
   xerronmean_low = cfg.getParameter<double>("xerronmean_low");
   xerronmean_high = cfg.getParameter<double>("xerronmean_high");
   yresid_low = cfg.getParameter<double>("yresid_low");
   yresid_high = cfg.getParameter<double>("yresid_high");
   ymean_low = cfg.getParameter<double>("ymean_low");
   ymean_high = cfg.getParameter<double>("ymean_high");
   ystdev_low = cfg.getParameter<double>("ystdev_low");
   ystdev_high = cfg.getParameter<double>("ystdev_high");
   yerronmean_low = cfg.getParameter<double>("yerronmean_low");
   yerronmean_high = cfg.getParameter<double>("yerronmean_high");
}

void AlignmentMonitorMuonResiduals::book() {
   m_numx.clear();
   m_x_w.clear();
   m_x_ww.clear();
   m_x_wx.clear();
   m_x_wxx.clear();
   m_numy.clear();
   m_y_w.clear();
   m_y_ww.clear();
   m_y_wy.clear();
   m_y_wyy.clear();

   std::vector<Alignable*> chambers;
   std::vector<Alignable*> tmp1 = pMuon()->DTChambers();
   for (std::vector<Alignable*>::const_iterator iter = tmp1.begin();  iter != tmp1.end();  ++iter) chambers.push_back(*iter);
   std::vector<Alignable*> tmp2 = pMuon()->CSCChambers();
   for (std::vector<Alignable*>::const_iterator iter = tmp2.begin();  iter != tmp2.end();  ++iter) chambers.push_back(*iter);

   for (std::vector<Alignable*>::const_iterator chamber = chambers.begin();  chamber != chambers.end();  ++chamber) {
      int id = (*chamber)->geomDetId().rawId();
      m_numx[id] = 0;
      m_x_w[id] = 0;
      m_x_ww[id] = 0;
      m_x_wx[id] = 0;
      m_x_wxx[id] = 0;
      m_numy[id] = 0;
      m_y_w[id] = 0;
      m_y_ww[id] = 0;
      m_y_wy[id] = 0;
      m_y_wyy[id] = 0;
   }

   m_sumnumx = book1D("/iterN/", "numx", "number of x hits", chambers.size(), 0.5, chambers.size() + 0.5);
   m_sumnumy = book1D("/iterN/", "numy", "number of y hits", chambers.size(), 0.5, chambers.size() + 0.5);
   m_xsummary = book1D("/iterN/", "xsummary", "summary of x means and errors (mm vertical axis)", chambers.size(), 0.5, chambers.size() + 0.5);
   m_ysummary = book1D("/iterN/", "ysummary", "summary of y means and errors (mm vertical axis)", chambers.size(), 0.5, chambers.size() + 0.5);

   m_xresid = book1D("/iterN/", "xresid", "x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mb = book1D("/iterN/mb/", "xresid_mb", "barrel x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me = book1D("/iterN/me/", "xresid_me", "endcap x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mb1 = book1D("/iterN/mb1/", "xresid_mb1", "MB station 1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mb2 = book1D("/iterN/mb2/", "xresid_mb2", "MB station 2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mb3 = book1D("/iterN/mb3/", "xresid_mb3", "MB station 3 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mb4 = book1D("/iterN/mb4/", "xresid_mb4", "MB station 4 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_minus2 = book1D("/iterN/minus2/", "xresid_minus2", "MB wheel -2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_minus1 = book1D("/iterN/minus1/", "xresid_minus1", "MB wheel -1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_zero = book1D("/iterN/zero/", "xresid_zero", "MB wheel 0 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_plus1 = book1D("/iterN/plus1/", "xresid_plus1", "MB wheel +1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_plus2 = book1D("/iterN/plus2/", "xresid_plus2", "MB wheel +2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep11 = book1D("/iterN/mep11/", "xresid_mep11", "ME+1/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep12 = book1D("/iterN/mep12/", "xresid_mep12", "ME+1/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep13 = book1D("/iterN/mep13/", "xresid_mep13", "ME+1/3 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep14 = book1D("/iterN/mep14/", "xresid_mep14", "ME+1/4 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep21 = book1D("/iterN/mep21/", "xresid_mep21", "ME+2/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep22 = book1D("/iterN/mep22/", "xresid_mep22", "ME+2/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep31 = book1D("/iterN/mep31/", "xresid_mep31", "ME+3/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep32 = book1D("/iterN/mep32/", "xresid_mep32", "ME+3/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep41 = book1D("/iterN/mep41/", "xresid_mep41", "ME+4/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem11 = book1D("/iterN/mem11/", "xresid_mem11", "ME-1/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem12 = book1D("/iterN/mem12/", "xresid_mem12", "ME-1/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem13 = book1D("/iterN/mem13/", "xresid_mem13", "ME-1/3 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem14 = book1D("/iterN/mem14/", "xresid_mem14", "ME-1/4 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem21 = book1D("/iterN/mem21/", "xresid_mem21", "ME-2/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem22 = book1D("/iterN/mem22/", "xresid_mem22", "ME-2/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem31 = book1D("/iterN/mem31/", "xresid_mem31", "ME-3/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem32 = book1D("/iterN/mem32/", "xresid_mem32", "ME-3/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem41 = book1D("/iterN/mem41/", "xresid_mem41", "ME-4/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me11 = book1D("/iterN/me11/", "xresid_me11", "ME1/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me12 = book1D("/iterN/me12/", "xresid_me12", "ME1/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me13 = book1D("/iterN/me13/", "xresid_me13", "ME1/3 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me14 = book1D("/iterN/me14/", "xresid_me14", "ME1/4 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me21 = book1D("/iterN/me21/", "xresid_me21", "ME2/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me22 = book1D("/iterN/me22/", "xresid_me22", "ME2/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me31 = book1D("/iterN/me31/", "xresid_me31", "ME3/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me32 = book1D("/iterN/me32/", "xresid_me32", "ME3/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me41 = book1D("/iterN/me41/", "xresid_me41", "ME4/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);

   m_xmean = book1D("/iterN/", "xmean", "weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mb = book1D("/iterN/mb/", "xmean_mb", "barrel weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me = book1D("/iterN/me/", "xmean_me", "endcap weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mb1 = book1D("/iterN/mb1/", "xmean_mb1", "MB station 1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mb2 = book1D("/iterN/mb2/", "xmean_mb2", "MB station 2 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mb3 = book1D("/iterN/mb3/", "xmean_mb3", "MB station 3 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mb4 = book1D("/iterN/mb4/", "xmean_mb4", "MB station 4 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_minus2 = book1D("/iterN/minus2/", "xmean_minus2", "MB wheel -2 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_minus1 = book1D("/iterN/minus1/", "xmean_minus1", "MB wheel -1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_zero = book1D("/iterN/zero/", "xmean_zero", "MB wheel 0 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_plus1 = book1D("/iterN/plus1/", "xmean_plus1", "MB wheel +1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_plus2 = book1D("/iterN/plus2/", "xmean_plus2", "MB wheel +2 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep11 = book1D("/iterN/mep11/", "xmean_mep11", "ME+1/1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep12 = book1D("/iterN/mep12/", "xmean_mep12", "ME+1/2 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep13 = book1D("/iterN/mep13/", "xmean_mep13", "ME+1/3 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep14 = book1D("/iterN/mep14/", "xmean_mep14", "ME+1/4 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep21 = book1D("/iterN/mep21/", "xmean_mep21", "ME+2/1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep22 = book1D("/iterN/mep22/", "xmean_mep22", "ME+2/2 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep31 = book1D("/iterN/mep31/", "xmean_mep31", "ME+3/1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep32 = book1D("/iterN/mep32/", "xmean_mep32", "ME+3/2 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep41 = book1D("/iterN/mep41/", "xmean_mep41", "ME+4/1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem11 = book1D("/iterN/mem11/", "xmean_mem11", "ME-1/1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem12 = book1D("/iterN/mem12/", "xmean_mem12", "ME-1/2 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem13 = book1D("/iterN/mem13/", "xmean_mem13", "ME-1/3 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem14 = book1D("/iterN/mem14/", "xmean_mem14", "ME-1/4 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem21 = book1D("/iterN/mem21/", "xmean_mem21", "ME-2/1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem22 = book1D("/iterN/mem22/", "xmean_mem22", "ME-2/2 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem31 = book1D("/iterN/mem31/", "xmean_mem31", "ME-3/1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem32 = book1D("/iterN/mem32/", "xmean_mem32", "ME-3/2 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem41 = book1D("/iterN/mem41/", "xmean_mem41", "ME-4/1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me11 = book1D("/iterN/me11/", "xmean_me11", "ME1/1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me12 = book1D("/iterN/me12/", "xmean_me12", "ME1/2 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me13 = book1D("/iterN/me13/", "xmean_me13", "ME1/3 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me14 = book1D("/iterN/me14/", "xmean_me14", "ME1/4 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me21 = book1D("/iterN/me21/", "xmean_me21", "ME2/1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me22 = book1D("/iterN/me22/", "xmean_me22", "ME2/2 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me31 = book1D("/iterN/me31/", "xmean_me31", "ME3/1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me32 = book1D("/iterN/me32/", "xmean_me32", "ME3/2 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me41 = book1D("/iterN/me41/", "xmean_me41", "ME4/1 weighted mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);

   m_xstdev = book1D("/iterN/", "xstdev", "weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mb = book1D("/iterN/mb/", "xstdev_mb", "barrel weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me = book1D("/iterN/me/", "xstdev_me", "endcap weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mb1 = book1D("/iterN/mb1/", "xstdev_mb1", "MB station 1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mb2 = book1D("/iterN/mb2/", "xstdev_mb2", "MB station 2 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mb3 = book1D("/iterN/mb3/", "xstdev_mb3", "MB station 3 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mb4 = book1D("/iterN/mb4/", "xstdev_mb4", "MB station 4 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_minus2 = book1D("/iterN/minus2/", "xstdev_minus2", "MB wheel -2 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_minus1 = book1D("/iterN/minus1/", "xstdev_minus1", "MB wheel -1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_zero = book1D("/iterN/zero/", "xstdev_zero", "MB wheel 0 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_plus1 = book1D("/iterN/plus1/", "xstdev_plus1", "MB wheel +1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_plus2 = book1D("/iterN/plus2/", "xstdev_plus2", "MB wheel +2 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep11 = book1D("/iterN/mep11/", "xstdev_mep11", "ME+1/1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep12 = book1D("/iterN/mep12/", "xstdev_mep12", "ME+1/2 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep13 = book1D("/iterN/mep13/", "xstdev_mep13", "ME+1/3 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep14 = book1D("/iterN/mep14/", "xstdev_mep14", "ME+1/4 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep21 = book1D("/iterN/mep21/", "xstdev_mep21", "ME+2/1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep22 = book1D("/iterN/mep22/", "xstdev_mep22", "ME+2/2 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep31 = book1D("/iterN/mep31/", "xstdev_mep31", "ME+3/1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep32 = book1D("/iterN/mep32/", "xstdev_mep32", "ME+3/2 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep41 = book1D("/iterN/mep41/", "xstdev_mep41", "ME+4/1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem11 = book1D("/iterN/mem11/", "xstdev_mem11", "ME-1/1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem12 = book1D("/iterN/mem12/", "xstdev_mem12", "ME-1/2 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem13 = book1D("/iterN/mem13/", "xstdev_mem13", "ME-1/3 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem14 = book1D("/iterN/mem14/", "xstdev_mem14", "ME-1/4 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem21 = book1D("/iterN/mem21/", "xstdev_mem21", "ME-2/1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem22 = book1D("/iterN/mem22/", "xstdev_mem22", "ME-2/2 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem31 = book1D("/iterN/mem31/", "xstdev_mem31", "ME-3/1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem32 = book1D("/iterN/mem32/", "xstdev_mem32", "ME-3/2 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem41 = book1D("/iterN/mem41/", "xstdev_mem41", "ME-4/1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me11 = book1D("/iterN/me11/", "xstdev_me11", "ME1/1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me12 = book1D("/iterN/me12/", "xstdev_me12", "ME1/2 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me13 = book1D("/iterN/me13/", "xstdev_me13", "ME1/3 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me14 = book1D("/iterN/me14/", "xstdev_me14", "ME1/4 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me21 = book1D("/iterN/me21/", "xstdev_me21", "ME2/1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me22 = book1D("/iterN/me22/", "xstdev_me22", "ME2/2 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me31 = book1D("/iterN/me31/", "xstdev_me31", "ME3/1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me32 = book1D("/iterN/me32/", "xstdev_me32", "ME3/2 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me41 = book1D("/iterN/me41/", "xstdev_me41", "ME4/1 weighted stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);

   m_xerronmean = book1D("/iterN/", "xerronmean", "error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mb = book1D("/iterN/mb/", "xerronmean_mb", "barrel error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_me = book1D("/iterN/me/", "xerronmean_me", "endcap error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mb1 = book1D("/iterN/mb1/", "xerronmean_mb1", "MB station 1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mb2 = book1D("/iterN/mb2/", "xerronmean_mb2", "MB station 2 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mb3 = book1D("/iterN/mb3/", "xerronmean_mb3", "MB station 3 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mb4 = book1D("/iterN/mb4/", "xerronmean_mb4", "MB station 4 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_minus2 = book1D("/iterN/minus2/", "xerronmean_minus2", "MB wheel -2 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_minus1 = book1D("/iterN/minus1/", "xerronmean_minus1", "MB wheel -1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_zero = book1D("/iterN/zero/", "xerronmean_zero", "MB wheel 0 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_plus1 = book1D("/iterN/plus1/", "xerronmean_plus1", "MB wheel +1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_plus2 = book1D("/iterN/plus2/", "xerronmean_plus2", "MB wheel +2 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mep11 = book1D("/iterN/mep11/", "xerronmean_mep11", "ME+1/1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mep12 = book1D("/iterN/mep12/", "xerronmean_mep12", "ME+1/2 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mep13 = book1D("/iterN/mep13/", "xerronmean_mep13", "ME+1/3 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mep14 = book1D("/iterN/mep14/", "xerronmean_mep14", "ME+1/4 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mep21 = book1D("/iterN/mep21/", "xerronmean_mep21", "ME+2/1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mep22 = book1D("/iterN/mep22/", "xerronmean_mep22", "ME+2/2 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mep31 = book1D("/iterN/mep31/", "xerronmean_mep31", "ME+3/1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mep32 = book1D("/iterN/mep32/", "xerronmean_mep32", "ME+3/2 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mep41 = book1D("/iterN/mep41/", "xerronmean_mep41", "ME+4/1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mem11 = book1D("/iterN/mem11/", "xerronmean_mem11", "ME-1/1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mem12 = book1D("/iterN/mem12/", "xerronmean_mem12", "ME-1/2 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mem13 = book1D("/iterN/mem13/", "xerronmean_mem13", "ME-1/3 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mem14 = book1D("/iterN/mem14/", "xerronmean_mem14", "ME-1/4 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mem21 = book1D("/iterN/mem21/", "xerronmean_mem21", "ME-2/1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mem22 = book1D("/iterN/mem22/", "xerronmean_mem22", "ME-2/2 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mem31 = book1D("/iterN/mem31/", "xerronmean_mem31", "ME-3/1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mem32 = book1D("/iterN/mem32/", "xerronmean_mem32", "ME-3/2 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_mem41 = book1D("/iterN/mem41/", "xerronmean_mem41", "ME-4/1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_me11 = book1D("/iterN/me11/", "xerronmean_me11", "ME1/1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_me12 = book1D("/iterN/me12/", "xerronmean_me12", "ME1/2 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_me13 = book1D("/iterN/me13/", "xerronmean_me13", "ME1/3 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_me14 = book1D("/iterN/me14/", "xerronmean_me14", "ME1/4 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_me21 = book1D("/iterN/me21/", "xerronmean_me21", "ME2/1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_me22 = book1D("/iterN/me22/", "xerronmean_me22", "ME2/2 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_me31 = book1D("/iterN/me31/", "xerronmean_me31", "ME3/1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_me32 = book1D("/iterN/me32/", "xerronmean_me32", "ME3/2 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);
   m_xerronmean_me41 = book1D("/iterN/me41/", "xerronmean_me41", "ME4/1 error on x weighted mean residual per chamber (mm)", xerronmean_bins, xerronmean_low, xerronmean_high);

   m_yresid = book1D("/iterN/", "yresid", "y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mb = book1D("/iterN/mb/", "yresid_mb", "barrel y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me = book1D("/iterN/me/", "yresid_me", "endcap y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mb1 = book1D("/iterN/mb1/", "yresid_mb1", "MB station 1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mb2 = book1D("/iterN/mb2/", "yresid_mb2", "MB station 2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mb3 = book1D("/iterN/mb3/", "yresid_mb3", "MB station 3 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mb4 = book1D("/iterN/mb4/", "yresid_mb4", "MB station 4 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_minus2 = book1D("/iterN/minus2/", "yresid_minus2", "MB wheel -2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_minus1 = book1D("/iterN/minus1/", "yresid_minus1", "MB wheel -1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_zero = book1D("/iterN/zero/", "yresid_zero", "MB wheel 0 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_plus1 = book1D("/iterN/plus1/", "yresid_plus1", "MB wheel +1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_plus2 = book1D("/iterN/plus2/", "yresid_plus2", "MB wheel +2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep11 = book1D("/iterN/mep11/", "yresid_mep11", "ME+1/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep12 = book1D("/iterN/mep12/", "yresid_mep12", "ME+1/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep13 = book1D("/iterN/mep13/", "yresid_mep13", "ME+1/3 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep14 = book1D("/iterN/mep14/", "yresid_mep14", "ME+1/4 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep21 = book1D("/iterN/mep21/", "yresid_mep21", "ME+2/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep22 = book1D("/iterN/mep22/", "yresid_mep22", "ME+2/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep31 = book1D("/iterN/mep31/", "yresid_mep31", "ME+3/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep32 = book1D("/iterN/mep32/", "yresid_mep32", "ME+3/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep41 = book1D("/iterN/mep41/", "yresid_mep41", "ME+4/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem11 = book1D("/iterN/mem11/", "yresid_mem11", "ME-1/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem12 = book1D("/iterN/mem12/", "yresid_mem12", "ME-1/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem13 = book1D("/iterN/mem13/", "yresid_mem13", "ME-1/3 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem14 = book1D("/iterN/mem14/", "yresid_mem14", "ME-1/4 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem21 = book1D("/iterN/mem21/", "yresid_mem21", "ME-2/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem22 = book1D("/iterN/mem22/", "yresid_mem22", "ME-2/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem31 = book1D("/iterN/mem31/", "yresid_mem31", "ME-3/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem32 = book1D("/iterN/mem32/", "yresid_mem32", "ME-3/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem41 = book1D("/iterN/mem41/", "yresid_mem41", "ME-4/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me11 = book1D("/iterN/me11/", "yresid_me11", "ME1/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me12 = book1D("/iterN/me12/", "yresid_me12", "ME1/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me13 = book1D("/iterN/me13/", "yresid_me13", "ME1/3 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me14 = book1D("/iterN/me14/", "yresid_me14", "ME1/4 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me21 = book1D("/iterN/me21/", "yresid_me21", "ME2/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me22 = book1D("/iterN/me22/", "yresid_me22", "ME2/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me31 = book1D("/iterN/me31/", "yresid_me31", "ME3/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me32 = book1D("/iterN/me32/", "yresid_me32", "ME3/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me41 = book1D("/iterN/me41/", "yresid_me41", "ME4/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);

   m_ymean = book1D("/iterN/", "ymean", "weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mb = book1D("/iterN/mb/", "ymean_mb", "barrel weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me = book1D("/iterN/me/", "ymean_me", "endcap weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mb1 = book1D("/iterN/mb1/", "ymean_mb1", "MB station 1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mb2 = book1D("/iterN/mb2/", "ymean_mb2", "MB station 2 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mb3 = book1D("/iterN/mb3/", "ymean_mb3", "MB station 3 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mb4 = book1D("/iterN/mb4/", "ymean_mb4", "MB station 4 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_minus2 = book1D("/iterN/minus2/", "ymean_minus2", "MB wheel -2 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_minus1 = book1D("/iterN/minus1/", "ymean_minus1", "MB wheel -1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_zero = book1D("/iterN/zero/", "ymean_zero", "MB wheel 0 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_plus1 = book1D("/iterN/plus1/", "ymean_plus1", "MB wheel +1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_plus2 = book1D("/iterN/plus2/", "ymean_plus2", "MB wheel +2 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep11 = book1D("/iterN/mep11/", "ymean_mep11", "ME+1/1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep12 = book1D("/iterN/mep12/", "ymean_mep12", "ME+1/2 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep13 = book1D("/iterN/mep13/", "ymean_mep13", "ME+1/3 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep14 = book1D("/iterN/mep14/", "ymean_mep14", "ME+1/4 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep21 = book1D("/iterN/mep21/", "ymean_mep21", "ME+2/1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep22 = book1D("/iterN/mep22/", "ymean_mep22", "ME+2/2 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep31 = book1D("/iterN/mep31/", "ymean_mep31", "ME+3/1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep32 = book1D("/iterN/mep32/", "ymean_mep32", "ME+3/2 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep41 = book1D("/iterN/mep41/", "ymean_mep41", "ME+4/1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem11 = book1D("/iterN/mem11/", "ymean_mem11", "ME-1/1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem12 = book1D("/iterN/mem12/", "ymean_mem12", "ME-1/2 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem13 = book1D("/iterN/mem13/", "ymean_mem13", "ME-1/3 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem14 = book1D("/iterN/mem14/", "ymean_mem14", "ME-1/4 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem21 = book1D("/iterN/mem21/", "ymean_mem21", "ME-2/1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem22 = book1D("/iterN/mem22/", "ymean_mem22", "ME-2/2 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem31 = book1D("/iterN/mem31/", "ymean_mem31", "ME-3/1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem32 = book1D("/iterN/mem32/", "ymean_mem32", "ME-3/2 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem41 = book1D("/iterN/mem41/", "ymean_mem41", "ME-4/1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me11 = book1D("/iterN/me11/", "ymean_me11", "ME1/1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me12 = book1D("/iterN/me12/", "ymean_me12", "ME1/2 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me13 = book1D("/iterN/me13/", "ymean_me13", "ME1/3 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me14 = book1D("/iterN/me14/", "ymean_me14", "ME1/4 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me21 = book1D("/iterN/me21/", "ymean_me21", "ME2/1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me22 = book1D("/iterN/me22/", "ymean_me22", "ME2/2 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me31 = book1D("/iterN/me31/", "ymean_me31", "ME3/1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me32 = book1D("/iterN/me32/", "ymean_me32", "ME3/2 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me41 = book1D("/iterN/me41/", "ymean_me41", "ME4/1 weighted mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);

   m_ystdev = book1D("/iterN/", "ystdev", "weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mb = book1D("/iterN/mb/", "ystdev_mb", "barrel weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me = book1D("/iterN/me/", "ystdev_me", "endcap weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mb1 = book1D("/iterN/mb1/", "ystdev_mb1", "MB station 1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mb2 = book1D("/iterN/mb2/", "ystdev_mb2", "MB station 2 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mb3 = book1D("/iterN/mb3/", "ystdev_mb3", "MB station 3 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mb4 = book1D("/iterN/mb4/", "ystdev_mb4", "MB station 4 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_minus2 = book1D("/iterN/minus2/", "ystdev_minus2", "MB wheel -2 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_minus1 = book1D("/iterN/minus1/", "ystdev_minus1", "MB wheel -1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_zero = book1D("/iterN/zero/", "ystdev_zero", "MB wheel 0 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_plus1 = book1D("/iterN/plus1/", "ystdev_plus1", "MB wheel +1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_plus2 = book1D("/iterN/plus2/", "ystdev_plus2", "MB wheel +2 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep11 = book1D("/iterN/mep11/", "ystdev_mep11", "ME+1/1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep12 = book1D("/iterN/mep12/", "ystdev_mep12", "ME+1/2 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep13 = book1D("/iterN/mep13/", "ystdev_mep13", "ME+1/3 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep14 = book1D("/iterN/mep14/", "ystdev_mep14", "ME+1/4 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep21 = book1D("/iterN/mep21/", "ystdev_mep21", "ME+2/1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep22 = book1D("/iterN/mep22/", "ystdev_mep22", "ME+2/2 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep31 = book1D("/iterN/mep31/", "ystdev_mep31", "ME+3/1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep32 = book1D("/iterN/mep32/", "ystdev_mep32", "ME+3/2 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep41 = book1D("/iterN/mep41/", "ystdev_mep41", "ME+4/1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem11 = book1D("/iterN/mem11/", "ystdev_mem11", "ME-1/1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem12 = book1D("/iterN/mem12/", "ystdev_mem12", "ME-1/2 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem13 = book1D("/iterN/mem13/", "ystdev_mem13", "ME-1/3 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem14 = book1D("/iterN/mem14/", "ystdev_mem14", "ME-1/4 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem21 = book1D("/iterN/mem21/", "ystdev_mem21", "ME-2/1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem22 = book1D("/iterN/mem22/", "ystdev_mem22", "ME-2/2 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem31 = book1D("/iterN/mem31/", "ystdev_mem31", "ME-3/1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem32 = book1D("/iterN/mem32/", "ystdev_mem32", "ME-3/2 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem41 = book1D("/iterN/mem41/", "ystdev_mem41", "ME-4/1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me11 = book1D("/iterN/me11/", "ystdev_me11", "ME1/1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me12 = book1D("/iterN/me12/", "ystdev_me12", "ME1/2 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me13 = book1D("/iterN/me13/", "ystdev_me13", "ME1/3 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me14 = book1D("/iterN/me14/", "ystdev_me14", "ME1/4 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me21 = book1D("/iterN/me21/", "ystdev_me21", "ME2/1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me22 = book1D("/iterN/me22/", "ystdev_me22", "ME2/2 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me31 = book1D("/iterN/me31/", "ystdev_me31", "ME3/1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me32 = book1D("/iterN/me32/", "ystdev_me32", "ME3/2 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me41 = book1D("/iterN/me41/", "ystdev_me41", "ME4/1 weighted stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);

   m_yerronmean = book1D("/iterN/", "yerronmean", "error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mb = book1D("/iterN/mb/", "yerronmean_mb", "barrel error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_me = book1D("/iterN/me/", "yerronmean_me", "endcap error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mb1 = book1D("/iterN/mb1/", "yerronmean_mb1", "MB station 1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mb2 = book1D("/iterN/mb2/", "yerronmean_mb2", "MB station 2 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mb3 = book1D("/iterN/mb3/", "yerronmean_mb3", "MB station 3 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mb4 = book1D("/iterN/mb4/", "yerronmean_mb4", "MB station 4 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_minus2 = book1D("/iterN/minus2/", "yerronmean_minus2", "MB wheel -2 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_minus1 = book1D("/iterN/minus1/", "yerronmean_minus1", "MB wheel -1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_zero = book1D("/iterN/zero/", "yerronmean_zero", "MB wheel 0 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_plus1 = book1D("/iterN/plus1/", "yerronmean_plus1", "MB wheel +1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_plus2 = book1D("/iterN/plus2/", "yerronmean_plus2", "MB wheel +2 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mep11 = book1D("/iterN/mep11/", "yerronmean_mep11", "ME+1/1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mep12 = book1D("/iterN/mep12/", "yerronmean_mep12", "ME+1/2 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mep13 = book1D("/iterN/mep13/", "yerronmean_mep13", "ME+1/3 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mep14 = book1D("/iterN/mep14/", "yerronmean_mep14", "ME+1/4 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mep21 = book1D("/iterN/mep21/", "yerronmean_mep21", "ME+2/1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mep22 = book1D("/iterN/mep22/", "yerronmean_mep22", "ME+2/2 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mep31 = book1D("/iterN/mep31/", "yerronmean_mep31", "ME+3/1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mep32 = book1D("/iterN/mep32/", "yerronmean_mep32", "ME+3/2 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mep41 = book1D("/iterN/mep41/", "yerronmean_mep41", "ME+4/1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mem11 = book1D("/iterN/mem11/", "yerronmean_mem11", "ME-1/1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mem12 = book1D("/iterN/mem12/", "yerronmean_mem12", "ME-1/2 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mem13 = book1D("/iterN/mem13/", "yerronmean_mem13", "ME-1/3 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mem14 = book1D("/iterN/mem14/", "yerronmean_mem14", "ME-1/4 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mem21 = book1D("/iterN/mem21/", "yerronmean_mem21", "ME-2/1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mem22 = book1D("/iterN/mem22/", "yerronmean_mem22", "ME-2/2 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mem31 = book1D("/iterN/mem31/", "yerronmean_mem31", "ME-3/1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mem32 = book1D("/iterN/mem32/", "yerronmean_mem32", "ME-3/2 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_mem41 = book1D("/iterN/mem41/", "yerronmean_mem41", "ME-4/1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_me11 = book1D("/iterN/me11/", "yerronmean_me11", "ME1/1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_me12 = book1D("/iterN/me12/", "yerronmean_me12", "ME1/2 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_me13 = book1D("/iterN/me13/", "yerronmean_me13", "ME1/3 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_me14 = book1D("/iterN/me14/", "yerronmean_me14", "ME1/4 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_me21 = book1D("/iterN/me21/", "yerronmean_me21", "ME2/1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_me22 = book1D("/iterN/me22/", "yerronmean_me22", "ME2/2 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_me31 = book1D("/iterN/me31/", "yerronmean_me31", "ME3/1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_me32 = book1D("/iterN/me32/", "yerronmean_me32", "ME3/2 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);
   m_yerronmean_me41 = book1D("/iterN/me41/", "yerronmean_me41", "ME4/1 error on y weighted mean residual per chamber (mm)", yerronmean_bins, yerronmean_low, yerronmean_high);

   m_chambers = directory("/iterN/")->make<TTree>("chambers", "residual statistics for each chamber");
   m_chambers->Branch("rawid", &m_chambers_rawid, "rawid/I");
   m_chambers->Branch("endcap", &m_chambers_endcap, "endcap/I");
   m_chambers->Branch("wheel", &m_chambers_wheel, "wheel/I");
   m_chambers->Branch("station", &m_chambers_station, "station/I");
   m_chambers->Branch("sector", &m_chambers_sector, "sector/I");
   m_chambers->Branch("ring", &m_chambers_ring, "ring/I");
   m_chambers->Branch("chamber", &m_chambers_chamber, "chamber/I");
   m_chambers->Branch("numx", &m_chambers_numx, "numx/I");
   m_chambers->Branch("x_w", &m_chambers_x_w, "x_w/F");
   m_chambers->Branch("x_ww", &m_chambers_x_ww, "x_ww/F");
   m_chambers->Branch("x_wx", &m_chambers_x_wx, "x_wx/F");
   m_chambers->Branch("x_wxx", &m_chambers_x_wxx, "x_wxx/F");
   m_chambers->Branch("numy", &m_chambers_numy, "numy/I");
   m_chambers->Branch("y_w", &m_chambers_y_w, "y_w/F");
   m_chambers->Branch("y_ww", &m_chambers_y_ww, "y_ww/F");
   m_chambers->Branch("y_wy", &m_chambers_y_wy, "y_wy/F");
   m_chambers->Branch("y_wyy", &m_chambers_y_wyy, "y_wyy/F");

}

void AlignmentMonitorMuonResiduals::event(const edm::Event &iEvent, const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& tracks) {
   TrajectoryStateCombiner tsoscomb;

   for (ConstTrajTrackPairCollection::const_iterator it = tracks.begin();  it != tracks.end();  ++it) {
      const Trajectory *traj = it->first;
//      const reco::Track *track = it->second;

      std::vector<TrajectoryMeasurement> measurements = traj->measurements();

      for (std::vector<TrajectoryMeasurement>::const_iterator im = measurements.begin();  im != measurements.end();  ++im) {
	 const TrajectoryMeasurement meas = *im;
	 const TransientTrackingRecHit* hit = &(*meas.recHit());
	 const DetId id = hit->geographicalId();

	 if (hit->isValid()  &&  pNavigator()->detAndSubdetInMap(id)) {
	    TrajectoryStateOnSurface tsosc = tsoscomb.combine(meas.forwardPredictedState(), meas.backwardPredictedState());
	    align::LocalPoint trackPos = tsosc.localPosition();
	    LocalError trackErr = tsosc.localError().positionError();
	    align::LocalPoint hitPos = hit->localPosition();
	    LocalError hitErr = hit->localPositionError(); // CPE+APE

	    // subtract APEs from hitErr (if existing) from covariance matrix
	    auto det = static_cast<const TrackerGeomDet*>(hit->det());
	    const auto localAPE = det->localAlignmentError();
	    if (localAPE.valid()) {
	      hitErr = LocalError(hitErr.xx() - localAPE.xx(),
				  hitErr.xy() - localAPE.xy(),
				  hitErr.yy() - localAPE.yy());
	    }

	    double x_residual = 10. * (trackPos.x() - hitPos.x());
	    double y_residual = 10. * (trackPos.y() - hitPos.y());
	    double x_reserr2 = 100. * (trackErr.xx() + hitErr.xx());
	    double y_reserr2 = 100. * (trackErr.yy() + hitErr.yy());
// 	    double xpos = trackPos.x();
// 	    double ypos = trackPos.y();

	    if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
	       if (fabs(hit->surface()->toGlobal(align::LocalVector(0,1,0)).z()) < 0.1) {
                  // local y != global z: it's a middle (y-measuring) superlayer
		  y_residual = x_residual;
		  y_reserr2 = x_reserr2;
		  
		  x_residual = 0.;
		  x_reserr2 = 0.;
	       }
	       else {
		  y_residual = 0.;
		  y_reserr2 = 0.;
	       }

	       if (x_reserr2 > 0.) {
		  m_xresid->Fill(x_residual, 1./x_reserr2);
		  m_xresid_mb->Fill(x_residual, 1./x_reserr2);
	       }
	       if (y_reserr2 > 0.) {
		  m_yresid->Fill(y_residual, 1./y_reserr2);
		  m_yresid_mb->Fill(y_residual, 1./y_reserr2);
	       }
	         
	       DTChamberId dtId(id.rawId());
	       int rawId = dtId.rawId();
	       if (x_reserr2 > 0.) {
		  m_numx[rawId]++;
		  m_x_w[rawId] += 1./x_reserr2;
		  m_x_ww[rawId] += 1./x_reserr2/x_reserr2;
		  m_x_wx[rawId] += x_residual/x_reserr2;
		  m_x_wxx[rawId] += x_residual*x_residual/x_reserr2;
	       }
	       if (y_reserr2 > 0.) {
		  m_numy[rawId]++;
		  m_y_w[rawId] += 1./y_reserr2;
		  m_y_ww[rawId] += 1./y_reserr2/y_reserr2;
		  m_y_wy[rawId] += y_residual/y_reserr2;
		  m_y_wyy[rawId] += y_residual*y_residual/y_reserr2;
	       }

	       if (dtId.station() == 1) {
		  if (x_reserr2 > 0.) {
		     m_xresid_mb1->Fill(x_residual, 1./x_reserr2);
		  }
		  if (y_reserr2 > 0.) {
		     m_yresid_mb1->Fill(y_residual, 1./y_reserr2);
		  }
	       }
	       else if (dtId.station() == 2) {
		  if (x_reserr2 > 0.) {
		     m_xresid_mb2->Fill(x_residual, 1./x_reserr2);
		  }
		  if (y_reserr2 > 0.) {
		     m_yresid_mb2->Fill(y_residual, 1./y_reserr2);
		  }
	       }
	       else if (dtId.station() == 3) {
		  if (x_reserr2 > 0.) {
		     m_xresid_mb3->Fill(x_residual, 1./x_reserr2);
		  }
		  if (y_reserr2 > 0.) {
		     m_yresid_mb3->Fill(y_residual, 1./y_reserr2);
		  }
	       }
	       else if (dtId.station() == 4) {
		  if (x_reserr2 > 0.) {
		     m_xresid_mb4->Fill(x_residual, 1./x_reserr2);
		  }
		  if (y_reserr2 > 0.) {
		     m_yresid_mb4->Fill(y_residual, 1./y_reserr2);
		  }
	       }

	       if (dtId.wheel() == -2) {
		  if (x_reserr2 > 0.) {
		     m_xresid_minus2->Fill(x_residual, 1./x_reserr2);
		  }
		  if (y_reserr2 > 0.) {
		     m_yresid_minus2->Fill(y_residual, 1./y_reserr2);
		  }
	       }
	       else if (dtId.wheel() == -1) {
		  if (x_reserr2 > 0.) {
		     m_xresid_minus1->Fill(x_residual, 1./x_reserr2);
		  }
		  if (y_reserr2 > 0.) {
		     m_yresid_minus1->Fill(y_residual, 1./y_reserr2);
		  }
	       }
	       else if (dtId.wheel() == 0) {
		  if (x_reserr2 > 0.) {
		     m_xresid_zero->Fill(x_residual, 1./x_reserr2);
		  }
		  if (y_reserr2 > 0.) {
		     m_yresid_zero->Fill(y_residual, 1./y_reserr2);
		  }
	       }
	       else if (dtId.wheel() == 1) {
		  if (x_reserr2 > 0.) {
		     m_xresid_plus1->Fill(x_residual, 1./x_reserr2);
		  }
		  if (y_reserr2 > 0.) {
		     m_yresid_plus1->Fill(y_residual, 1./y_reserr2);
		  }
	       }
	       else if (dtId.wheel() == 2) {
		  if (x_reserr2 > 0.) {
		     m_xresid_plus2->Fill(x_residual, 1./x_reserr2);
		  }
		  if (y_reserr2 > 0.) {
		     m_yresid_plus2->Fill(y_residual, 1./y_reserr2);
		  }
	       }
	    } // end if DT

	    else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
	       m_xresid->Fill(x_residual, 1./x_reserr2);
	       m_yresid->Fill(y_residual, 1./y_reserr2);

	       m_xresid_me->Fill(x_residual, 1./x_reserr2);
	       m_yresid_me->Fill(y_residual, 1./y_reserr2);

	       CSCDetId cscId(id.rawId());
	       int rawId = cscId.chamberId().rawId();
	       if (x_reserr2 > 0.) {
		  m_numx[rawId]++;
		  m_x_w[rawId] += 1./x_reserr2;
		  m_x_ww[rawId] += 1./x_reserr2/x_reserr2;
		  m_x_wx[rawId] += x_residual/x_reserr2;
		  m_x_wxx[rawId] += x_residual*x_residual/x_reserr2;
	       }
	       if (y_reserr2 > 0.) {
		  m_numy[rawId]++;
		  m_y_w[rawId] += 1./y_reserr2;
		  m_y_ww[rawId] += 1./y_reserr2/y_reserr2;
		  m_y_wy[rawId] += y_residual/y_reserr2;
		  m_y_wyy[rawId] += y_residual*y_residual/y_reserr2;
	       }

	       if ((cscId.endcap() == 1? 1: -1)*cscId.station() == 1  &&  cscId.ring() == 1) {
		  m_xresid_mep11->Fill(x_residual, 1./x_reserr2);  m_yresid_mep11->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me11->Fill(x_residual, 1./x_reserr2);  m_yresid_me11->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == -1  &&  cscId.ring() == 1) {
		  m_xresid_mem11->Fill(x_residual, 1./x_reserr2);  m_yresid_mem11->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me11->Fill(x_residual, 1./x_reserr2);  m_yresid_me11->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == 1  &&  cscId.ring() == 2) {
		  m_xresid_mep12->Fill(x_residual, 1./x_reserr2);  m_yresid_mep12->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me12->Fill(x_residual, 1./x_reserr2);  m_yresid_me12->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == -1  &&  cscId.ring() == 2) {
		  m_xresid_mem12->Fill(x_residual, 1./x_reserr2);  m_yresid_mem12->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me12->Fill(x_residual, 1./x_reserr2);  m_yresid_me12->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == 1  &&  cscId.ring() == 3) {
		  m_xresid_mep13->Fill(x_residual, 1./x_reserr2);  m_yresid_mep13->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me13->Fill(x_residual, 1./x_reserr2);  m_yresid_me13->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == -1  &&  cscId.ring() == 3) {
		  m_xresid_mem13->Fill(x_residual, 1./x_reserr2);  m_yresid_mem13->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me13->Fill(x_residual, 1./x_reserr2);  m_yresid_me13->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == 1  &&  cscId.ring() == 4) {
		  m_xresid_mep14->Fill(x_residual, 1./x_reserr2);  m_yresid_mep14->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me14->Fill(x_residual, 1./x_reserr2);  m_yresid_me14->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == -1  &&  cscId.ring() == 4) {
		  m_xresid_mem14->Fill(x_residual, 1./x_reserr2);  m_yresid_mem14->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me14->Fill(x_residual, 1./x_reserr2);  m_yresid_me14->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == 2  &&  cscId.ring() == 1) {
		  m_xresid_mep21->Fill(x_residual, 1./x_reserr2);  m_yresid_mep21->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me21->Fill(x_residual, 1./x_reserr2);  m_yresid_me21->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == -2  &&  cscId.ring() == 1) {
		  m_xresid_mem21->Fill(x_residual, 1./x_reserr2);  m_yresid_mem21->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me21->Fill(x_residual, 1./x_reserr2);  m_yresid_me21->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == 2  &&  cscId.ring() == 2) {
		  m_xresid_mep22->Fill(x_residual, 1./x_reserr2);  m_yresid_mep22->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me22->Fill(x_residual, 1./x_reserr2);  m_yresid_me22->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == -2  &&  cscId.ring() == 2) {
		  m_xresid_mem22->Fill(x_residual, 1./x_reserr2);  m_yresid_mem22->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me22->Fill(x_residual, 1./x_reserr2);  m_yresid_me22->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == 3  &&  cscId.ring() == 1) {
		  m_xresid_mep31->Fill(x_residual, 1./x_reserr2);  m_yresid_mep31->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me31->Fill(x_residual, 1./x_reserr2);  m_yresid_me31->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == -3  &&  cscId.ring() == 1) {
		  m_xresid_mem31->Fill(x_residual, 1./x_reserr2);  m_yresid_mem31->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me31->Fill(x_residual, 1./x_reserr2);  m_yresid_me31->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == 3  &&  cscId.ring() == 2) {
		  m_xresid_mep32->Fill(x_residual, 1./x_reserr2);  m_yresid_mep32->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me32->Fill(x_residual, 1./x_reserr2);  m_yresid_me32->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == -3  &&  cscId.ring() == 2) {
		  m_xresid_mem32->Fill(x_residual, 1./x_reserr2);  m_yresid_mem32->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me32->Fill(x_residual, 1./x_reserr2);  m_yresid_me32->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == 4  &&  cscId.ring() == 1) {
		  m_xresid_mep41->Fill(x_residual, 1./x_reserr2);  m_yresid_mep41->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me41->Fill(x_residual, 1./x_reserr2);  m_yresid_me41->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == -4  &&  cscId.ring() == 1) {
		  m_xresid_mem41->Fill(x_residual, 1./x_reserr2);  m_yresid_mem41->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me41->Fill(x_residual, 1./x_reserr2);  m_yresid_me41->Fill(y_residual, 1./y_reserr2);
	       }
	    } // else if CSC

	 } // end if good hit
      } // end loop over measurements

   } // end loop over track-trajectories
}

void AlignmentMonitorMuonResiduals::afterAlignment() {
   std::vector<Alignable*> chambers;
   std::vector<Alignable*> tmp1 = pMuon()->DTChambers();
   for (std::vector<Alignable*>::const_iterator iter = tmp1.begin();  iter != tmp1.end();  ++iter) chambers.push_back(*iter);
   std::vector<Alignable*> tmp2 = pMuon()->CSCChambers();
   for (std::vector<Alignable*>::const_iterator iter = tmp2.begin();  iter != tmp2.end();  ++iter) chambers.push_back(*iter);

   int index = 0;
   for (std::vector<Alignable*>::const_iterator chamber = chambers.begin();  chamber != chambers.end();  ++chamber) {
      const int id = (*chamber)->geomDetId().rawId();
      
      m_chambers_rawid = id;
      m_chambers_numx = m_numx[id];
      m_chambers_x_w = m_x_w[id];
      m_chambers_x_ww = m_x_ww[id];
      m_chambers_x_wx = m_x_wx[id];
      m_chambers_x_wxx = m_x_wxx[id];
      m_chambers_numy = m_numy[id];
      m_chambers_y_w = m_y_w[id];
      m_chambers_y_ww = m_y_ww[id];
      m_chambers_y_wy = m_y_wy[id];
      m_chambers_y_wyy = m_y_wyy[id];

      index++;
      m_sumnumx->SetBinContent(index, m_numx[id]);
      m_sumnumy->SetBinContent(index, m_numy[id]);

      std::ostringstream name;
      if ((*chamber)->geomDetId().subdetId() == MuonSubdetId::DT) {
	 DTChamberId dtId((*chamber)->geomDetId());
	 name << "MB" << dtId.wheel() << "/" << dtId.station() << " (" << dtId.sector() << ")";
	 m_chambers_endcap = 0;
	 m_chambers_wheel = dtId.wheel();
	 m_chambers_station = dtId.station();
	 m_chambers_sector = dtId.sector();
	 m_chambers_ring = 0;
	 m_chambers_chamber = 0;
      }
      else {
	 CSCDetId cscId((*chamber)->geomDetId());
	 name << "ME" << (cscId.endcap() == 1? "+": "-") << cscId.station() << "/" << cscId.ring() << " (" << cscId.chamber() << ")";
	 m_chambers_endcap = cscId.endcap();
	 m_chambers_wheel = 0;
	 m_chambers_station = cscId.station();
	 m_chambers_sector = 0;
	 m_chambers_ring = cscId.ring();
	 m_chambers_chamber = cscId.chamber();
      }
      m_chambers->Fill();

      m_sumnumx->GetXaxis()->SetBinLabel(index, name.str().c_str());
      m_sumnumy->GetXaxis()->SetBinLabel(index, name.str().c_str());
      m_xsummary->GetXaxis()->SetBinLabel(index, name.str().c_str());
      m_ysummary->GetXaxis()->SetBinLabel(index, name.str().c_str());

      if (m_numx[id] > 0.) {
	 double xmean = m_x_wx[id] / m_x_w[id];
	 double xstdev = sqrt(((m_x_wxx[id] * m_x_w[id]) - (m_x_wx[id] * m_x_wx[id]))/((m_x_w[id] * m_x_w[id]) - m_x_ww[id]));
	 double xerronmean = xstdev / sqrt(m_numx[id]);

	 m_xsummary->SetBinContent(index, xmean);
	 m_xsummary->SetBinError(index, xerronmean);

	 m_xmean->Fill(xmean);  m_xstdev->Fill(xstdev);  m_xerronmean->Fill(xerronmean);
	 if ((*chamber)->geomDetId().subdetId() == MuonSubdetId::DT) {
	    m_xmean_mb->Fill(xmean);  m_xstdev_mb->Fill(xstdev);  m_xerronmean_mb->Fill(xerronmean);
	    DTChamberId id((*chamber)->geomDetId().rawId());
	    if (id.station() == 1) {
	       m_xmean_mb1->Fill(xmean);  m_xstdev_mb1->Fill(xstdev);  m_xerronmean_mb1->Fill(xerronmean);
	    }
	    else if (id.station() == 2) {
	       m_xmean_mb2->Fill(xmean);  m_xstdev_mb2->Fill(xstdev);  m_xerronmean_mb2->Fill(xerronmean);
	    }
	    else if (id.station() == 3) {
	       m_xmean_mb3->Fill(xmean);  m_xstdev_mb3->Fill(xstdev);  m_xerronmean_mb3->Fill(xerronmean);
	    }
	    else if (id.station() == 4) {
	       m_xmean_mb4->Fill(xmean);  m_xstdev_mb4->Fill(xstdev);  m_xerronmean_mb4->Fill(xerronmean);
	    }

	    if (id.wheel() == -2) {
	       m_xmean_minus2->Fill(xmean);  m_xstdev_minus2->Fill(xstdev);  m_xerronmean_minus2->Fill(xerronmean);
	    }
	    else if (id.wheel() == -1) {
	       m_xmean_minus1->Fill(xmean);  m_xstdev_minus1->Fill(xstdev);  m_xerronmean_minus1->Fill(xerronmean);
	    }
	    else if (id.wheel() == 0) {
	       m_xmean_zero->Fill(xmean);  m_xstdev_zero->Fill(xstdev);  m_xerronmean_zero->Fill(xerronmean);
	    }
	    else if (id.wheel() == 1) {
	       m_xmean_plus1->Fill(xmean);  m_xstdev_plus1->Fill(xstdev);  m_xerronmean_plus1->Fill(xerronmean);
	    }
	    else if (id.wheel() == 2) {
	       m_xmean_plus2->Fill(xmean);  m_xstdev_plus2->Fill(xstdev);  m_xerronmean_plus2->Fill(xerronmean);
	    }
	 } // end if DT
	 else {
	    m_xmean_me->Fill(xmean);  m_xstdev_me->Fill(xstdev);  m_xerronmean_me->Fill(xerronmean);

	    CSCDetId id((*chamber)->geomDetId().rawId());

	    if ((id.endcap() == 1? 1: -1)*id.station() == 1  &&  id.ring() == 1) {
	       m_xmean_mep11->Fill(xmean);  m_xstdev_mep11->Fill(xstdev);  m_xerronmean_mep11->Fill(xerronmean);
	       m_xmean_me11->Fill(xmean);  m_xstdev_me11->Fill(xstdev);  m_xerronmean_me11->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  id.ring() == 1) {
	       m_xmean_mem11->Fill(xmean);  m_xstdev_mem11->Fill(xstdev);  m_xerronmean_mem11->Fill(xerronmean);
	       m_xmean_me11->Fill(xmean);  m_xstdev_me11->Fill(xstdev);  m_xerronmean_me11->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 1  &&  id.ring() == 2) {
	       m_xmean_mep12->Fill(xmean);  m_xstdev_mep12->Fill(xstdev);  m_xerronmean_mep12->Fill(xerronmean);
	       m_xmean_me12->Fill(xmean);  m_xstdev_me12->Fill(xstdev);  m_xerronmean_me12->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  id.ring() == 2) {
	       m_xmean_mem12->Fill(xmean);  m_xstdev_mem12->Fill(xstdev);  m_xerronmean_mem12->Fill(xerronmean);
	       m_xmean_me12->Fill(xmean);  m_xstdev_me12->Fill(xstdev);  m_xerronmean_me12->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 1  &&  id.ring() == 3) {
	       m_xmean_mep13->Fill(xmean);  m_xstdev_mep13->Fill(xstdev);  m_xerronmean_mep13->Fill(xerronmean);
	       m_xmean_me13->Fill(xmean);  m_xstdev_me13->Fill(xstdev);  m_xerronmean_me13->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  id.ring() == 3) {
	       m_xmean_mem13->Fill(xmean);  m_xstdev_mem13->Fill(xstdev);  m_xerronmean_mem13->Fill(xerronmean);
	       m_xmean_me13->Fill(xmean);  m_xstdev_me13->Fill(xstdev);  m_xerronmean_me13->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 1  &&  id.ring() == 4) {
	       m_xmean_mep14->Fill(xmean);  m_xstdev_mep14->Fill(xstdev);  m_xerronmean_mep14->Fill(xerronmean);
	       m_xmean_me14->Fill(xmean);  m_xstdev_me14->Fill(xstdev);  m_xerronmean_me14->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  id.ring() == 4) {
	       m_xmean_mem14->Fill(xmean);  m_xstdev_mem14->Fill(xstdev);  m_xerronmean_mem14->Fill(xerronmean);
	       m_xmean_me14->Fill(xmean);  m_xstdev_me14->Fill(xstdev);  m_xerronmean_me14->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 2  &&  id.ring() == 1) {
	       m_xmean_mep21->Fill(xmean);  m_xstdev_mep21->Fill(xstdev);  m_xerronmean_mep21->Fill(xerronmean);
	       m_xmean_me21->Fill(xmean);  m_xstdev_me21->Fill(xstdev);  m_xerronmean_me21->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -2  &&  id.ring() == 1) {
	       m_xmean_mem21->Fill(xmean);  m_xstdev_mem21->Fill(xstdev);  m_xerronmean_mem21->Fill(xerronmean);
	       m_xmean_me21->Fill(xmean);  m_xstdev_me21->Fill(xstdev);  m_xerronmean_me21->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 2  &&  id.ring() == 2) {
	       m_xmean_mep22->Fill(xmean);  m_xstdev_mep22->Fill(xstdev);  m_xerronmean_mep22->Fill(xerronmean);
	       m_xmean_me22->Fill(xmean);  m_xstdev_me22->Fill(xstdev);  m_xerronmean_me22->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -2  &&  id.ring() == 2) {
	       m_xmean_mem22->Fill(xmean);  m_xstdev_mem22->Fill(xstdev);  m_xerronmean_mem22->Fill(xerronmean);
	       m_xmean_me22->Fill(xmean);  m_xstdev_me22->Fill(xstdev);  m_xerronmean_me22->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 3  &&  id.ring() == 1) {
	       m_xmean_mep31->Fill(xmean);  m_xstdev_mep31->Fill(xstdev);  m_xerronmean_mep31->Fill(xerronmean);
	       m_xmean_me31->Fill(xmean);  m_xstdev_me31->Fill(xstdev);  m_xerronmean_me31->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -3  &&  id.ring() == 1) {
	       m_xmean_mem31->Fill(xmean);  m_xstdev_mem31->Fill(xstdev);  m_xerronmean_mem31->Fill(xerronmean);
	       m_xmean_me31->Fill(xmean);  m_xstdev_me31->Fill(xstdev);  m_xerronmean_me31->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 3  &&  id.ring() == 2) {
	       m_xmean_mep32->Fill(xmean);  m_xstdev_mep32->Fill(xstdev);  m_xerronmean_mep32->Fill(xerronmean);
	       m_xmean_me32->Fill(xmean);  m_xstdev_me32->Fill(xstdev);  m_xerronmean_me32->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -3  &&  id.ring() == 2) {
	       m_xmean_mem32->Fill(xmean);  m_xstdev_mem32->Fill(xstdev);  m_xerronmean_mem32->Fill(xerronmean);
	       m_xmean_me32->Fill(xmean);  m_xstdev_me32->Fill(xstdev);  m_xerronmean_me32->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 4  &&  id.ring() == 1) {
	       m_xmean_mep41->Fill(xmean);  m_xstdev_mep41->Fill(xstdev);  m_xerronmean_mep41->Fill(xerronmean);
	       m_xmean_me41->Fill(xmean);  m_xstdev_me41->Fill(xstdev);  m_xerronmean_me41->Fill(xerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -4  &&  id.ring() == 1) {
	       m_xmean_mem41->Fill(xmean);  m_xstdev_mem41->Fill(xstdev);  m_xerronmean_mem41->Fill(xerronmean);
	       m_xmean_me41->Fill(xmean);  m_xstdev_me41->Fill(xstdev);  m_xerronmean_me41->Fill(xerronmean);
	    }
	 } // else itis CSC
      } // end if xmean, xstdev exist
      
      if (m_numy[id] > 0.) {
	 double ymean = m_y_wy[id] / m_y_w[id];
	 double ystdev = sqrt(((m_y_wyy[id] * m_y_w[id]) - (m_y_wy[id] * m_y_wy[id]))/((m_y_w[id] * m_y_w[id]) - m_y_ww[id]));
	 double yerronmean = ystdev / sqrt(m_numy[id]);

	 m_ysummary->SetBinContent(index, ymean);
	 m_ysummary->SetBinError(index, yerronmean);

	 m_ymean->Fill(ymean);  m_ystdev->Fill(ystdev);  m_yerronmean->Fill(yerronmean);
	 if ((*chamber)->geomDetId().subdetId() == MuonSubdetId::DT) {
	    m_ymean_mb->Fill(ymean);  m_ystdev_mb->Fill(ystdev);  m_yerronmean_mb->Fill(yerronmean);
	    DTChamberId id((*chamber)->geomDetId().rawId());
	    if (id.station() == 1) {
	       m_ymean_mb1->Fill(ymean);  m_ystdev_mb1->Fill(ystdev);  m_yerronmean_mb1->Fill(yerronmean);
	    }
	    else if (id.station() == 2) {
	       m_ymean_mb2->Fill(ymean);  m_ystdev_mb2->Fill(ystdev);  m_yerronmean_mb2->Fill(yerronmean);
	    }
	    else if (id.station() == 3) {
	       m_ymean_mb3->Fill(ymean);  m_ystdev_mb3->Fill(ystdev);  m_yerronmean_mb3->Fill(yerronmean);
	    }
	    else if (id.station() == 4) {
	       m_ymean_mb4->Fill(ymean);  m_ystdev_mb4->Fill(ystdev);  m_yerronmean_mb4->Fill(yerronmean);
	    }

	    if (id.wheel() == -2) {
	       m_ymean_minus2->Fill(ymean);  m_ystdev_minus2->Fill(ystdev);  m_yerronmean_minus2->Fill(yerronmean);
	    }
	    else if (id.wheel() == -1) {
	       m_ymean_minus1->Fill(ymean);  m_ystdev_minus1->Fill(ystdev);  m_yerronmean_minus1->Fill(yerronmean);
	    }
	    else if (id.wheel() == 0) {
	       m_ymean_zero->Fill(ymean);  m_ystdev_zero->Fill(ystdev);  m_yerronmean_zero->Fill(yerronmean);
	    }
	    else if (id.wheel() == 1) {
	       m_ymean_plus1->Fill(ymean);  m_ystdev_plus1->Fill(ystdev);  m_yerronmean_plus1->Fill(yerronmean);
	    }
	    else if (id.wheel() == 2) {
	       m_ymean_plus2->Fill(ymean);  m_ystdev_plus2->Fill(ystdev);  m_yerronmean_plus2->Fill(yerronmean);
	    }
	 } // end if DT
	 else {
	    m_ymean_me->Fill(ymean);  m_ystdev_me->Fill(ystdev);  m_yerronmean_me->Fill(yerronmean);

	    CSCDetId id((*chamber)->geomDetId().rawId());

	    if ((id.endcap() == 1? 1: -1)*id.station() == 1  &&  id.ring() == 1) {
	       m_ymean_mep11->Fill(ymean);  m_ystdev_mep11->Fill(ystdev);  m_yerronmean_mep11->Fill(yerronmean);
	       m_ymean_me11->Fill(ymean);  m_ystdev_me11->Fill(ystdev);  m_yerronmean_me11->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  id.ring() == 1) {
	       m_ymean_mem11->Fill(ymean);  m_ystdev_mem11->Fill(ystdev);  m_yerronmean_mem11->Fill(yerronmean);
	       m_ymean_me11->Fill(ymean);  m_ystdev_me11->Fill(ystdev);  m_yerronmean_me11->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 1  &&  id.ring() == 2) {
	       m_ymean_mep12->Fill(ymean);  m_ystdev_mep12->Fill(ystdev);  m_yerronmean_mep12->Fill(yerronmean);
	       m_ymean_me12->Fill(ymean);  m_ystdev_me12->Fill(ystdev);  m_yerronmean_me12->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  id.ring() == 2) {
	       m_ymean_mem12->Fill(ymean);  m_ystdev_mem12->Fill(ystdev);  m_yerronmean_mem12->Fill(yerronmean);
	       m_ymean_me12->Fill(ymean);  m_ystdev_me12->Fill(ystdev);  m_yerronmean_me12->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 1  &&  id.ring() == 3) {
	       m_ymean_mep13->Fill(ymean);  m_ystdev_mep13->Fill(ystdev);  m_yerronmean_mep13->Fill(yerronmean);
	       m_ymean_me13->Fill(ymean);  m_ystdev_me13->Fill(ystdev);  m_yerronmean_me13->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  id.ring() == 3) {
	       m_ymean_mem13->Fill(ymean);  m_ystdev_mem13->Fill(ystdev);  m_yerronmean_mem13->Fill(yerronmean);
	       m_ymean_me13->Fill(ymean);  m_ystdev_me13->Fill(ystdev);  m_yerronmean_me13->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 1  &&  id.ring() == 4) {
	       m_ymean_mep14->Fill(ymean);  m_ystdev_mep14->Fill(ystdev);  m_yerronmean_mep14->Fill(yerronmean);
	       m_ymean_me14->Fill(ymean);  m_ystdev_me14->Fill(ystdev);  m_yerronmean_me14->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  id.ring() == 4) {
	       m_ymean_mem14->Fill(ymean);  m_ystdev_mem14->Fill(ystdev);  m_yerronmean_mem14->Fill(yerronmean);
	       m_ymean_me14->Fill(ymean);  m_ystdev_me14->Fill(ystdev);  m_yerronmean_me14->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 2  &&  id.ring() == 1) {
	       m_ymean_mep21->Fill(ymean);  m_ystdev_mep21->Fill(ystdev);  m_yerronmean_mep21->Fill(yerronmean);
	       m_ymean_me21->Fill(ymean);  m_ystdev_me21->Fill(ystdev);  m_yerronmean_me21->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -2  &&  id.ring() == 1) {
	       m_ymean_mem21->Fill(ymean);  m_ystdev_mem21->Fill(ystdev);  m_yerronmean_mem21->Fill(yerronmean);
	       m_ymean_me21->Fill(ymean);  m_ystdev_me21->Fill(ystdev);  m_yerronmean_me21->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 2  &&  id.ring() == 2) {
	       m_ymean_mep22->Fill(ymean);  m_ystdev_mep22->Fill(ystdev);  m_yerronmean_mep22->Fill(yerronmean);
	       m_ymean_me22->Fill(ymean);  m_ystdev_me22->Fill(ystdev);  m_yerronmean_me22->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -2  &&  id.ring() == 2) {
	       m_ymean_mem22->Fill(ymean);  m_ystdev_mem22->Fill(ystdev);  m_yerronmean_mem22->Fill(yerronmean);
	       m_ymean_me22->Fill(ymean);  m_ystdev_me22->Fill(ystdev);  m_yerronmean_me22->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 3  &&  id.ring() == 1) {
	       m_ymean_mep31->Fill(ymean);  m_ystdev_mep31->Fill(ystdev);  m_yerronmean_mep31->Fill(yerronmean);
	       m_ymean_me31->Fill(ymean);  m_ystdev_me31->Fill(ystdev);  m_yerronmean_me31->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -3  &&  id.ring() == 1) {
	       m_ymean_mem31->Fill(ymean);  m_ystdev_mem31->Fill(ystdev);  m_yerronmean_mem31->Fill(yerronmean);
	       m_ymean_me31->Fill(ymean);  m_ystdev_me31->Fill(ystdev);  m_yerronmean_me31->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 3  &&  id.ring() == 2) {
	       m_ymean_mep32->Fill(ymean);  m_ystdev_mep32->Fill(ystdev);  m_yerronmean_mep32->Fill(yerronmean);
	       m_ymean_me32->Fill(ymean);  m_ystdev_me32->Fill(ystdev);  m_yerronmean_me32->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -3  &&  id.ring() == 2) {
	       m_ymean_mem32->Fill(ymean);  m_ystdev_mem32->Fill(ystdev);  m_yerronmean_mem32->Fill(yerronmean);
	       m_ymean_me32->Fill(ymean);  m_ystdev_me32->Fill(ystdev);  m_yerronmean_me32->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 4  &&  id.ring() == 1) {
	       m_ymean_mep41->Fill(ymean);  m_ystdev_mep41->Fill(ystdev);  m_yerronmean_mep41->Fill(yerronmean);
	       m_ymean_me41->Fill(ymean);  m_ystdev_me41->Fill(ystdev);  m_yerronmean_me41->Fill(yerronmean);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -4  &&  id.ring() == 1) {
	       m_ymean_mem41->Fill(ymean);  m_ystdev_mem41->Fill(ystdev);  m_yerronmean_mem41->Fill(yerronmean);
	       m_ymean_me41->Fill(ymean);  m_ystdev_me41->Fill(ystdev);  m_yerronmean_me41->Fill(yerronmean);
	    }
	 } // else itis CSC
      } // end if ymean, ystdev exist

   } // end loop over chambers
}

//
// constructors and destructor
//

// AlignmentMonitorMuonResiduals::AlignmentMonitorMuonResiduals(const AlignmentMonitorMuonResiduals& rhs)
// {
//    // do actual copying here;
// }

//
// assignment operators
//
// const AlignmentMonitorMuonResiduals& AlignmentMonitorMuonResiduals::operator=(const AlignmentMonitorMuonResiduals& rhs)
// {
//   //An exception safe implementation is
//   AlignmentMonitorMuonResiduals temp(rhs);
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

DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorMuonResiduals, "AlignmentMonitorMuonResiduals");
