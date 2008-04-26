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
#include "TH1F.h"

// user include files

// 
// class definition
// 

class AlignmentMonitorMuonResiduals: public AlignmentMonitorBase {
   public:
      AlignmentMonitorMuonResiduals(const edm::ParameterSet& cfg);
      ~AlignmentMonitorMuonResiduals() {};

      void book();
      void event(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& iTrajTracks);
      void afterAlignment(const edm::EventSetup &iSetup);

   private:
      std::map<int, double> m_nx;
      std::map<int, double> m_x1;
      std::map<int, double> m_x2;
      std::map<int, double> m_ny;
      std::map<int, double> m_y1;
      std::map<int, double> m_y2;

      TH1F *m_xresid, *m_xresid_mb, *m_xresid_me,
	 *m_xresid_mb1, *m_xresid_mb2, *m_xresid_mb3, *m_xresid_mb4,
	 *m_xresid_minus2, *m_xresid_minus1, *m_xresid_zero, *m_xresid_plus1, *m_xresid_plus2,
	 *m_xresid_mep11, *m_xresid_mep12, *m_xresid_mep13,
	 *m_xresid_mep21, *m_xresid_mep22, *m_xresid_mep31, *m_xresid_mep32, *m_xresid_mep41,
	 *m_xresid_mem11, *m_xresid_mem12, *m_xresid_mem13,
	 *m_xresid_mem21, *m_xresid_mem22, *m_xresid_mem31, *m_xresid_mem32, *m_xresid_mem41,
	 *m_xresid_me11, *m_xresid_me12, *m_xresid_me13,
	 *m_xresid_me21, *m_xresid_me22, *m_xresid_me31, *m_xresid_me32, *m_xresid_me41;

      TH1F *m_xmean, *m_xmean_mb, *m_xmean_me,
	 *m_xmean_mb1, *m_xmean_mb2, *m_xmean_mb3, *m_xmean_mb4,
	 *m_xmean_minus2, *m_xmean_minus1, *m_xmean_zero, *m_xmean_plus1, *m_xmean_plus2,
	 *m_xmean_mep11, *m_xmean_mep12, *m_xmean_mep13,
	 *m_xmean_mep21, *m_xmean_mep22, *m_xmean_mep31, *m_xmean_mep32, *m_xmean_mep41,
	 *m_xmean_mem11, *m_xmean_mem12, *m_xmean_mem13,
	 *m_xmean_mem21, *m_xmean_mem22, *m_xmean_mem31, *m_xmean_mem32, *m_xmean_mem41,
	 *m_xmean_me11, *m_xmean_me12, *m_xmean_me13,
	 *m_xmean_me21, *m_xmean_me22, *m_xmean_me31, *m_xmean_me32, *m_xmean_me41;

      TH1F *m_xstdev, *m_xstdev_mb, *m_xstdev_me,
	 *m_xstdev_mb1, *m_xstdev_mb2, *m_xstdev_mb3, *m_xstdev_mb4,
	 *m_xstdev_minus2, *m_xstdev_minus1, *m_xstdev_zero, *m_xstdev_plus1, *m_xstdev_plus2,
	 *m_xstdev_mep11, *m_xstdev_mep12, *m_xstdev_mep13,
	 *m_xstdev_mep21, *m_xstdev_mep22, *m_xstdev_mep31, *m_xstdev_mep32, *m_xstdev_mep41,
	 *m_xstdev_mem11, *m_xstdev_mem12, *m_xstdev_mem13,
	 *m_xstdev_mem21, *m_xstdev_mem22, *m_xstdev_mem31, *m_xstdev_mem32, *m_xstdev_mem41,
	 *m_xstdev_me11, *m_xstdev_me12, *m_xstdev_me13,
	 *m_xstdev_me21, *m_xstdev_me22, *m_xstdev_me31, *m_xstdev_me32, *m_xstdev_me41;

      TH1F *m_yresid, *m_yresid_mb, *m_yresid_me,
	 *m_yresid_mb1, *m_yresid_mb2, *m_yresid_mb3, *m_yresid_mb4,
	 *m_yresid_minus2, *m_yresid_minus1, *m_yresid_zero, *m_yresid_plus1, *m_yresid_plus2,
	 *m_yresid_mep11, *m_yresid_mep12, *m_yresid_mep13,
	 *m_yresid_mep21, *m_yresid_mep22, *m_yresid_mep31, *m_yresid_mep32, *m_yresid_mep41,
	 *m_yresid_mem11, *m_yresid_mem12, *m_yresid_mem13,
	 *m_yresid_mem21, *m_yresid_mem22, *m_yresid_mem31, *m_yresid_mem32, *m_yresid_mem41,
	 *m_yresid_me11, *m_yresid_me12, *m_yresid_me13,
	 *m_yresid_me21, *m_yresid_me22, *m_yresid_me31, *m_yresid_me32, *m_yresid_me41;

      TH1F *m_ymean, *m_ymean_mb, *m_ymean_me,
	 *m_ymean_mb1, *m_ymean_mb2, *m_ymean_mb3, *m_ymean_mb4,
	 *m_ymean_minus2, *m_ymean_minus1, *m_ymean_zero, *m_ymean_plus1, *m_ymean_plus2,
	 *m_ymean_mep11, *m_ymean_mep12, *m_ymean_mep13,
	 *m_ymean_mep21, *m_ymean_mep22, *m_ymean_mep31, *m_ymean_mep32, *m_ymean_mep41,
	 *m_ymean_mem11, *m_ymean_mem12, *m_ymean_mem13,
	 *m_ymean_mem21, *m_ymean_mem22, *m_ymean_mem31, *m_ymean_mem32, *m_ymean_mem41,
	 *m_ymean_me11, *m_ymean_me12, *m_ymean_me13,
	 *m_ymean_me21, *m_ymean_me22, *m_ymean_me31, *m_ymean_me32, *m_ymean_me41;

      TH1F *m_ystdev, *m_ystdev_mb, *m_ystdev_me,
	 *m_ystdev_mb1, *m_ystdev_mb2, *m_ystdev_mb3, *m_ystdev_mb4,
	 *m_ystdev_minus2, *m_ystdev_minus1, *m_ystdev_zero, *m_ystdev_plus1, *m_ystdev_plus2,
	 *m_ystdev_mep11, *m_ystdev_mep12, *m_ystdev_mep13,
	 *m_ystdev_mep21, *m_ystdev_mep22, *m_ystdev_mep31, *m_ystdev_mep32, *m_ystdev_mep41,
	 *m_ystdev_mem11, *m_ystdev_mem12, *m_ystdev_mem13,
	 *m_ystdev_mem21, *m_ystdev_mem22, *m_ystdev_mem31, *m_ystdev_mem32, *m_ystdev_mem41,
	 *m_ystdev_me11, *m_ystdev_me12, *m_ystdev_me13,
	 *m_ystdev_me21, *m_ystdev_me22, *m_ystdev_me31, *m_ystdev_me32, *m_ystdev_me41;

      unsigned int xresid_bins, xmean_bins, xstdev_bins, yresid_bins, ymean_bins, ystdev_bins;
      double xresid_low, xresid_high, xmean_low, xmean_high, xstdev_low, xstdev_high, yresid_low, yresid_high, ymean_low, ymean_high, ystdev_low, ystdev_high;
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
   yresid_bins = cfg.getParameter<unsigned int>("yresid_bins");
   ymean_bins = cfg.getParameter<unsigned int>("ymean_bins");
   ystdev_bins = cfg.getParameter<unsigned int>("ystdev_bins");
   xresid_low = cfg.getParameter<double>("xresid_low");
   xresid_high = cfg.getParameter<double>("xresid_high");
   xmean_low = cfg.getParameter<double>("xmean_low");
   xmean_high = cfg.getParameter<double>("xmean_high");
   xstdev_low = cfg.getParameter<double>("xstdev_low");
   xstdev_high = cfg.getParameter<double>("xstdev_high");
   yresid_low = cfg.getParameter<double>("yresid_low");
   yresid_high = cfg.getParameter<double>("yresid_high");
   ymean_low = cfg.getParameter<double>("ymean_low");
   ymean_high = cfg.getParameter<double>("ymean_high");
   ystdev_low = cfg.getParameter<double>("ystdev_low");
   ystdev_high = cfg.getParameter<double>("ystdev_high");
}

void AlignmentMonitorMuonResiduals::book() {
   m_nx.clear();
   m_x1.clear();
   m_x2.clear();
   m_ny.clear();
   m_y1.clear();
   m_y2.clear();

   std::vector<Alignable*> chambers;
   std::vector<Alignable*> tmp1 = pMuon()->DTChambers();
   for (std::vector<Alignable*>::const_iterator iter = tmp1.begin();  iter != tmp1.end();  ++iter) chambers.push_back(*iter);
   std::vector<Alignable*> tmp2 = pMuon()->CSCChambers();
   for (std::vector<Alignable*>::const_iterator iter = tmp2.begin();  iter != tmp2.end();  ++iter) chambers.push_back(*iter);

   for (std::vector<Alignable*>::const_iterator chamber = chambers.begin();  chamber != chambers.end();  ++chamber) {
      int id = (*chamber)->geomDetId().rawId();
      m_nx[id] = 0;
      m_x1[id] = 0;
      m_x2[id] = 0;
      m_ny[id] = 0;
      m_y1[id] = 0;
      m_y2[id] = 0;
   }

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
   m_xresid_mep21 = book1D("/iterN/mep21/", "xresid_mep21", "ME+2/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep22 = book1D("/iterN/mep22/", "xresid_mep22", "ME+2/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep31 = book1D("/iterN/mep31/", "xresid_mep31", "ME+3/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep32 = book1D("/iterN/mep32/", "xresid_mep32", "ME+3/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mep41 = book1D("/iterN/mep41/", "xresid_mep41", "ME+4/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem11 = book1D("/iterN/mem11/", "xresid_mem11", "ME-1/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem12 = book1D("/iterN/mem12/", "xresid_mem12", "ME-1/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem13 = book1D("/iterN/mem13/", "xresid_mem13", "ME-1/3 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem21 = book1D("/iterN/mem21/", "xresid_mem21", "ME-2/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem22 = book1D("/iterN/mem22/", "xresid_mem22", "ME-2/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem31 = book1D("/iterN/mem31/", "xresid_mem31", "ME-3/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem32 = book1D("/iterN/mem32/", "xresid_mem32", "ME-3/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_mem41 = book1D("/iterN/mem41/", "xresid_mem41", "ME-4/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me11 = book1D("/iterN/me11/", "xresid_me11", "ME1/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me12 = book1D("/iterN/me12/", "xresid_me12", "ME1/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me13 = book1D("/iterN/me13/", "xresid_me13", "ME1/3 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me21 = book1D("/iterN/me21/", "xresid_me21", "ME2/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me22 = book1D("/iterN/me22/", "xresid_me22", "ME2/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me31 = book1D("/iterN/me31/", "xresid_me31", "ME3/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me32 = book1D("/iterN/me32/", "xresid_me32", "ME3/2 x residual (mm)", xresid_bins, xresid_low, xresid_high);
   m_xresid_me41 = book1D("/iterN/me41/", "xresid_me41", "ME4/1 x residual (mm)", xresid_bins, xresid_low, xresid_high);

   m_xmean = book1D("/iterN/", "xmean", "mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mb = book1D("/iterN/mb/", "xmean_mb", "barrel mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me = book1D("/iterN/me/", "xmean_me", "endcap mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mb1 = book1D("/iterN/mb1/", "xmean_mb1", "MB station 1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mb2 = book1D("/iterN/mb2/", "xmean_mb2", "MB station 2 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mb3 = book1D("/iterN/mb3/", "xmean_mb3", "MB station 3 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mb4 = book1D("/iterN/mb4/", "xmean_mb4", "MB station 4 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_minus2 = book1D("/iterN/minus2/", "xmean_minus2", "MB wheel -2 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_minus1 = book1D("/iterN/minus1/", "xmean_minus1", "MB wheel -1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_zero = book1D("/iterN/zero/", "xmean_zero", "MB wheel 0 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_plus1 = book1D("/iterN/plus1/", "xmean_plus1", "MB wheel +1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_plus2 = book1D("/iterN/plus2/", "xmean_plus2", "MB wheel +2 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep11 = book1D("/iterN/mep11/", "xmean_mep11", "ME+1/1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep12 = book1D("/iterN/mep12/", "xmean_mep12", "ME+1/2 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep13 = book1D("/iterN/mep13/", "xmean_mep13", "ME+1/3 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep21 = book1D("/iterN/mep21/", "xmean_mep21", "ME+2/1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep22 = book1D("/iterN/mep22/", "xmean_mep22", "ME+2/2 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep31 = book1D("/iterN/mep31/", "xmean_mep31", "ME+3/1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep32 = book1D("/iterN/mep32/", "xmean_mep32", "ME+3/2 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mep41 = book1D("/iterN/mep41/", "xmean_mep41", "ME+4/1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem11 = book1D("/iterN/mem11/", "xmean_mem11", "ME-1/1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem12 = book1D("/iterN/mem12/", "xmean_mem12", "ME-1/2 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem13 = book1D("/iterN/mem13/", "xmean_mem13", "ME-1/3 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem21 = book1D("/iterN/mem21/", "xmean_mem21", "ME-2/1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem22 = book1D("/iterN/mem22/", "xmean_mem22", "ME-2/2 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem31 = book1D("/iterN/mem31/", "xmean_mem31", "ME-3/1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem32 = book1D("/iterN/mem32/", "xmean_mem32", "ME-3/2 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_mem41 = book1D("/iterN/mem41/", "xmean_mem41", "ME-4/1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me11 = book1D("/iterN/me11/", "xmean_me11", "ME1/1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me12 = book1D("/iterN/me12/", "xmean_me12", "ME1/2 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me13 = book1D("/iterN/me13/", "xmean_me13", "ME1/3 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me21 = book1D("/iterN/me21/", "xmean_me21", "ME2/1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me22 = book1D("/iterN/me22/", "xmean_me22", "ME2/2 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me31 = book1D("/iterN/me31/", "xmean_me31", "ME3/1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me32 = book1D("/iterN/me32/", "xmean_me32", "ME3/2 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);
   m_xmean_me41 = book1D("/iterN/me41/", "xmean_me41", "ME4/1 mean x residual per chamber (mm)", xmean_bins, xmean_low, xmean_high);

   m_xstdev = book1D("/iterN/", "xstdev", "stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mb = book1D("/iterN/mb/", "xstdev_mb", "barrel stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me = book1D("/iterN/me/", "xstdev_me", "endcap stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mb1 = book1D("/iterN/mb1/", "xstdev_mb1", "MB station 1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mb2 = book1D("/iterN/mb2/", "xstdev_mb2", "MB station 2 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mb3 = book1D("/iterN/mb3/", "xstdev_mb3", "MB station 3 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mb4 = book1D("/iterN/mb4/", "xstdev_mb4", "MB station 4 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_minus2 = book1D("/iterN/minus2/", "xstdev_minus2", "MB wheel -2 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_minus1 = book1D("/iterN/minus1/", "xstdev_minus1", "MB wheel -1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_zero = book1D("/iterN/zero/", "xstdev_zero", "MB wheel 0 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_plus1 = book1D("/iterN/plus1/", "xstdev_plus1", "MB wheel +1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_plus2 = book1D("/iterN/plus2/", "xstdev_plus2", "MB wheel +2 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep11 = book1D("/iterN/mep11/", "xstdev_mep11", "ME+1/1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep12 = book1D("/iterN/mep12/", "xstdev_mep12", "ME+1/2 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep13 = book1D("/iterN/mep13/", "xstdev_mep13", "ME+1/3 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep21 = book1D("/iterN/mep21/", "xstdev_mep21", "ME+2/1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep22 = book1D("/iterN/mep22/", "xstdev_mep22", "ME+2/2 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep31 = book1D("/iterN/mep31/", "xstdev_mep31", "ME+3/1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep32 = book1D("/iterN/mep32/", "xstdev_mep32", "ME+3/2 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mep41 = book1D("/iterN/mep41/", "xstdev_mep41", "ME+4/1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem11 = book1D("/iterN/mem11/", "xstdev_mem11", "ME-1/1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem12 = book1D("/iterN/mem12/", "xstdev_mem12", "ME-1/2 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem13 = book1D("/iterN/mem13/", "xstdev_mem13", "ME-1/3 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem21 = book1D("/iterN/mem21/", "xstdev_mem21", "ME-2/1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem22 = book1D("/iterN/mem22/", "xstdev_mem22", "ME-2/2 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem31 = book1D("/iterN/mem31/", "xstdev_mem31", "ME-3/1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem32 = book1D("/iterN/mem32/", "xstdev_mem32", "ME-3/2 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_mem41 = book1D("/iterN/mem41/", "xstdev_mem41", "ME-4/1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me11 = book1D("/iterN/me11/", "xstdev_me11", "ME1/1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me12 = book1D("/iterN/me12/", "xstdev_me12", "ME1/2 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me13 = book1D("/iterN/me13/", "xstdev_me13", "ME1/3 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me21 = book1D("/iterN/me21/", "xstdev_me21", "ME2/1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me22 = book1D("/iterN/me22/", "xstdev_me22", "ME2/2 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me31 = book1D("/iterN/me31/", "xstdev_me31", "ME3/1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me32 = book1D("/iterN/me32/", "xstdev_me32", "ME3/2 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);
   m_xstdev_me41 = book1D("/iterN/me41/", "xstdev_me41", "ME4/1 stdev x residual per chamber (mm)", xstdev_bins, xstdev_low, xstdev_high);

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
   m_yresid_mep21 = book1D("/iterN/mep21/", "yresid_mep21", "ME+2/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep22 = book1D("/iterN/mep22/", "yresid_mep22", "ME+2/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep31 = book1D("/iterN/mep31/", "yresid_mep31", "ME+3/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep32 = book1D("/iterN/mep32/", "yresid_mep32", "ME+3/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mep41 = book1D("/iterN/mep41/", "yresid_mep41", "ME+4/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem11 = book1D("/iterN/mem11/", "yresid_mem11", "ME-1/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem12 = book1D("/iterN/mem12/", "yresid_mem12", "ME-1/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem13 = book1D("/iterN/mem13/", "yresid_mem13", "ME-1/3 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem21 = book1D("/iterN/mem21/", "yresid_mem21", "ME-2/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem22 = book1D("/iterN/mem22/", "yresid_mem22", "ME-2/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem31 = book1D("/iterN/mem31/", "yresid_mem31", "ME-3/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem32 = book1D("/iterN/mem32/", "yresid_mem32", "ME-3/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_mem41 = book1D("/iterN/mem41/", "yresid_mem41", "ME-4/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me11 = book1D("/iterN/me11/", "yresid_me11", "ME1/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me12 = book1D("/iterN/me12/", "yresid_me12", "ME1/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me13 = book1D("/iterN/me13/", "yresid_me13", "ME1/3 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me21 = book1D("/iterN/me21/", "yresid_me21", "ME2/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me22 = book1D("/iterN/me22/", "yresid_me22", "ME2/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me31 = book1D("/iterN/me31/", "yresid_me31", "ME3/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me32 = book1D("/iterN/me32/", "yresid_me32", "ME3/2 y residual (mm)", yresid_bins, yresid_low, yresid_high);
   m_yresid_me41 = book1D("/iterN/me41/", "yresid_me41", "ME4/1 y residual (mm)", yresid_bins, yresid_low, yresid_high);

   m_ymean = book1D("/iterN/", "ymean", "mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mb = book1D("/iterN/mb/", "ymean_mb", "barrel mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me = book1D("/iterN/me/", "ymean_me", "endcap mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mb1 = book1D("/iterN/mb1/", "ymean_mb1", "MB station 1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mb2 = book1D("/iterN/mb2/", "ymean_mb2", "MB station 2 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mb3 = book1D("/iterN/mb3/", "ymean_mb3", "MB station 3 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mb4 = book1D("/iterN/mb4/", "ymean_mb4", "MB station 4 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_minus2 = book1D("/iterN/minus2/", "ymean_minus2", "MB wheel -2 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_minus1 = book1D("/iterN/minus1/", "ymean_minus1", "MB wheel -1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_zero = book1D("/iterN/zero/", "ymean_zero", "MB wheel 0 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_plus1 = book1D("/iterN/plus1/", "ymean_plus1", "MB wheel +1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_plus2 = book1D("/iterN/plus2/", "ymean_plus2", "MB wheel +2 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep11 = book1D("/iterN/mep11/", "ymean_mep11", "ME+1/1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep12 = book1D("/iterN/mep12/", "ymean_mep12", "ME+1/2 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep13 = book1D("/iterN/mep13/", "ymean_mep13", "ME+1/3 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep21 = book1D("/iterN/mep21/", "ymean_mep21", "ME+2/1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep22 = book1D("/iterN/mep22/", "ymean_mep22", "ME+2/2 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep31 = book1D("/iterN/mep31/", "ymean_mep31", "ME+3/1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep32 = book1D("/iterN/mep32/", "ymean_mep32", "ME+3/2 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mep41 = book1D("/iterN/mep41/", "ymean_mep41", "ME+4/1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem11 = book1D("/iterN/mem11/", "ymean_mem11", "ME-1/1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem12 = book1D("/iterN/mem12/", "ymean_mem12", "ME-1/2 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem13 = book1D("/iterN/mem13/", "ymean_mem13", "ME-1/3 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem21 = book1D("/iterN/mem21/", "ymean_mem21", "ME-2/1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem22 = book1D("/iterN/mem22/", "ymean_mem22", "ME-2/2 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem31 = book1D("/iterN/mem31/", "ymean_mem31", "ME-3/1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem32 = book1D("/iterN/mem32/", "ymean_mem32", "ME-3/2 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_mem41 = book1D("/iterN/mem41/", "ymean_mem41", "ME-4/1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me11 = book1D("/iterN/me11/", "ymean_me11", "ME1/1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me12 = book1D("/iterN/me12/", "ymean_me12", "ME1/2 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me13 = book1D("/iterN/me13/", "ymean_me13", "ME1/3 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me21 = book1D("/iterN/me21/", "ymean_me21", "ME2/1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me22 = book1D("/iterN/me22/", "ymean_me22", "ME2/2 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me31 = book1D("/iterN/me31/", "ymean_me31", "ME3/1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me32 = book1D("/iterN/me32/", "ymean_me32", "ME3/2 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);
   m_ymean_me41 = book1D("/iterN/me41/", "ymean_me41", "ME4/1 mean y residual per chamber (mm)", ymean_bins, ymean_low, ymean_high);

   m_ystdev = book1D("/iterN/", "ystdev", "stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mb = book1D("/iterN/mb/", "ystdev_mb", "barrel stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me = book1D("/iterN/me/", "ystdev_me", "endcap stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mb1 = book1D("/iterN/mb1/", "ystdev_mb1", "MB station 1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mb2 = book1D("/iterN/mb2/", "ystdev_mb2", "MB station 2 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mb3 = book1D("/iterN/mb3/", "ystdev_mb3", "MB station 3 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mb4 = book1D("/iterN/mb4/", "ystdev_mb4", "MB station 4 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_minus2 = book1D("/iterN/minus2/", "ystdev_minus2", "MB wheel -2 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_minus1 = book1D("/iterN/minus1/", "ystdev_minus1", "MB wheel -1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_zero = book1D("/iterN/zero/", "ystdev_zero", "MB wheel 0 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_plus1 = book1D("/iterN/plus1/", "ystdev_plus1", "MB wheel +1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_plus2 = book1D("/iterN/plus2/", "ystdev_plus2", "MB wheel +2 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep11 = book1D("/iterN/mep11/", "ystdev_mep11", "ME+1/1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep12 = book1D("/iterN/mep12/", "ystdev_mep12", "ME+1/2 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep13 = book1D("/iterN/mep13/", "ystdev_mep13", "ME+1/3 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep21 = book1D("/iterN/mep21/", "ystdev_mep21", "ME+2/1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep22 = book1D("/iterN/mep22/", "ystdev_mep22", "ME+2/2 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep31 = book1D("/iterN/mep31/", "ystdev_mep31", "ME+3/1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep32 = book1D("/iterN/mep32/", "ystdev_mep32", "ME+3/2 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mep41 = book1D("/iterN/mep41/", "ystdev_mep41", "ME+4/1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem11 = book1D("/iterN/mem11/", "ystdev_mem11", "ME-1/1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem12 = book1D("/iterN/mem12/", "ystdev_mem12", "ME-1/2 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem13 = book1D("/iterN/mem13/", "ystdev_mem13", "ME-1/3 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem21 = book1D("/iterN/mem21/", "ystdev_mem21", "ME-2/1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem22 = book1D("/iterN/mem22/", "ystdev_mem22", "ME-2/2 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem31 = book1D("/iterN/mem31/", "ystdev_mem31", "ME-3/1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem32 = book1D("/iterN/mem32/", "ystdev_mem32", "ME-3/2 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_mem41 = book1D("/iterN/mem41/", "ystdev_mem41", "ME-4/1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me11 = book1D("/iterN/me11/", "ystdev_me11", "ME1/1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me12 = book1D("/iterN/me12/", "ystdev_me12", "ME1/2 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me13 = book1D("/iterN/me13/", "ystdev_me13", "ME1/3 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me21 = book1D("/iterN/me21/", "ystdev_me21", "ME2/1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me22 = book1D("/iterN/me22/", "ystdev_me22", "ME2/2 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me31 = book1D("/iterN/me31/", "ystdev_me31", "ME3/1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me32 = book1D("/iterN/me32/", "ystdev_me32", "ME3/2 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);
   m_ystdev_me41 = book1D("/iterN/me41/", "ystdev_me41", "ME4/1 stdev y residual per chamber (mm)", ystdev_bins, ystdev_low, ystdev_high);

}

void AlignmentMonitorMuonResiduals::event(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection& tracks) {
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
	    LocalPoint trackPos = tsosc.localPosition();
	    LocalError trackErr = tsosc.localError().positionError();
	    LocalPoint hitPos = hit->localPosition();
	    LocalError hitErr = hit->localPositionError();

	    double x_residual = trackPos.x() - hitPos.x();
	    double y_residual = trackPos.y() - hitPos.y();
	    double x_reserr2 = trackErr.xx() + hitErr.xx();
	    double y_reserr2 = trackErr.yy() + hitErr.yy();
// 	    double xpos = trackPos.x();
// 	    double ypos = trackPos.y();

	    if (id.subdetId() == MuonSubdetId::DT) {
	       if (fabs(hit->surface()->toGlobal(LocalVector(0,1,0)).z()) < 0.1) {
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
		  m_nx[rawId] += 1./x_reserr2;
		  m_x1[rawId] += x_residual / x_reserr2;
		  m_x2[rawId] += x_residual * x_residual / x_reserr2 / x_reserr2;
	       }
	       if (y_reserr2 > 0.) {
		  m_ny[rawId] += 1./y_reserr2;
		  m_y1[rawId] += y_residual / y_reserr2;
		  m_y2[rawId] += y_residual * y_residual / y_reserr2 / y_reserr2;
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

	    else {
	       m_xresid->Fill(x_residual, 1./x_reserr2);
	       m_yresid->Fill(y_residual, 1./y_reserr2);

	       m_xresid_me->Fill(x_residual, 1./x_reserr2);
	       m_yresid_me->Fill(y_residual, 1./y_reserr2);

	       CSCDetId cscId(id.rawId());
	       int rawId = cscId.chamberId().rawId();
	       if (x_reserr2 > 0.) {
		  m_nx[rawId] += 1./x_reserr2;
		  m_x1[rawId] += x_residual / x_reserr2;
		  m_x2[rawId] += x_residual * x_residual / x_reserr2 / x_reserr2;
	       }
	       if (y_reserr2 > 0.) {
		  m_ny[rawId] += 1./y_reserr2;
		  m_y1[rawId] += y_residual / y_reserr2;
		  m_y2[rawId] += y_residual * y_residual / y_reserr2 / y_reserr2;
	       }

	       if ((cscId.endcap() == 1? 1: -1)*cscId.station() == 1  &&  (cscId.ring() == 1  ||  cscId.ring() == 4)) {
		  m_xresid_mep11->Fill(x_residual, 1./x_reserr2);  m_yresid_mep11->Fill(y_residual, 1./y_reserr2);
		  m_xresid_me11->Fill(x_residual, 1./x_reserr2);  m_yresid_me11->Fill(y_residual, 1./y_reserr2);
	       }
	       else if ((cscId.endcap() == 1? 1: -1)*cscId.station() == -1  &&  (cscId.ring() == 1  ||  cscId.ring() == 4)) {
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

void AlignmentMonitorMuonResiduals::afterAlignment(const edm::EventSetup &iSetup) {
   std::vector<Alignable*> chambers;
   std::vector<Alignable*> tmp1 = pMuon()->DTChambers();
   for (std::vector<Alignable*>::const_iterator iter = tmp1.begin();  iter != tmp1.end();  ++iter) chambers.push_back(*iter);
   std::vector<Alignable*> tmp2 = pMuon()->CSCChambers();
   for (std::vector<Alignable*>::const_iterator iter = tmp2.begin();  iter != tmp2.end();  ++iter) chambers.push_back(*iter);

   for (std::vector<Alignable*>::const_iterator chamber = chambers.begin();  chamber != chambers.end();  ++chamber) {
      int id = (*chamber)->geomDetId().rawId();

      if (m_nx[id] > 0.) {
	 double xmean = m_x1[id] / m_nx[id];
	 double xstdev = sqrt(m_x2[id] / m_nx[id] - (m_x1[id] / m_nx[id]) * (m_x1[id] / m_nx[id]));

	 m_xmean->Fill(xmean);  m_xstdev->Fill(xstdev);
	 if ((*chamber)->geomDetId().subdetId() == MuonSubdetId::DT) {
	    m_xmean_mb->Fill(xmean);  m_xstdev_mb->Fill(xstdev);
	    DTChamberId id((*chamber)->geomDetId().rawId());
	    if (id.station() == 1) {
	       m_xmean_mb1->Fill(xmean);  m_xstdev_mb1->Fill(xstdev);
	    }
	    else if (id.station() == 2) {
	       m_xmean_mb2->Fill(xmean);  m_xstdev_mb2->Fill(xstdev);
	    }
	    else if (id.station() == 3) {
	       m_xmean_mb3->Fill(xmean);  m_xstdev_mb3->Fill(xstdev);
	    }
	    else if (id.station() == 4) {
	       m_xmean_mb4->Fill(xmean);  m_xstdev_mb4->Fill(xstdev);
	    }

	    if (id.wheel() == -2) {
	       m_xmean_minus2->Fill(xmean);  m_xstdev_minus2->Fill(xstdev);
	    }
	    else if (id.wheel() == -1) {
	       m_xmean_minus1->Fill(xmean);  m_xstdev_minus1->Fill(xstdev);
	    }
	    else if (id.wheel() == 0) {
	       m_xmean_zero->Fill(xmean);  m_xstdev_zero->Fill(xstdev);
	    }
	    else if (id.wheel() == 1) {
	       m_xmean_plus1->Fill(xmean);  m_xstdev_plus1->Fill(xstdev);
	    }
	    else if (id.wheel() == 2) {
	       m_xmean_plus2->Fill(xmean);  m_xstdev_plus2->Fill(xstdev);
	    }
	 } // end if DT
	 else {
	    m_xmean_me->Fill(xmean);  m_xstdev_me->Fill(xstdev);

	    CSCDetId id((*chamber)->geomDetId().rawId());

	    if ((id.endcap() == 1? 1: -1)*id.station() == 1  &&  (id.ring() == 1  ||  id.ring() == 4)) {
	       m_xmean_mep11->Fill(xmean);  m_xstdev_mep11->Fill(xstdev);
	       m_xmean_me11->Fill(xmean);  m_xstdev_me11->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  (id.ring() == 1  ||  id.ring() == 4)) {
	       m_xmean_mem11->Fill(xmean);  m_xstdev_mem11->Fill(xstdev);
	       m_xmean_me11->Fill(xmean);  m_xstdev_me11->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 1  &&  id.ring() == 2) {
	       m_xmean_mep12->Fill(xmean);  m_xstdev_mep12->Fill(xstdev);
	       m_xmean_me12->Fill(xmean);  m_xstdev_me12->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  id.ring() == 2) {
	       m_xmean_mem12->Fill(xmean);  m_xstdev_mem12->Fill(xstdev);
	       m_xmean_me12->Fill(xmean);  m_xstdev_me12->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 1  &&  id.ring() == 3) {
	       m_xmean_mep13->Fill(xmean);  m_xstdev_mep13->Fill(xstdev);
	       m_xmean_me13->Fill(xmean);  m_xstdev_me13->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  id.ring() == 3) {
	       m_xmean_mem13->Fill(xmean);  m_xstdev_mem13->Fill(xstdev);
	       m_xmean_me13->Fill(xmean);  m_xstdev_me13->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 2  &&  id.ring() == 1) {
	       m_xmean_mep21->Fill(xmean);  m_xstdev_mep21->Fill(xstdev);
	       m_xmean_me21->Fill(xmean);  m_xstdev_me21->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -2  &&  id.ring() == 1) {
	       m_xmean_mem21->Fill(xmean);  m_xstdev_mem21->Fill(xstdev);
	       m_xmean_me21->Fill(xmean);  m_xstdev_me21->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 2  &&  id.ring() == 2) {
	       m_xmean_mep22->Fill(xmean);  m_xstdev_mep22->Fill(xstdev);
	       m_xmean_me22->Fill(xmean);  m_xstdev_me22->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -2  &&  id.ring() == 2) {
	       m_xmean_mem22->Fill(xmean);  m_xstdev_mem22->Fill(xstdev);
	       m_xmean_me22->Fill(xmean);  m_xstdev_me22->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 3  &&  id.ring() == 1) {
	       m_xmean_mep31->Fill(xmean);  m_xstdev_mep31->Fill(xstdev);
	       m_xmean_me31->Fill(xmean);  m_xstdev_me31->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -3  &&  id.ring() == 1) {
	       m_xmean_mem31->Fill(xmean);  m_xstdev_mem31->Fill(xstdev);
	       m_xmean_me31->Fill(xmean);  m_xstdev_me31->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 3  &&  id.ring() == 2) {
	       m_xmean_mep32->Fill(xmean);  m_xstdev_mep32->Fill(xstdev);
	       m_xmean_me32->Fill(xmean);  m_xstdev_me32->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -3  &&  id.ring() == 2) {
	       m_xmean_mem32->Fill(xmean);  m_xstdev_mem32->Fill(xstdev);
	       m_xmean_me32->Fill(xmean);  m_xstdev_me32->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 4  &&  id.ring() == 1) {
	       m_xmean_mep41->Fill(xmean);  m_xstdev_mep41->Fill(xstdev);
	       m_xmean_me41->Fill(xmean);  m_xstdev_me41->Fill(xstdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -4  &&  id.ring() == 1) {
	       m_xmean_mem41->Fill(xmean);  m_xstdev_mem41->Fill(xstdev);
	       m_xmean_me41->Fill(xmean);  m_xstdev_me41->Fill(xstdev);
	    }
	 } // else itis CSC
      } // end if xmean, xstdev exist
      
      if (m_ny[id] > 0.) {
	 double ymean = m_y1[id] / m_ny[id];
	 double ystdev = sqrt(m_y2[id] / m_ny[id] - (m_y1[id] / m_ny[id]) * (m_y1[id] / m_ny[id]));

	 m_ymean->Fill(ymean);  m_ystdev->Fill(ystdev);
	 if ((*chamber)->geomDetId().subdetId() == MuonSubdetId::DT) {
	    m_ymean_mb->Fill(ymean);  m_ystdev_mb->Fill(ystdev);
	    DTChamberId id((*chamber)->geomDetId().rawId());
	    if (id.station() == 1) {
	       m_ymean_mb1->Fill(ymean);  m_ystdev_mb1->Fill(ystdev);
	    }
	    else if (id.station() == 2) {
	       m_ymean_mb2->Fill(ymean);  m_ystdev_mb2->Fill(ystdev);
	    }
	    else if (id.station() == 3) {
	       m_ymean_mb3->Fill(ymean);  m_ystdev_mb3->Fill(ystdev);
	    }
	    else if (id.station() == 4) {
	       m_ymean_mb4->Fill(ymean);  m_ystdev_mb4->Fill(ystdev);
	    }

	    if (id.wheel() == -2) {
	       m_ymean_minus2->Fill(ymean);  m_ystdev_minus2->Fill(ystdev);
	    }
	    else if (id.wheel() == -1) {
	       m_ymean_minus1->Fill(ymean);  m_ystdev_minus1->Fill(ystdev);
	    }
	    else if (id.wheel() == 0) {
	       m_ymean_zero->Fill(ymean);  m_ystdev_zero->Fill(ystdev);
	    }
	    else if (id.wheel() == 1) {
	       m_ymean_plus1->Fill(ymean);  m_ystdev_plus1->Fill(ystdev);
	    }
	    else if (id.wheel() == 2) {
	       m_ymean_plus2->Fill(ymean);  m_ystdev_plus2->Fill(ystdev);
	    }
	 } // end if DT
	 else {
	    m_ymean_me->Fill(ymean);  m_ystdev_me->Fill(ystdev);

	    CSCDetId id((*chamber)->geomDetId().rawId());

	    if ((id.endcap() == 1? 1: -1)*id.station() == 1  &&  (id.ring() == 1  ||  id.ring() == 4)) {
	       m_ymean_mep11->Fill(ymean);  m_ystdev_mep11->Fill(ystdev);
	       m_ymean_me11->Fill(ymean);  m_ystdev_me11->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  (id.ring() == 1  ||  id.ring() == 4)) {
	       m_ymean_mem11->Fill(ymean);  m_ystdev_mem11->Fill(ystdev);
	       m_ymean_me11->Fill(ymean);  m_ystdev_me11->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 1  &&  id.ring() == 2) {
	       m_ymean_mep12->Fill(ymean);  m_ystdev_mep12->Fill(ystdev);
	       m_ymean_me12->Fill(ymean);  m_ystdev_me12->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  id.ring() == 2) {
	       m_ymean_mem12->Fill(ymean);  m_ystdev_mem12->Fill(ystdev);
	       m_ymean_me12->Fill(ymean);  m_ystdev_me12->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 1  &&  id.ring() == 3) {
	       m_ymean_mep13->Fill(ymean);  m_ystdev_mep13->Fill(ystdev);
	       m_ymean_me13->Fill(ymean);  m_ystdev_me13->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -1  &&  id.ring() == 3) {
	       m_ymean_mem13->Fill(ymean);  m_ystdev_mem13->Fill(ystdev);
	       m_ymean_me13->Fill(ymean);  m_ystdev_me13->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 2  &&  id.ring() == 1) {
	       m_ymean_mep21->Fill(ymean);  m_ystdev_mep21->Fill(ystdev);
	       m_ymean_me21->Fill(ymean);  m_ystdev_me21->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -2  &&  id.ring() == 1) {
	       m_ymean_mem21->Fill(ymean);  m_ystdev_mem21->Fill(ystdev);
	       m_ymean_me21->Fill(ymean);  m_ystdev_me21->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 2  &&  id.ring() == 2) {
	       m_ymean_mep22->Fill(ymean);  m_ystdev_mep22->Fill(ystdev);
	       m_ymean_me22->Fill(ymean);  m_ystdev_me22->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -2  &&  id.ring() == 2) {
	       m_ymean_mem22->Fill(ymean);  m_ystdev_mem22->Fill(ystdev);
	       m_ymean_me22->Fill(ymean);  m_ystdev_me22->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 3  &&  id.ring() == 1) {
	       m_ymean_mep31->Fill(ymean);  m_ystdev_mep31->Fill(ystdev);
	       m_ymean_me31->Fill(ymean);  m_ystdev_me31->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -3  &&  id.ring() == 1) {
	       m_ymean_mem31->Fill(ymean);  m_ystdev_mem31->Fill(ystdev);
	       m_ymean_me31->Fill(ymean);  m_ystdev_me31->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 3  &&  id.ring() == 2) {
	       m_ymean_mep32->Fill(ymean);  m_ystdev_mep32->Fill(ystdev);
	       m_ymean_me32->Fill(ymean);  m_ystdev_me32->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -3  &&  id.ring() == 2) {
	       m_ymean_mem32->Fill(ymean);  m_ystdev_mem32->Fill(ystdev);
	       m_ymean_me32->Fill(ymean);  m_ystdev_me32->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == 4  &&  id.ring() == 1) {
	       m_ymean_mep41->Fill(ymean);  m_ystdev_mep41->Fill(ystdev);
	       m_ymean_me41->Fill(ymean);  m_ystdev_me41->Fill(ystdev);
	    }
	    else if((id.endcap() == 1? 1: -1)*id.station() == -4  &&  id.ring() == 1) {
	       m_ymean_mem41->Fill(ymean);  m_ystdev_mem41->Fill(ystdev);
	       m_ymean_me41->Fill(ymean);  m_ystdev_me41->Fill(ystdev);
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
