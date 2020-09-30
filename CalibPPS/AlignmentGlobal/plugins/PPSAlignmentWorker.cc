/****************************************************************************
 *
 *  CalibPPS/AlignmentGlobal/plugins/PPSAlignmentWorker.cc
 *
 *  Description : PPS Alignment DQM worker
 *
 *  Authors:
 *  - Jan Kašpar
 *  - Mateusz Kocot
 *
 ****************************************************************************/

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfig.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigRcd.h"

#include <map>
#include <string>
#include <cmath>

#include "TH2D.h"
#include "TGraph.h"

//----------------------------------------------------------------------------------------------------

class PPSAlignmentWorker : public DQMEDAnalyzer
{
public:
	PPSAlignmentWorker(const edm::ParameterSet &iConfig);
	~PPSAlignmentWorker() override {};

private:
	void bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, edm::EventSetup const &iSetup) override;
	void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) override;

	// ------------ structures ------------
	struct SectorData
	{
		SectorConfig scfg;

		// hit distributions
		std::map<unsigned int, MonitorElement*> m_h2_y_vs_x_bef_sel;

		std::map<unsigned int, MonitorElement*> m_h2_y_vs_x_mlt_sel;
		
		std::map<unsigned int, MonitorElement*> m_h2_y_vs_x_aft_sel;

		// cut plots
		MonitorElement *h_q_cut_h_bef, *h_q_cut_h_aft;
		MonitorElement *h2_cut_h_bef, *h2_cut_h_aft;

		MonitorElement *h_q_cut_v_bef, *h_q_cut_v_aft;
		MonitorElement *h2_cut_v_bef, *h2_cut_v_aft;

		// near-far plots
		MonitorElement *p_x_diffFN_vs_x_N;
		// MonitorElement *p_y_diffFN_vs_y_N;   // obsolete
		MonitorElement *p_y_diffFN_vs_y_F;

		struct SlicePlots
		{
			MonitorElement *h_y;
			MonitorElement *h2_y_diffFN_vs_y;
			MonitorElement *p_y_diffFN_vs_y;

			SlicePlots();
			SlicePlots(DQMStore::IBooker &iBooker, bool debug);
		};

		std::map<unsigned int, SlicePlots> x_slice_plots_N, x_slice_plots_F;

		void init(DQMStore::IBooker &iBooker, const edm::ESHandle<PPSAlignmentConfig> &cfg, 
		          const SectorConfig &_scfg, const std::string &folder, bool debug);

		unsigned int process(const CTPPSLocalTrackLiteCollection &tracks, 
		                     const edm::ESHandle<PPSAlignmentConfig> &cfg, bool debug);
	};

	// ------------ member data ------------
	edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tracksToken_;

	SectorData sectorData45;
	SectorData sectorData56;

	std::string folder_;
	std::string label_;
	bool debug_;
};

// -------------------------------- SectorData and SlicePlots methods --------------------------------

PPSAlignmentWorker::SectorData::SlicePlots::SlicePlots() {}

PPSAlignmentWorker::SectorData::SlicePlots::SlicePlots(DQMStore::IBooker &iBooker, bool debug)
{
	h_y = iBooker.book1DD("h_y", ";y", 100, -10., 10.);
	auto *tmp = new TProfile("", ";y;x_{F} - y_{N}", 100, -10., 10.);
	p_y_diffFN_vs_y = iBooker.bookProfile("p_y_diffFN_vs_y", tmp);

	if (debug)
		h2_y_diffFN_vs_y = iBooker.book2DD("h2_y_diffFN_vs_y", ";y;x_{F} - y_{N}", 100, -10., 10., 100, -2., 2.);
}

void PPSAlignmentWorker::SectorData::init(DQMStore::IBooker &iBooker, const edm::ESHandle<PPSAlignmentConfig> &cfg, 
                                          const SectorConfig &_scfg, const std::string &folder, bool debug)
{
	scfg = _scfg;

	// binning
	const double bin_size_x = cfg->binning().bin_size_x;
	const unsigned int n_bins_x = cfg->binning().n_bins_x;

	const double pixel_x_offset = cfg->binning().pixel_x_offset;

	const double x_min_pix = pixel_x_offset, x_max_pix = pixel_x_offset + n_bins_x * bin_size_x;
	const double x_min_str = 0., x_max_str = n_bins_x * bin_size_x;

	const unsigned int n_bins_y = cfg->binning().n_bins_y;
	const double y_min = cfg->binning().y_min, y_max = cfg->binning().y_max;

	// hit distributions
	iBooker.setCurrentFolder(folder + "/" + scfg.name + "/before selection/" + scfg.rp_N.name);
	m_h2_y_vs_x_bef_sel[scfg.rp_N.id] = iBooker.book2DD("h2_y_vs_x", ";x;y", n_bins_x, x_min_str, x_max_str, 
	                                                    n_bins_y, y_min, y_max);
	iBooker.setCurrentFolder(folder + "/" + scfg.name + "/before selection/" + scfg.rp_F.name);
	m_h2_y_vs_x_bef_sel[scfg.rp_F.id] = iBooker.book2DD("h2_y_vs_x", ";x;y", n_bins_x, x_min_pix, x_max_pix, 
	                                                    n_bins_y, y_min, y_max);

	iBooker.setCurrentFolder(folder + "/" + scfg.name + "/multiplicity selection/" + scfg.rp_N.name);
	m_h2_y_vs_x_mlt_sel[scfg.rp_N.id] = iBooker.book2DD("h2_y_vs_x", ";x;y", n_bins_x, x_min_str, x_max_str, 
	                                                     n_bins_y, y_min, y_max);
	iBooker.setCurrentFolder(folder + "/" + scfg.name + "/multiplicity selection/" + scfg.rp_F.name);
	m_h2_y_vs_x_mlt_sel[scfg.rp_F.id] = iBooker.book2DD("h2_y_vs_x", ";x;y", n_bins_x, x_min_pix, x_max_pix, 
	                                                     n_bins_y, y_min, y_max);

	iBooker.setCurrentFolder(folder + "/" + scfg.name + "/after selection/" + scfg.rp_N.name);
	m_h2_y_vs_x_aft_sel[scfg.rp_N.id] = iBooker.book2DD("h2_y_vs_x", ";x;y", n_bins_x, x_min_str, x_max_str, 
	                                                     n_bins_y, y_min, y_max);
	iBooker.setCurrentFolder(folder + "/" + scfg.name + "/after selection/" + scfg.rp_F.name);
	m_h2_y_vs_x_aft_sel[scfg.rp_F.id] = iBooker.book2DD("h2_y_vs_x", ";x;y", n_bins_x, x_min_pix, x_max_pix, 
	                                                     n_bins_y, y_min, y_max);

	// cut plots
	iBooker.setCurrentFolder(folder + "/" + scfg.name + "/cuts/cut_h");
	h_q_cut_h_bef = iBooker.book1DD("h_q_cut_h_bef", ";cq_h", 400, -2., 2.);
	h_q_cut_h_aft = iBooker.book1DD("h_q_cut_h_aft", ";cq_h", 400, -2., 2.);
	h2_cut_h_bef = iBooker.book2DD("h2_cut_h_bef", ";x_up;x_dw", n_bins_x, x_min_str, x_max_str, n_bins_x, 
	                               x_min_pix, x_max_pix);
	h2_cut_h_aft = iBooker.book2DD("h2_cut_h_aft", ";x_up;x_dw", n_bins_x, x_min_str, x_max_str, n_bins_x, 
	                               x_min_pix, x_max_pix);

	iBooker.setCurrentFolder(folder + "/" + scfg.name + "/cuts/cut_v");
	h_q_cut_v_bef = iBooker.book1DD("h_q_cut_v_bef", ";cq_v", 400, -2., 2.);
	h_q_cut_v_aft = iBooker.book1DD("h_q_cut_v_aft", ";cq_v", 400, -2., 2.);
	h2_cut_v_bef = iBooker.book2DD("h2_cut_v_bef", ";y_up;y_dw", n_bins_y, y_min, y_max, n_bins_y, y_min, y_max);
	h2_cut_v_aft = iBooker.book2DD("h2_cut_v_aft", ";y_up;y_dw", n_bins_y, y_min, y_max, n_bins_y, y_min, y_max);

	// near-far plots
	iBooker.setCurrentFolder(folder + "/" + scfg.name + "/near_far");
	auto *tmp = new TProfile("", ";x_{N};x_{F} - x_{N}", 100, 0., 20.);
	p_x_diffFN_vs_x_N = iBooker.bookProfile("p_x_diffFN_vs_x_N", tmp);
	// auto tmp_p_y_diffFN_vs_y_N = new TProfile("", ";y_{N};y_{F} - y_{N}", 200, -10., 10.);   // obsolete
	// p_y_diffFN_vs_y_N = iBooker.bookProfile("p_y_diffFN_vs_y_N", tmp_p_y_diffFN_vs_y_N);
	// auto tmp_p_y_diffFN_vs_y_F = new TProfile("", ";y_{F};y_{F} - y_{N}", 200, -10., 10.);
	// p_y_diffFN_vs_y_F = iBooker.bookProfile("p_y_diffFN_vs_y_F", tmp_p_y_diffFN_vs_y_F);

	for (int i = 0; i < scfg.rp_N.x_slice_n; i++)
	{
		const double xMin = scfg.rp_N.x_slice_min + i * scfg.rp_N.x_slice_w;
		const double xMax = scfg.rp_N.x_slice_min + (i + 1) * scfg.rp_N.x_slice_w;

		char buf[100];
		sprintf(buf, "%.1f-%.1f", xMin, xMax);

		iBooker.setCurrentFolder(folder + "/" + scfg.name + "/near_far/x slices, N/" + buf);
		x_slice_plots_N.insert({i, SlicePlots(iBooker, debug)});
	}

	for (int i = 0; i < scfg.rp_F.x_slice_n; i++)
	{
		const double xMin = scfg.rp_F.x_slice_min + i * scfg.rp_F.x_slice_w;
		const double xMax = scfg.rp_F.x_slice_min + (i + 1) * scfg.rp_F.x_slice_w;

		char buf[100];
		sprintf(buf, "%.1f-%.1f", xMin, xMax);

		iBooker.setCurrentFolder(folder + "/" + scfg.name + "/near_far/x slices, F/" + buf);
		x_slice_plots_F.insert({i, SlicePlots(iBooker, debug)});
	}
}

unsigned int PPSAlignmentWorker::SectorData::process(const CTPPSLocalTrackLiteCollection &tracks, 
                                                     const edm::ESHandle<PPSAlignmentConfig> &cfg, bool debug)
{
	CTPPSLocalTrackLiteCollection tracksUp, tracksDw;

	for (const auto &tr : tracks)
	{
		CTPPSDetId rpId(tr.rpId());
		unsigned int rpDecId = rpId.arm()*100 + rpId.station()*10 + rpId.rp();

		if (rpDecId != scfg.rp_N.id && rpDecId != scfg.rp_F.id)
			continue;

		double x = tr.x();
		double y = tr.y();

		// re-build track object
		CTPPSLocalTrackLite trCorr(tr.rpId(), x, 0., y, 0.,
		tr.tx(), tr.txUnc(), tr.ty(), tr.tyUnc(),
		tr.chiSquaredOverNDF(), tr.pixelTrackRecoInfo(), tr.numberOfPointsUsedForFit(),
		tr.time(), tr.timeUnc());

		// store corrected track into the right collection
		if (rpDecId == scfg.rp_N.id)
			tracksUp.push_back(std::move(trCorr));
		if (rpDecId == scfg.rp_F.id)
			tracksDw.push_back(std::move(trCorr));
	}

	// update plots before selection
	for (const auto &tr : tracksUp)
		m_h2_y_vs_x_bef_sel[scfg.rp_N.id]->Fill(tr.x(), tr.y());

	for (const auto &tr : tracksDw)
		m_h2_y_vs_x_bef_sel[scfg.rp_F.id]->Fill(tr.x(), tr.y());

	// skip crowded events
	if (tracksUp.size() > cfg->maxRPTracksSize())
		return 0;

	if (tracksDw.size() > cfg->maxRPTracksSize())
		return 0;

	// update plots with multiplicity selection
	for (const auto &tr : tracksUp)
		m_h2_y_vs_x_mlt_sel[scfg.rp_N.id]->Fill(tr.x(), tr.y());

	for (const auto &tr : tracksDw)
		m_h2_y_vs_x_mlt_sel[scfg.rp_F.id]->Fill(tr.x(), tr.y());

	// do the selection
	unsigned int pairsSelected = 0;

	for (const auto &trUp : tracksUp)
	{
		for (const auto &trDw : tracksDw)
		{
			h2_cut_h_bef->Fill(trUp.x(), trDw.x());
			h2_cut_v_bef->Fill(trUp.y(), trDw.y());

			const double cq_h = trDw.x() + scfg.cut_h_a * trUp.x() + scfg.cut_h_c;
			h_q_cut_h_bef->Fill(cq_h);
			const bool cv_h = (std::fabs(cq_h) < cfg->n_si() * scfg.cut_h_si);

			const double cq_v = trDw.y() + scfg.cut_v_a * trUp.y() + scfg.cut_v_c;
			h_q_cut_v_bef->Fill(cq_v);
			const bool cv_v = (std::fabs(cq_v) < cfg->n_si() * scfg.cut_v_si);

			bool cutsPassed = true;
			if (scfg.cut_h_apply)
				cutsPassed &= cv_h;
			if (scfg.cut_v_apply)
				cutsPassed &= cv_v;

			if (cutsPassed)
			{
				pairsSelected++;

				h_q_cut_h_aft->Fill(cq_h);
				h_q_cut_v_aft->Fill(cq_v);

				h2_cut_h_aft->Fill(trUp.x(), trDw.x());
				h2_cut_v_aft->Fill(trUp.y(), trDw.y());

				m_h2_y_vs_x_aft_sel[scfg.rp_N.id]->Fill(trUp.x(), trUp.y());
				m_h2_y_vs_x_aft_sel[scfg.rp_F.id]->Fill(trDw.x(), trDw.y());

				p_x_diffFN_vs_x_N->Fill(trUp.x(), trDw.x() - trUp.x());

				// const auto &range = cfg->alignment_y_alt_ranges()[scfg.rp_N.id];   // obsolete   
				// if (trUp.x() > range.x_min && trUp.x() < range.x_max)            
				// {                                                                
				// 	p_y_diffFN_vs_y_N->Fill(trUp.y(), trDw.y() - trUp.y());         
				// 	p_y_diffFN_vs_y_F->Fill(trDw.y(), trDw.y() - trUp.y());         
				// }                                                                

				int idx = (trUp.x() - scfg.rp_N.x_slice_min) / scfg.rp_N.x_slice_w;
				if (idx >= 0 && idx < scfg.rp_N.x_slice_n)
				{
					x_slice_plots_N[idx].h_y->Fill(trUp.y());
					x_slice_plots_N[idx].p_y_diffFN_vs_y->Fill(trUp.y(), trDw.y() - trUp.y());
					if (debug)
						x_slice_plots_N[idx].h2_y_diffFN_vs_y->Fill(trUp.y(), trDw.y() - trUp.y());
				}

				idx = (trDw.x() - scfg.rp_F.x_slice_min) / scfg.rp_F.x_slice_w;
				if (idx >= 0 && idx < scfg.rp_F.x_slice_n)
				{
					x_slice_plots_F[idx].h_y->Fill(trDw.y());
					x_slice_plots_F[idx].p_y_diffFN_vs_y->Fill(trDw.y(), trDw.y() - trUp.y());
					if (debug)
						x_slice_plots_F[idx].h2_y_diffFN_vs_y->Fill(trDw.y(), trDw.y() - trUp.y());
				}
			}
		}
	}

	return pairsSelected;
}

// -------------------------------- PPSAlignmentWorker methods --------------------------------

PPSAlignmentWorker::PPSAlignmentWorker(const edm::ParameterSet &iConfig) 
	: tracksToken_(consumes<CTPPSLocalTrackLiteCollection>(iConfig.getParameter<edm::InputTag>("tagTracks"))),
	  folder_(iConfig.getParameter<std::string>("folder")),
	  label_(iConfig.getParameter<std::string>("label")),
	  debug_(iConfig.getParameter<bool>("debug"))
{}

void PPSAlignmentWorker::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, 
                                        edm::EventSetup const &iSetup)
{
	edm::ESHandle<PPSAlignmentConfig> cfg;
	iSetup.get<PPSAlignmentConfigRcd>().get(label_, cfg);

	sectorData45.init(iBooker, cfg, cfg->sectorConfig45(), folder_ + "/worker", debug_);
	sectorData56.init(iBooker, cfg, cfg->sectorConfig56(), folder_ + "/worker", debug_);
}

void PPSAlignmentWorker::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
	const auto &tracks = iEvent.get(tracksToken_);
	
	edm::ESHandle<PPSAlignmentConfig> cfg;
	iSetup.get<PPSAlignmentConfigRcd>().get(label_, cfg);

	sectorData45.process(tracks, cfg, debug_);
	sectorData56.process(tracks, cfg, debug_);
}

DEFINE_FWK_MODULE(PPSAlignmentWorker);