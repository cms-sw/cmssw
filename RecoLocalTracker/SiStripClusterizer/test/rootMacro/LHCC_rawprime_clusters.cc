#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <functional>
#include <cassert>

#include "TFile.h"
#include "TDirectoryFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TGraphErrors.h"
#include "TStopwatch.h"

#include "TCanvas.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TLine.h"
#include "TError.h"

#include "hist_auxiliary.h"
#include "TrackAlgo.h"

#define DEBUG 0

using namespace std;

int W = 800;
int H = 600;
int H_ref = 600; 
int W_ref = 800; 

// references for T, B, L, R
float T = 0.08*H_ref;
float B = 0.12*H_ref; 
float L = 0.12*W_ref;
float R = 0.04*W_ref;
  //
void formatLegend(TLegend* leg, double textsize=27)
{
        leg->SetBorderSize(0);
        leg->SetTextFont(43);
        leg->SetTextSize(textsize);
        leg->SetFillStyle(0);
        leg->SetFillColor(0);
        leg->SetLineColor(0);
}

struct cluster{
	int idx;
	unsigned int event;
	int run;
	int lumi;

	// for approxCluster
	uint32_t    detId;
	uint16_t    firstStrip;
        uint16_t    endStrip;
	float       barycenter;
	uint16_t    size;
	int         charge;

	int      	matchLevel; // 1: exact 
							// 2: having overlap

	cluster(): idx(0), event(0), run(0), lumi(0), detId(0), 
		  firstStrip(0), endStrip(0), barycenter(0.), size(0), charge(0),
		  matchLevel(-1) {};
	cluster(int in_idx,
			unsigned int in_event,
			int in_run,
			int in_lumi,
			// for approxCluster
			uint32_t    in_detId,
			uint16_t    in_firstStrip,
			uint16_t    in_endStrip,
			float       in_barycenter,
			uint16_t    in_size,
			int in_charge) :
			idx(in_idx),
			event(in_event),
			run(in_run),
			lumi(in_lumi),
			  // for approxCluster
			detId(in_detId),
			firstStrip(in_firstStrip),
			endStrip(in_endStrip),
			barycenter(in_barycenter),
			size(in_size),
			charge(in_charge),
			matchLevel(-1) {};

	void print()
	{
		cout << "idx: " << idx <<
				", event: " << event <<
				", run: " << run <<
				", lumi: " << lumi <<

				", detId: " << detId <<
				", firstStrip: " << firstStrip <<
				", endStrip: " << endStrip <<
				", barycenter: " << barycenter <<
				", size: " << size <<
				", charge: " << charge << 
				", matchLevel: " << matchLevel << endl;
	}

};

int main(int argc, char const *argv[])
{
	bool faster = 1;
	std::string expTag = "clusterStudy";
	
	/* *******************************
	 * 0.2 Loading clusters & dead strips
	 * *******************************/
	TFile* f1 = TFile::Open(argv[1],"read");//RawPrimeFlatTupleInt8.root", "read");
        TDirectoryFile* _1      = (TDirectoryFile*) f1->Get("sep19_2_1_dump_rawprime");//dump_rawprime");
        TTree* onlineClusterTree= (TTree*) _1->Get("onlineClusterTree");

	TFile* f2               = TFile::Open(argv[2], "read");
        TDirectoryFile* _2      = (TDirectoryFile*) f2->Get("sep19_2_2_dump_raw");
        TTree* offlineClusterTree= (TTree*) _2->Get("offlineClusterTree");

        TDirectoryFile* _3      = (TDirectoryFile*) f1->Get("sep19_3_dump_deadStrips");
        TTree* onlineDeadStripTree= (TTree*) _3->Get("deadStripTree");

        TDirectoryFile* _4      = (TDirectoryFile*) f2->Get("sep19_3_dump_deadStrips");
        TTree* offlineDeadStripTree= (TTree*) _4->Get("deadStripTree");
	
	const static int nMax = 5000;
	////// for rawprime
	unsigned int rp_event;
	int rp_run;
	int rp_lumi;

	// for approxCluster
	uint32_t    rp_detId;
	uint16_t    rp_firstStrip;
	uint16_t    rp_endStrip;
	float       rp_barycenter;
        UShort_t    rp_falling_barycenter; 
	uint16_t    rp_size;
	int         rp_charge;
        UChar_t        rp_low_pt_trk_cluster;
        UChar_t        rp_high_pt_trk_cluster;
        int         rp_trk_algo;

	float       rp_hitX[nMax];
	float       rp_hitY[nMax];
	uint16_t    rp_channel[nMax];
	uint16_t    rp_adc[nMax];

	// // for reference of approxCluster
	// uint16_t    rp_ref_firstStrip;
	// uint16_t    rp_ref_endStrip;
	// float       rp_ref_barycenter;
	// uint16_t    rp_ref_size;
	// int         rp_ref_charge;

	// float       rp_ref_hitX[nMax];
	// float       rp_ref_hitY[nMax];
	// uint16_t    rp_ref_channel[nMax];
	// uint16_t    rp_ref_adc[nMax];

	// for dead strip
  	unsigned int rp_d_event;
	int rp_d_run;
	int rp_d_lumi;
	int    		rp_d_detId;
	uint16_t    rp_d_size;
	uint16_t    rp_d_channel[800];


	////// for raw
	unsigned int r_event;
	int r_run;
	int r_lumi;

	// for stripCluster
	uint32_t    r_detId;
	uint16_t    r_firstStrip;
	uint16_t    r_endStrip;
	float       r_barycenter;
	uint16_t    r_size;
	int         r_charge;
        UChar_t        r_low_pt_trk_cluster;
        UChar_t        r_high_pt_trk_cluster;
        int         r_trk_algo;

	float       r_hitX[nMax];
	float       r_hitY[nMax];
	uint16_t    r_channel[nMax];
	uint16_t    r_adc[nMax];

	// for dead strip
  	unsigned int r_d_event;
	int r_d_run;
	int r_d_lumi;
	int    		r_d_detId;
	uint16_t    r_d_size;
	uint16_t    r_d_channel[800];

	onlineClusterTree->SetBranchAddress("event", &rp_event);
	onlineClusterTree->SetBranchAddress("run",   &rp_run);
	onlineClusterTree->SetBranchAddress("lumi",  &rp_lumi);

	onlineClusterTree->SetBranchAddress("detId", &rp_detId);
	onlineClusterTree->SetBranchAddress("firstStrip", &rp_firstStrip);
	onlineClusterTree->SetBranchAddress("endStrip", &rp_endStrip);
	onlineClusterTree->SetBranchAddress("barycenter", &rp_barycenter);
        onlineClusterTree->SetBranchAddress("falling_barycenter", &rp_falling_barycenter);
	onlineClusterTree->SetBranchAddress("size", &rp_size);
	onlineClusterTree->SetBranchAddress("charge", &rp_charge);
        onlineClusterTree->SetBranchAddress("low_pt_trk_cluster", &rp_low_pt_trk_cluster);
        onlineClusterTree->SetBranchAddress("high_pt_trk_cluster", &rp_high_pt_trk_cluster);
        onlineClusterTree->SetBranchAddress("trk_algo", &rp_trk_algo);

	onlineClusterTree->SetBranchAddress("x", rp_hitX);
	onlineClusterTree->SetBranchAddress("y", rp_hitY);
	onlineClusterTree->SetBranchAddress("channel", rp_channel);
	onlineClusterTree->SetBranchAddress("adc", rp_adc);

	// onlineClusterTree->SetBranchAddress("ref_firstStrip", &rp_ref_firstStrip);
	// onlineClusterTree->SetBranchAddress("ref_endStrip", &rp_ref_endStrip);
	// onlineClusterTree->SetBranchAddress("ref_barycenter", &rp_ref_barycenter);
	// onlineClusterTree->SetBranchAddress("ref_size", &rp_ref_size);
	// onlineClusterTree->SetBranchAddress("ref_charge", &rp_ref_charge);

	// onlineClusterTree->SetBranchAddress("ref_x", rp_ref_hitX);
	// onlineClusterTree->SetBranchAddress("ref_y", rp_ref_hitY);
	// onlineClusterTree->SetBranchAddress("ref_channel", rp_ref_channel);
	// onlineClusterTree->SetBranchAddress("ref_adc", rp_ref_adc);

	onlineDeadStripTree->SetBranchAddress("event", &rp_d_event);
	onlineDeadStripTree->SetBranchAddress("run",   &rp_d_run);
	onlineDeadStripTree->SetBranchAddress("lumi",  &rp_d_lumi);
	onlineDeadStripTree->SetBranchAddress("detId", &rp_d_detId);
	onlineDeadStripTree->SetBranchAddress("size", &rp_d_size);
	onlineDeadStripTree->SetBranchAddress("channel", rp_d_channel);

	offlineClusterTree->SetBranchAddress("event", &r_event);
	offlineClusterTree->SetBranchAddress("run",   &r_run);
	offlineClusterTree->SetBranchAddress("lumi",  &r_lumi);

	offlineClusterTree->SetBranchAddress("detId", &r_detId);
	offlineClusterTree->SetBranchAddress("firstStrip", &r_firstStrip);
	offlineClusterTree->SetBranchAddress("endStrip", &r_endStrip);
	offlineClusterTree->SetBranchAddress("barycenter", &r_barycenter);
	offlineClusterTree->SetBranchAddress("size", &r_size);
	offlineClusterTree->SetBranchAddress("charge", &r_charge);
        offlineClusterTree->SetBranchAddress("low_pt_trk_cluster", &r_low_pt_trk_cluster);
        offlineClusterTree->SetBranchAddress("high_pt_trk_cluster", &r_high_pt_trk_cluster);
        offlineClusterTree->SetBranchAddress("trk_algo", &r_trk_algo);

	offlineClusterTree->SetBranchAddress("x", r_hitX);
	offlineClusterTree->SetBranchAddress("y", r_hitY);
	offlineClusterTree->SetBranchAddress("channel", r_channel);
	offlineClusterTree->SetBranchAddress("adc", r_adc);

	offlineDeadStripTree->SetBranchAddress("event", &r_d_event);
	offlineDeadStripTree->SetBranchAddress("run",   &r_d_run);
	offlineDeadStripTree->SetBranchAddress("lumi",  &r_d_lumi);
	offlineDeadStripTree->SetBranchAddress("detId", &r_d_detId);
	offlineDeadStripTree->SetBranchAddress("size", &r_d_size);
	offlineDeadStripTree->SetBranchAddress("channel", r_d_channel);

	map< int, map< int, map<int, cluster> > > r_dict; 	// event, detId, idx
	map< int, map< int, map<int, cluster> > > rp_dict; 	// event, detId, idx
	map< int, map< int, map<int, cluster> > > r_d_dict; 	// event, detId, idx
	map< int, map< int, map<int, cluster> > > rp_d_dict; 	// event, detId, idx
	map<int, int> 	matched_sc2ac;
	vector<cluster>  	unmatched_scs;
	vector<cluster>  	unmatched_acs;

	TCanvas *canv0 = new TCanvas("canv0", "canv0", 600*3, 600*1);
	gStyle->SetOptTitle(0);
	gStyle->SetOptStat(0);
	gErrorIgnoreLevel = kWarning;
	canv0->Divide(3,1,0.001,0.001);

	TH1F * h_size_tot_sc      = new TH1F( "RAW_offline_size", "(offline) raw cluster; size; yield",  
	                                    50, 0., 50. );
	TH1F * h_charge_tot_sc     = new TH1F( "RAW_offline_charge", "(offline) raw cluster; charge; yield",  
	                                    88, 0., 704. );
	TH1F * h_barycenter_tot_sc = new TH1F( "RAW_offline_barrycenter", "(offline) raw cluster barycenter; yield",  
	                                    950, 0., 950. );
        TH1F* h_falling_barycenter_tot_ac = new TH1F("falling_barycenter", ";compressed barycenter;yield", 77, 0, 7700.); 

	TH1F * h_size_tot_ac      = new TH1F( "RAW'_online_size", "(online) raw' cluster; size; yield",  
	                                    50, 0., 50. );
	TH1F * h_charge_tot_ac     = new TH1F( "RAW'_online_charge", "(online) raw' cluster; charge; yield",  
	                                    88, 0., 704. );
	TH1F * h_barycenter_tot_ac = new TH1F( "RAW'_online_barycenter", "(online) raw' cluster; barycenter; yield",  
	                                    950, 0., 950. );



        std::map<string, TH1F* >hists;
        gStyle->SetOptStat(1);
        std::vector<int> trk_algo {8,9,10,26,4,11,13,14,23,22,5,24,7,6};
        for(auto& raw: {"raw", "rawp"})
        {
            for(const auto& pt: {"all", "low_pt", "high_pt", "unmatched"})
            {
               for(const auto& var: {"width", "charge", "avg_charge", "detId"})
               {
                   auto key = Form("%s_%s_cluster_%s", raw, pt, var);
                   if(var=="width")
                         hists[key] = new TH1F(key, Form(";%s;yield", var), 40,0,40);
                   else if(var=="detId")
                         hists[key] = new TH1F(key, Form(";%s;yield", var), 8,-0.5,7.5);
                   else
                         hists[key] = new TH1F(key, Form(";cluster_%s;yield", var), 100,0,1000);
               }
            }
            for(const auto& algo: trk_algo)
            {
              auto key = Form("%s_trk_algo_%d_charge", raw, algo);
              hists[key] = new TH1F(key, Form("%s;cluster_charge;yield", algoNames[algo].c_str()), 100,0,1000);
            }
        }
        gStyle->SetOptStat(0);
	const Int_t r_nEntries = offlineClusterTree->GetEntries();
	for (int sc_idx = 0; sc_idx < r_nEntries; ++sc_idx)
	{
		if(sc_idx%1000000 == 0) std::cout << "Scanning raw clusters: " << sc_idx << "/" << r_nEntries << std::endl;
		offlineClusterTree->GetEntry(sc_idx);
                //if(r_event != 8180236 ||  r_run != 382216 || r_lumi !=99) continue;
                //cout << "event " << r_event << "\t" << r_run << "\t" << r_lumi << endl;
                fillWithOverFlow(hists["raw_all_cluster_width"], r_size);
                fillWithOverFlow(hists["raw_all_cluster_charge"], r_charge);
                fillWithOverFlow(hists["raw_all_cluster_avg_charge"], r_charge/r_size);
                fillWithOverFlow(hists["raw_all_cluster_detId"], (r_detId >> 25)&0x7);
                if (std::find(trk_algo.begin(), trk_algo.end(), r_trk_algo) !=trk_algo.end()) {
                  auto key = Form("raw_trk_algo_%d_charge", r_trk_algo);
                   fillWithOverFlow(hists[key], r_charge);
                }
                if(int(r_low_pt_trk_cluster))
                {
                   fillWithOverFlow(hists["raw_low_pt_cluster_width"], r_size);
                   fillWithOverFlow(hists["raw_low_pt_cluster_charge"], r_charge);
                   fillWithOverFlow(hists["raw_low_pt_cluster_avg_charge"], r_charge/r_size);
                   fillWithOverFlow(hists["raw_low_pt_cluster_detId"], (r_detId >>25 )&0x7);
                }
                else if (int(r_high_pt_trk_cluster))
                {
                   fillWithOverFlow(hists["raw_high_pt_cluster_width"], r_size);
                   fillWithOverFlow(hists["raw_high_pt_cluster_charge"], r_charge);
                   fillWithOverFlow(hists["raw_high_pt_cluster_avg_charge"], r_charge/r_size);
                   fillWithOverFlow(hists["raw_high_pt_cluster_detId"], (r_detId >>25 )&0x7);
                }
                else if (int(r_low_pt_trk_cluster) == int(r_high_pt_trk_cluster))
                {
                 fillWithOverFlow(hists["raw_unmatched_cluster_width"], r_size);
                   fillWithOverFlow(hists["raw_unmatched_cluster_charge"], r_charge);
                   fillWithOverFlow(hists["raw_unmatched_cluster_avg_charge"], r_charge/r_size);
                   fillWithOverFlow(hists["raw_unmatched_cluster_detId"], (r_detId >>25 )&0x7); 
                }
		r_dict[ r_event ][ r_detId ][ sc_idx ] = cluster( sc_idx, r_event, r_run, r_lumi,
												r_detId, r_firstStrip, r_endStrip, r_barycenter,
												r_size, r_charge );
	}
	const Int_t rp_nEntries = onlineClusterTree->GetEntries();
	for (int ac_idx = 0; ac_idx < rp_nEntries; ++ac_idx)
	{
		if(ac_idx%1000000 == 0) std::cout << "Scanning rawprime clusters: " << ac_idx << "/" << rp_nEntries << std::endl;
		onlineClusterTree->GetEntry(ac_idx);
                fillWithOverFlow(hists["rawp_all_cluster_width"], rp_size);
                fillWithOverFlow(hists["rawp_all_cluster_charge"], rp_charge);
                fillWithOverFlow(hists["rawp_all_cluster_avg_charge"], rp_charge/rp_size);
                fillWithOverFlow(hists["rawp_all_cluster_detId"], (rp_detId >> 25)&0x7);
                if (std::find(trk_algo.begin(), trk_algo.end(), rp_trk_algo) !=trk_algo.end()) {
                  auto key = Form("rawp_trk_algo_%d_charge", rp_trk_algo);
                   fillWithOverFlow(hists[key], rp_charge);
                }
                if(rp_low_pt_trk_cluster)
                {
                   fillWithOverFlow(hists["rawp_low_pt_cluster_width"], rp_size);
                   fillWithOverFlow(hists["rawp_low_pt_cluster_charge"], rp_charge);
                   fillWithOverFlow(hists["rawp_low_pt_cluster_avg_charge"], rp_charge/rp_size);
                   fillWithOverFlow(hists["rawp_low_pt_cluster_detId"], (rp_detId >>25 )&0x7);
                }
                else if (rp_high_pt_trk_cluster)
                {
                   fillWithOverFlow(hists["rawp_high_pt_cluster_width"], rp_size);
                   fillWithOverFlow(hists["rawp_high_pt_cluster_charge"], rp_charge);
                   fillWithOverFlow(hists["rawp_high_pt_cluster_avg_charge"], rp_charge/rp_size);
                   fillWithOverFlow(hists["rawp_high_pt_cluster_detId"], (rp_detId >> 25)&0x7);
                }
                else if (rp_low_pt_trk_cluster == rp_high_pt_trk_cluster)
                {
                 fillWithOverFlow(hists["rawp_unmatched_cluster_width"], r_size);
                   fillWithOverFlow(hists["rawp_unmatched_cluster_charge"], r_charge);
                   fillWithOverFlow(hists["rawp_unmatched_cluster_avg_charge"], r_charge/r_size);
                   fillWithOverFlow(hists["rawp_unmatched_cluster_detId"], (r_detId >>25 )&0x7);
                }
		if (r_dict.find(rp_event)==r_dict.end()) continue;

		rp_dict[ rp_event ][ rp_detId ][ ac_idx ] = cluster( ac_idx, rp_event, rp_run, rp_lumi,
												rp_detId, rp_firstStrip, rp_endStrip, rp_barycenter,
												rp_size, rp_charge );
		h_size_tot_ac->Fill( rp_size );
		h_charge_tot_ac->Fill( rp_charge );
		h_barycenter_tot_ac->Fill( rp_barycenter );
                h_falling_barycenter_tot_ac->Fill( rp_falling_barycenter );
                if((rp_barycenter*10) > 7680.) std::cout << "found " << std::endl;
	}

	const Int_t r_d_nEntries = offlineDeadStripTree->GetEntries();
	for (int idx = 0; idx < r_d_nEntries; ++idx)
	{
		if(idx%1000000 == 0) std::cout << "Scanning strip (for offline): " << idx << "/" << r_d_nEntries << std::endl;
		offlineDeadStripTree->GetEntry(idx);

		if (r_dict.find(r_d_event)==r_dict.end()) continue;
		if (r_d_size==0) continue;
		r_d_dict[ r_d_event ][ r_d_detId ][ idx ] = cluster( idx, r_d_event, r_d_run, r_d_lumi,
												r_d_detId, -1, -1, -1,
												r_d_size, -1 );
	}
	const Int_t rp_d_nEntries = onlineDeadStripTree->GetEntries();
	for (int idx = 0; idx < rp_d_nEntries; ++idx)
	{
		if(idx%1000000 == 0) std::cout << "Scanning strip (for online HLT): " << idx << "/" << rp_d_nEntries << std::endl;
		onlineDeadStripTree->GetEntry(idx);

		if (r_dict.find(rp_d_event)==r_dict.end()) continue;

		if (rp_d_size==0) continue;
		rp_d_dict[ rp_d_event ][ rp_d_detId ][ idx ] = cluster( idx, rp_d_event, rp_d_run, rp_d_lumi,
												rp_d_detId, -1, -1, -1,
												rp_d_size, -1 );
	}

	for (auto it = r_dict.cbegin(); it != r_dict.cend() /* not hoisted */; /* no increment */)
	{
		if (rp_dict.find(it->first)==rp_dict.end())
		{
			printf("[Error] Unexpected, found unmatched event in raw!\n"); // temp, only for 09/21
			r_dict.erase(it++);    // or "it = m.erase(it)" since C++11
		}
		else
		{
			for (auto& _scs_perEvt_perDetId: it->second) 
			{
				for (auto& _sc: _scs_perEvt_perDetId.second) 
				{
					cluster sc(_sc.second);
					h_size_tot_sc->Fill( sc.size );
					h_charge_tot_sc->Fill( sc.charge );
					h_barycenter_tot_sc->Fill( sc.barycenter);
				}
			}
			++it;
		}
	}
	/* *******************************
	 * 1.0 Plotting total cluster distributions (matched events)
	 * *******************************/
	canv0->cd(1);	
	canv0->GetPad(1)->SetMargin (0.18, 0.05, 0.15, 0.05);
	PlotStyle(h_size_tot_ac); h_size_tot_ac->SetLineColor(46); 	h_size_tot_ac->Draw("");
	PlotStyle(h_size_tot_sc); h_size_tot_sc->SetLineWidth(0); h_size_tot_sc->SetFillColorAlpha(31, 0.4); h_size_tot_sc->SetLineColorAlpha(31, 0.4);  	h_size_tot_sc->Draw("same");
	TLegend* leg0 = canv0->GetPad(1)->BuildLegend(.4, .6, .85, .8);
	formatLegend(leg0);
	canv0->cd(2);	
	canv0->GetPad(2)->SetMargin (0.18, 0.05, 0.15, 0.05);
	PlotStyle(h_charge_tot_ac); h_charge_tot_ac->SetLineColor(46); 	h_charge_tot_ac->Draw("");
	PlotStyle(h_charge_tot_sc); h_charge_tot_sc->SetLineWidth(0); h_charge_tot_sc->SetFillColorAlpha(31, 0.4); h_charge_tot_sc->SetLineColorAlpha(31, 0.4); 	h_charge_tot_sc->Draw("same");
	canv0->cd(3);	
	canv0->GetPad(3)->SetMargin (0.18, 0.05, 0.15, 0.05);
	PlotStyle(h_barycenter_tot_ac); h_barycenter_tot_ac->SetLineColor(46); 	h_barycenter_tot_ac->Draw("");
	PlotStyle(h_barycenter_tot_sc); h_barycenter_tot_sc->SetLineWidth(0); h_barycenter_tot_sc->SetFillColorAlpha(31, 0.4); h_barycenter_tot_sc->SetLineColorAlpha(31, 0.4); 	h_barycenter_tot_sc->Draw("same");
	
	canv0->SaveAs((expTag+"_TotalClusters.png").c_str());

	delete canv0;

	TLatex latex;

	TCanvas *canvSingle0 = new TCanvas("canvSingle0", "canvSingle0", 700, 600);
	
	h_size_tot_ac->GetYaxis()->SetRangeUser(
		h_size_tot_ac->GetYaxis()->GetXmin(),
		h_size_tot_ac->GetMaximum()*1.2 );
	h_size_tot_ac->Draw("");
	h_size_tot_sc->Draw("same");
	h_size_tot_ac->Draw("same");
	leg0 = new TLegend(.62, .6, .87, .8);
	leg0->AddEntry(h_size_tot_sc, "RAW", "f");
	leg0->AddEntry(h_size_tot_ac, "RAW\'", "l");
	// leg0 = canvSingle0->GetPad(0)->BuildLegend(.62, .6, .87, .8);
	formatLegend(leg0);
	leg0->Draw();
	latex.SetTextFont(63);
	latex.SetTextSize(31);
	latex.DrawLatexNDC(0.22,0.84,"CMS");
	latex.SetTextFont(53);
	latex.SetTextSize(22);
	latex.DrawLatexNDC(0.32,0.84,"Preliminary");
	latex.SetTextFont(43);
	latex.SetTextSize(24);
	latex.DrawLatexNDC(0.33,0.945,"2024 PbPb Data #sqrt{s_{NN}} = 5.36 TeV");
	canvSingle0->SaveAs((expTag+"_TotalClusters_size.png").c_str());

	h_charge_tot_ac->GetYaxis()->SetRangeUser(
		h_charge_tot_ac->GetYaxis()->GetXmin(),
		h_charge_tot_ac->GetMaximum()*1.2 );
	h_charge_tot_ac->Draw("");
	h_charge_tot_sc->Draw("same");
	h_charge_tot_ac->Draw("same");
	// leg0 = canvSingle0->GetPad(0)->BuildLegend(.62, .6, .87, .8);
	formatLegend(leg0);
	leg0->Draw();
	latex.DrawLatexNDC(0.33,0.945,"2024 PbPb Data #sqrt{s_{NN}} = 5.36 TeV");
	canvSingle0->SaveAs((expTag+"_TotalClusters_charge.png").c_str());

	h_barycenter_tot_ac->Draw("");
	h_barycenter_tot_sc->Draw("same");
	h_barycenter_tot_ac->Draw("same");
	// leg0 = canvSingle0->GetPad(0)->BuildLegend(.62, .6, .87, .8);
	formatLegend(leg0);
	leg0->Draw();
	latex.DrawLatexNDC(0.33,0.945,"2024 PbPb Data #sqrt{s_{NN}} = 5.36 TeV");
	canvSingle0->SaveAs((expTag+"_TotalClusters_barycenter.png").c_str());
	
	delete canvSingle0;
	/*
	cout << "map of raw clusters: " << endl;
	for(auto& _cs_perEvt: r_dict) 
	{
		cout << "event: " << _cs_perEvt.first << endl;
		for (auto& _cs_perEvt_perDetId: _cs_perEvt.second) 
		{
			cout << "detId: " << _cs_perEvt_perDetId.first << endl;
			for (auto& _c: _cs_perEvt_perDetId.second) 
			{
				cout << "idx: " << _c.first << endl;
				_c.second.print();
			}
		}
	}
	cout << "map of rawprime clusters: " << endl;
	for(auto& _cs_perEvt: rp_dict) 
	{
		cout << "event: " << _cs_perEvt.first << endl;
		for (auto& _cs_perEvt_perDetId: _cs_perEvt.second) 
		{
			cout << "detId: " << _cs_perEvt_perDetId.first << endl;
			for (auto& _c: _cs_perEvt_perDetId.second) 
			{
				cout << "idx: " << _c.first << endl;
				_c.second.print();
			}
		}
	}
	*/

	/* *******************************
	 * 2. cluster matching
	 * *******************************/
	for(auto& _acs_perEvt: rp_dict) 
	{
		unsigned int this_event = _acs_perEvt.first;
		for (auto& _acs_perEvt_perDetId: _acs_perEvt.second) 
		{
			int this_detId = _acs_perEvt_perDetId.first;
			for (auto& _ac: _acs_perEvt_perDetId.second) 
			{
				// int this_idx = _ac.first;

				cluster ac(_ac.second);
				float distance(9999.);
				cluster closet_sc; closet_sc.idx = -999;

				for (auto& _sc: r_dict[this_event][this_detId]) 
				{
					cluster sc(_sc.second);
					// cluster matching: exact
					if (ac.size == sc.size &&
						std::max( ac.firstStrip, sc.firstStrip ) <= \
						std::min( ac.endStrip,   sc.endStrip ) ) 
					{
						if (std::abs(ac.barycenter - sc.barycenter) < distance) 
						{
							closet_sc 	= sc;
							distance 	= std::abs(ac.barycenter - sc.barycenter);
						}
					} // end exact matching scan
				} // end strip cluster loop

				if (closet_sc.idx!=-999) 
				{
					if ( matched_sc2ac.count( closet_sc.idx ) == 0 )
					{
						if (ac.firstStrip == closet_sc.firstStrip) {
							closet_sc.matchLevel = 1;
							ac.matchLevel = 1;
						} else {
							closet_sc.matchLevel = 2;
							ac.matchLevel = 2;
						}
						matched_sc2ac[ closet_sc.idx ] = ac.idx;
					} 
					else 
					{
						cluster prev_ac = rp_dict[this_event][this_detId] \
												 [	matched_sc2ac[ closet_sc.idx ] ]; // extract previous ac that matches to sc, get its idx

						if (std::abs( ac.barycenter - closet_sc.barycenter ) < \
							std::abs( prev_ac.barycenter - closet_sc.barycenter ) ) {
							unmatched_acs.push_back(prev_ac);
							matched_sc2ac[ closet_sc.idx ] = ac.idx;
						}
                                                else {
                                                   unmatched_acs.push_back(ac);
                                                }
					}
				}
				else 
				{
					unmatched_acs.push_back(ac);
				} 
			}
		}
	}
        assert((unmatched_acs.size()+matched_sc2ac.size()) == rp_nEntries);
	for(auto& _scs_perEvt: r_dict) 
	{
		for (auto& _scs_perEvt_perDetId: _scs_perEvt.second) 
		{
			for (auto& _sc: _scs_perEvt_perDetId.second) 
			{
				cluster sc(_sc.second);
				if ( matched_sc2ac.count(sc.idx)==0 ) unmatched_scs.push_back(sc);
			}
		}
	}

	printf("[Summary] matched_sc2ac.size(): %ld, unmatched_acs.size(): %ld, unmatched_scs.size(): %ld\n", 
	                matched_sc2ac.size(), unmatched_acs.size(), unmatched_scs.size());
        
        cout << setprecision(2);
        cout << "total cluster in raw: " << r_nEntries << endl;
        cout << "total cluster in rawp: " << rp_nEntries << endl;  
        cout << "not matched cluster in raw " << (100.*unmatched_scs.size() / r_nEntries) << "%" << endl;
        cout << "not matched cluster in rawp " << (100.*unmatched_acs.size() / rp_nEntries) << "%" << endl;;
	/* *******************************
	 * 3.1 plotting matched cluster pairs
	 * *******************************/
	TH2F * h_size      = new TH2F( "size", 
	                                    "; RAW SiStripCluster size; RAW' ApproxCluster size",  
	                                    50, 0., 50.,
	                                    50, 0., 50. );
	TH2F * h_charge     = new TH2F( "charge", 
	                                    "; RAW SiStripCluster charge; RAW' ApproxCluster charge",  
	                                    88, 0., 704.,
	                                    88, 0., 704. );
	TH2F * h_barycenter = new TH2F( "barycenter", 
	                                    "; RAW SiStripCluster barycenter; RAW' ApproxCluster barycenter",  
	                                    192, 0., 768.,
	                                    192, 0., 768. );
	TH2F * h_barycenter_vs_charge = new TH2F( "barycenter_vs_charge", 
	                                    "; #Delta barycenter (RAW'-RAW); #Delta charge (RAW'-RAW)",  
	                                    100, -2., 2.,
	                                    100, -700, 300 );

	TH1F * h_size_res      = new TH1F( "size_res", 
	                                    "; size (RAW'-RAW)/RAW; yield",
	                                    50, -.1, .1);
	TH1F * h_charge_res     = new TH1F( "chagre_res", 
	                                    "; charge (RAW'-RAW)/RAW; yield",
	                                    50, -.1, .1);
	TH1F * h_barycenter_res = new TH1F( "barycenter_res", 
	                                    "; barycenter (RAW'-RAW)/RAW; yield",
	                                    50, -.1, .1);

	ofstream matched_sc2ac_txt;
	matched_sc2ac_txt.open(Form("log/%s_matched_sc2ac.txt", expTag.c_str()));
	matched_sc2ac_txt << "event detId sc_idx barycenter size charge firstStrip endStrip ac_idx barycenter size charge firstStrip endStrip\n";
        TFile * f = new TFile("cluster_study.root", "recreate");
        f->cd();
        for(auto [key, hist]: hists)
        {
           hist->Write();
           delete hist;
        }
	for (auto& idx_pair: matched_sc2ac) 
	{
		// printf("[Debug] approxCluster %p, SiStripCluster %p\n", ac_ptr, sc_ptr);
		offlineClusterTree->GetEntry(idx_pair.first);
		onlineClusterTree->GetEntry( idx_pair.second);


		h_size     ->Fill( r_size, rp_size );
		h_charge    ->Fill( r_charge, rp_charge );
		h_barycenter->Fill( r_barycenter, rp_barycenter ); 
		h_barycenter_vs_charge->Fill( 	rp_barycenter - r_barycenter,
						rp_charge - r_charge ); 

		h_size_res     ->Fill( ( rp_size - r_size )/((float) r_size) );
		int charge_withovrflow = fillWithOverFlow(h_charge_res, ( rp_charge - r_charge )/((float) r_charge), 1 );
		int bary = fillWithOverFlow(h_barycenter_res, (rp_barycenter - r_barycenter)/r_barycenter,1);
		matched_sc2ac_txt << r_event << " " << r_detId << " " 
						  << idx_pair.first << " " << r_barycenter << " " << r_size << " " << r_charge << " " << r_firstStrip << " " << r_endStrip << " "
						  << idx_pair.second << " " << rp_barycenter << " " << rp_size << " " << rp_charge << " " << rp_firstStrip << " " << rp_endStrip << "\n";
		if (charge_withovrflow && DEBUG) {
		        map<int, cluster> r_map = r_dict[r_event][r_detId];
			map<int, cluster> rp_map = rp_dict[rp_event][rp_detId];
			if ((r_event != rp_event) | (r_detId != rp_detId)) return 0;
			TCanvas *canvSingle = new TCanvas("canv", "canvSingle", 700, 600);
			canvSingle->cd();
                        gStyle->SetOptTitle(0);
			//if (idx_pair.first != 97471) continue;
                        //if (idx_pair.first != 3654862) continue;
			//if ( std::abs( (int)r_map.size() - (int)rp_map.size()) >=1) std::cout << r_run << "\t" << r_lumi << "\t" << r_event << "\t" << r_map.size() << "\t" << rp_map.size() << "\t" << r_detId << "\t" << rp_detId << std::endl;
			//return 0;

			for (auto& sc: r_map) {
                           offlineClusterTree->GetEntry(sc.first);
			   bool matched = false;
			   TH1F * h_adc_matched(NULL); 
			   TH1F * h_adc_unmatched(NULL);
                           if (sc.first == idx_pair.first) {
				   matched = true;
				   h_adc_matched = new TH1F(Form("Raw_matched_cluster_%d-%d", idx_pair.first, idx_pair.second), Form("RAW_matched_cluster (%d-%d); strip; ADC", r_firstStrip, r_endStrip), 800,0.,800);
			   }
			   else {
				  h_adc_unmatched = new TH1F(Form("Raw_other_cluster_%d-%d", idx_pair.first, sc.first), Form("RAW_other_cluster (%d-%d); strip; ADC", r_firstStrip, r_endStrip), 800,0.,800);
                           }
			   for (uint16_t i=0; i < r_size; ++i) {
                                if (matched) h_adc_matched->Fill(r_channel[i], r_adc[i]);
				else h_adc_unmatched->Fill(r_channel[i], r_adc[i]);
                           }
			   if  (matched) {
			     h_adc_matched->SetLineWidth(0);
                             h_adc_matched->SetFillColorAlpha(31, 0.4);
                             h_adc_matched->SetLineColorAlpha(31, 0.4);
                             h_adc_matched->GetYaxis()->SetRangeUser(0, h_adc_matched->GetMaximum()*4);
                             h_adc_matched->DrawClone("hist same");
			     h_adc_matched->Write();
			     delete h_adc_matched;
			   }
                           else {
			     h_adc_unmatched->SetLineColor(46);
			     h_adc_unmatched->DrawClone("hist same");
			     h_adc_unmatched->Write();
			     delete h_adc_unmatched;
			   }
			}
			for (auto& c: rp_map) {
                           onlineClusterTree->GetEntry(c.first);
                           bool matched = false;
                           TH1F * h_adc_matched(NULL);
                           TH1F * h_adc_unmatched(NULL);
                           if (c.first == idx_pair.second) {
                                   matched = true;
                                   h_adc_matched = new TH1F(Form("Raw'_matched_cluster_%d-%d", idx_pair.first, idx_pair.second), Form("RAW'_matched_cluster (%d-%d); strip; ADC", rp_firstStrip, rp_endStrip), 800,0.,800);
                           }
                           else {
                                  h_adc_unmatched = new TH1F(Form("Raw'_other_cluster_%d-%d", idx_pair.second, c.first), Form("RAW'_other_cluster (%d-%d); strip; ADC", rp_firstStrip, rp_endStrip), 800,0.,800);
                           }
                           for (uint16_t i=0; i < rp_size; ++i) {
                                if (matched) h_adc_matched->Fill(rp_channel[i], rp_adc[i]);
                                else h_adc_unmatched->Fill(rp_channel[i], rp_adc[i]);
                           }
                           if  (matched) {
                             h_adc_matched->SetLineWidth(0);
                             h_adc_matched->SetFillColorAlpha(8, 0.4);
                             h_adc_matched->SetLineColorAlpha(8, 0.4);
                             h_adc_matched->GetYaxis()->SetRangeUser(0, h_adc_matched->GetMaximum()*3);
                             h_adc_matched->DrawClone("same hist");
			     h_adc_matched->Write();
                             delete h_adc_matched;
                           }
                           else {
                             h_adc_unmatched->SetLineColor(9);
                             h_adc_unmatched->DrawClone("hist same");
			     h_adc_unmatched->Write();
                             delete h_adc_unmatched;
                           }
                        }
			TLegend* leg = canvSingle->BuildLegend(.5, .6, .85, .9);
                        gPad->Modified();
                        gPad->Update();
                        canvSingle->SaveAs(Form("%sRawp_overflow_matched_vs_unmatched_idx_%d.png",expTag.c_str(), idx_pair.first));
			delete canvSingle;
		}
	}

	matched_sc2ac_txt.close();

        if ( DEBUG ) {

	   map<int, cluster> r_map = r_dict[8076454][369120301];
	   map<int, cluster> rp_map = rp_dict[8076454][369120301];
	   std::cout << "offline" << std::endl;

	   for (auto& _sc: r_map) {
             offlineClusterTree->GetEntry(_sc.first);
	     std::cout << r_firstStrip << "\t" << r_endStrip << std::endl;
	   }

	  std::cout << "online" << std::endl;

          for (auto& _ac: rp_map) {
            onlineClusterTree->GetEntry(_ac.first);
            std::cout << rp_firstStrip << "\t" << rp_endStrip << std::endl;
          }

	  int count(0);
	  int not_present(0);
          for(auto& sc: unmatched_scs)
          {
	     if (rp_dict.find(sc.event) == rp_dict.end()){
		 ++not_present;      
		 continue;
	     }
             if (count >10) break;
               count +=1;
               map<int, cluster> r_map = r_dict[sc.event][sc.detId];
               map<int, cluster> rp_map = rp_dict[sc.event][sc.detId];
               std::cout << r_map.size() << "\t" << rp_map.size() << std::endl;
	       std::cout << sc.event << "\t" << sc.detId << std::endl;
	       TCanvas *canvSingle = new TCanvas("canv", "canvSingle", 700, 600);

               for (auto& _sc: r_map) {
                  offlineClusterTree->GetEntry(_sc.first);
                  TH1F* h_adc = new TH1F(Form("Raw_cluster_%d-%d", sc.idx, _sc.first), Form("RAW_cluster (%d-%d); strip; ADC", r_firstStrip, r_endStrip), 800,0.,800);
	          if (_sc.second.idx == sc.idx) h_adc->SetTitle(Form("RAW_cluster_unmatched (%d-%d); strip; ADC", r_firstStrip, r_endStrip));
                  for (uint16_t i=0; i < r_size; ++i) h_adc->Fill(r_channel[i], r_adc[i]);
                  h_adc->SetLineColor(46);
                  h_adc->DrawClone("hist same");
                  h_adc->Write();
                  delete h_adc;
              }

              for (auto& _ac: rp_map) {
                 onlineClusterTree->GetEntry(_ac.first);
                 TH1F* h_adc = new TH1F(Form("Rawp_cluster_%d-%d", sc.idx, _ac.first), Form("RAWp_cluster (%d-%d); strip; ADC", rp_firstStrip, rp_endStrip), 800,0.,800);
                 for (uint16_t i=0; i < rp_size; ++i) h_adc->Fill(rp_channel[i], rp_adc[i]);
                 h_adc->SetLineColor(9);
                 h_adc->DrawClone("hist same");
                 h_adc->Write();
                 delete h_adc;
              }
              TLegend* leg = canvSingle->BuildLegend(.5, .6, .85, .9);
              gPad->Modified();
              gPad->Update();
              canvSingle->SaveAs(Form("%sunmatched_cluster_%d.png",expTag.c_str(), sc.idx));
              delete canvSingle;
	      count++;

          }
	std::cout << "not present " << not_present << std::endl;
	}

	f->Close();
	TCanvas *canv = new TCanvas("canv", "canv", 700*4, 600*2);


	canv->Divide(4,2,0.001,0.001);

	canv->cd(1); canv->GetPad(1)->SetMargin (0.18, 0.20, 0.12, 0.07);
	PlotStyle(h_size);  	h_size->Draw("COLZ");
	canv->cd(2); canv->GetPad(2)->SetMargin (0.18, 0.20, 0.12, 0.07);
	PlotStyle(h_charge);  	h_charge->Draw("COLZ");
	canv->cd(3); canv->GetPad(3)->SetMargin (0.18, 0.20, 0.12, 0.07);
	PlotStyle(h_barycenter);  	h_barycenter->Draw("COLZ");
	canv->cd(4); canv->GetPad(4)->SetMargin (0.18, 0.20, 0.12, 0.07);
	PlotStyle(h_barycenter_vs_charge);  	h_barycenter_vs_charge->Draw("COLZ");
	
	canv->cd(5); canv->GetPad(5)->SetLogy(); canv->GetPad(5)->SetMargin (0.18, 0.20, 0.12, 0.07);
	PlotStyle(h_size_res);  		h_size_res->Draw("");
	canv->cd(6); canv->GetPad(6)->SetLogy(); canv->GetPad(6)->SetMargin (0.18, 0.20, 0.12, 0.07);
	PlotStyle(h_charge_res);  		h_charge_res->Draw("");
	canv->cd(7); canv->GetPad(7)->SetLogy(); canv->GetPad(7)->SetMargin (0.18, 0.20, 0.12, 0.07);
	PlotStyle(h_barycenter_res);  	h_barycenter_res->Draw("");


	canv->SaveAs((expTag+"_MatchedClusters.png").c_str());

	delete canv;


	TCanvas *canvSingle = new TCanvas("canvSingle", "canvSingle", 700, 600);
	gStyle->SetOptTitle(0);
	gErrorIgnoreLevel = kWarning;
	canvSingle->GetPad(0)->SetMargin (0.18, 0.20, 0.12, 0.07);
	canvSingle->cd();
	
	h_size->GetZaxis()->SetTitleOffset(1.8);
	h_size->GetZaxis()->SetTitle("number of clusters");
	h_size->Draw("COLZ");
	latex.DrawLatexNDC(0.33,0.945,"2024 PbPb Data #sqrt{s_{NN}} = 5.36 TeV");
	canvSingle->SaveAs((expTag+"_MatchedClusters_size_scat.png").c_str());

	h_charge->GetZaxis()->SetTitleOffset(1.8);
	h_charge->GetZaxis()->SetTitle("number of clusters");
	h_charge->Draw("COLZ");
	latex.DrawLatexNDC(0.33,0.945,"2024 PbPb Data #sqrt{s_{NN}} = 5.36 TeV");
	canvSingle->SaveAs((expTag+"_MatchedClusters_charge_scat.png").c_str());

	h_barycenter->GetZaxis()->SetTitleOffset(1.8);
	h_barycenter->GetZaxis()->SetTitle("number of clusters");
	h_barycenter->Draw("COLZ");
	latex.DrawLatexNDC(0.33,0.945,"2024 PbPb Data #sqrt{s_{NN}} = 5.36 TeV");
	canvSingle->SaveAs((expTag+"_MatchedClusters_barycenter_scat.png").c_str());

	h_barycenter_vs_charge->Draw("COLZ");
	latex.DrawLatexNDC(0.33,0.945,"2024 PbPb Data #sqrt{s_{NN}} = 5.36 TeV");
	canvSingle->SaveAs((expTag+"_MatchedClusters_del_barycenter_del_charge_scat.png").c_str());

	h_size_res->Draw("");
	latex.DrawLatexNDC(0.33,0.945,"2024 PbPb Data #sqrt{s_{NN}} = 5.36 TeV");
	
	latex.SetTextFont(43);
	latex.DrawLatexNDC(0.60,0.80,Form("Mean=%.2f", h_size_res->GetMean()));
	latex.DrawLatexNDC(0.60,0.75,Form("Std Dev=%.2f", h_size_res->GetStdDev()));
	canvSingle->SaveAs((expTag+"_MatchedClusters_size_res.png").c_str());

        h_charge_res->Scale(1/h_charge_res->Integral());
	h_charge_res->Draw("");
	latex.DrawLatexNDC(0.21,0.84,"CMS");
	latex.DrawLatexNDC(0.21,0.80,"Preliminary");
	latex.DrawLatexNDC(0.33,0.945,"2024 PbPb Data #sqrt{s_{NN}} = 5.36 TeV");
	latex.SetTextFont(43);
	latex.DrawLatexNDC(0.60,0.80,Form("Mean=%.2f", h_charge_res->GetMean()));
	latex.DrawLatexNDC(0.60,0.75,Form("Std Dev=%.2f", h_charge_res->GetStdDev()));
	canvSingle->SetLogy(true);
	canvSingle->SaveAs((expTag+"_MatchedClusters_charge_res.png").c_str());

        h_barycenter_res->Scale(1/h_barycenter_res->Integral());
	h_barycenter_res->Draw("");
	latex.DrawLatexNDC(0.21,0.84,"CMS");
        latex.DrawLatexNDC(0.21,0.80,"Preliminary");
	latex.DrawLatexNDC(0.33,0.945,"2024 PbPb Data #sqrt{s_{NN}} = 5.36 TeV");
	latex.SetTextFont(43);
	latex.DrawLatexNDC(0.60,0.80,Form("Mean=%.2f", h_barycenter_res->GetMean()));
	latex.DrawLatexNDC(0.60,0.75,Form("Std Dev=%.2f", h_barycenter_res->GetStdDev()));
	canvSingle->SaveAs((expTag+"_MatchedClusters_barycenter_res.png").c_str());

        PlotStyle(h_falling_barycenter_tot_ac);
        h_falling_barycenter_tot_ac->GetXaxis()->SetNdivisions(606);
        h_falling_barycenter_tot_ac->Scale(1/h_falling_barycenter_tot_ac->Integral());
        canvSingle->SetLogy(true);
        h_falling_barycenter_tot_ac->Draw("HIST");
        latex.DrawLatexNDC(0.31,0.84,"CMS Preliminary");
        latex.DrawLatexNDC(0.33,0.945,"2024 PbPb Data #sqrt{s_{NN}} = 5.36 TeV");
        latex.SetTextFont(43);
        canvSingle->SaveAs((expTag+"falling_barycenter.png").c_str());
        delete h_falling_barycenter_tot_ac;

	ofstream unmatched_scs_txt;
	unmatched_scs_txt.open(Form("log/%s_unmatched_scs.txt",expTag.c_str()));
	unmatched_scs_txt << "event detId sc_idx barycenter size charge firstStrip endStrip\n";
	for(auto& sc: unmatched_scs)
	{
		unmatched_scs_txt << sc.event << " " << sc.detId << " " 
						  << sc.idx << " " << sc.barycenter << " " << sc.size << " " << sc.charge << " "
						  << sc.firstStrip << " " << sc.endStrip << "\n";
	}
	unmatched_scs_txt.close();


	ofstream unmatched_acs_txt;
	unmatched_acs_txt.open(Form("log/%s_unmatched_acs.txt",expTag.c_str()));
	unmatched_acs_txt << "event detId ac_idx barycenter size charge firstStrip endStrip\n";
	for(auto& ac: unmatched_acs)
	{
		unmatched_acs_txt << ac.event << " " << ac.detId << " " 
						  << ac.idx << " " << ac.barycenter << " " << ac.size << " " << ac.charge << " "
						  << ac.firstStrip << " " << ac.endStrip << "\n";
	}
	unmatched_acs_txt.close();

	/* *******************************
	 * 3.2 Plotting unmatched raw SiStripCluster
	 * *******************************/

	if (DEBUG)
	{

	   for(auto& sc: unmatched_scs)
	   {
		offlineClusterTree->GetEntry(sc.idx);
		if (!faster) sc.print();
		// find ac
		if( rp_dict.find( sc.event ) == rp_dict.end() ) continue;
		if( rp_dict[ sc.event ].find( sc.detId ) == rp_dict[ sc.event ].end() ) {
			printf("[Warning:1] no matched ac detId %d in event %d for sc\n", sc.detId, sc.event);
			cout << "sc: " << endl;
			sc.print();
			cout << "offline dead strip (sc) " << endl;
			for (auto& r_ds: r_d_dict[ sc.event ][ sc.detId ]) r_ds.second.print();
			cout << "online dead strip (sc) " << endl;
			for (auto& rp_ds: rp_d_dict[ sc.event ][ sc.detId ]) rp_ds.second.print();
			continue;
		}

		TCanvas *canv2 = new TCanvas("canv2", "canv2", 600, 600);
		gStyle->SetOptStat(0);

		TH1F * h_sc = new TH1F( Form("sc%d", sc.idx), Form("RAW (%d-%d); strip; ADC",sc.firstStrip,sc.endStrip), 800,0,800 );

		for(uint16_t i=0; i < r_size; ++i) h_sc->Fill(r_channel[i], r_adc[i]);
		// PlotStyle(h_sc);  	
		h_sc->SetLineWidth(0);
		h_sc->SetFillColorAlpha(31, 0.4);
		h_sc->SetLineColorAlpha(31, 0.4);
		h_sc->GetYaxis()->SetRangeUser(0, h_sc->GetMaximum()*3);
		h_sc->DrawClone("hist");

		/* *******************************
		 * 3.2.1 plot ac on the same detId
		 * *******************************/
		map<int, cluster> ac_map = rp_dict[sc.event][sc.detId];
		if (!faster) printf("found %ld ac in same event and same detId\n", ac_map.size());

		bool foundOverlap(false);
		// bool foundOverlap_ref(false);
		for (auto& ac: ac_map) 
		{

			onlineClusterTree->GetEntry(ac.first);
			if (!faster) ac.second.print();

			if (!foundOverlap) {
				foundOverlap =  std::max( ac.second.firstStrip, sc.firstStrip ) <= \
				                std::min( ac.second.endStrip,   sc.endStrip );
			}

			// if (!foundOverlap_ref) {
			// 	foundOverlap_ref =  std::max( rp_ref_firstStrip, sc.firstStrip ) <= \
			// 	                std::min( rp_ref_endStrip,   sc.endStrip );
			// }

			TH1F * h_ac = new TH1F(Form("ac_%d_%d", sc.idx, ac.first), Form("RAW' (%d-%d); strip; ADC",rp_firstStrip,rp_endStrip), 800,0,800 );
			for(uint16_t i=0; i < rp_size; ++i) h_ac->Fill(rp_channel[i], rp_adc[i]);
			// PlotStyle(h_ac);
			h_ac->SetLineColor(46);

			// TH1F * h_ac_ref = new TH1F(Form("ac_ref_%d_%d", sc.idx, ac.first), Form("RAW' ref (%d); strip; ADC",ac.first), 800,0,800 );
			// for(uint16_t i=0; i < rp_ref_size; ++i) h_ac_ref->Fill(rp_ref_channel[i], rp_ref_adc[i]);
			// // PlotStyle(h_ac_ref);
			// h_ac_ref->SetLineColor(kRed);

			h_ac->DrawClone("hist same");
			// h_ac_ref->DrawClone("hist same");
			
			delete h_ac; 
			// delete h_ac_ref; 
		}
		
		string prefix("");
		if (!foundOverlap)
		{
			printf("[Warning:2] no matched ac in event %d detId %d for sc\n", sc.event, sc.detId);
			prefix = "Warning2_";
			cout << "sc: " << endl;
			sc.print();
			cout << "ac: " << endl;
			for (auto& ac: ac_map) {
				cout << "approx: "; ac.second.print();
				// onlineClusterTree->GetEntry(ac.first);
				// cluster ac_ref = cluster( ac.first, rp_event, rp_run, rp_lumi,
				// 						  rp_detId, rp_ref_firstStrip, rp_ref_endStrip, rp_ref_barycenter,
				// 						  rp_ref_size, rp_ref_charge );
				// cout << "ref    : "; ac_ref.print();
			}
		}

		// if (!foundOverlap_ref)
		// {
		// 	printf("[Warning:3] no matched ac ref in event %d detId %d for sc\n", sc.event, sc.detId);
		// 	prefix = "Warning3_";
		// 	cout << "sc: " << endl;
		// 	sc.print();
		// 	cout << "ac: " << endl;
		// 	for (auto& ac: ac_map) {
		// 		cout << "approx: "; ac.second.print();
		// 		onlineClusterTree->GetEntry(ac.first);
		// 		cluster ac_ref = cluster( ac.first, rp_event, rp_run, rp_lumi,
		// 								  rp_detId, rp_ref_firstStrip, rp_ref_endStrip, rp_ref_barycenter,
		// 								  rp_ref_size, rp_ref_charge );
		// 		cout << "ref    : "; ac_ref.print();
		// 	}
		// }

		/* *******************************
		 * 3.2.2 plot ds of raw on the same detId
		 * *******************************/
		map<int, cluster> r_ds_map = r_d_dict[sc.event][sc.detId];
		if (!faster) printf("found %ld online (raw') ds in same event and same detId\n", r_ds_map.size());

		if (r_ds_map.size()==1)
		{
			cluster& r_ds = r_ds_map.begin()->second;
			offlineDeadStripTree->GetEntry(r_ds.idx);

			TH1F * h_r_ds = new TH1F(Form("r_ds_%d_%d", sc.idx, r_ds.idx), Form("RAW dead (%d-%d); strip; ADC",r_ds.firstStrip, r_ds.endStrip), 800,0,800 );
			int r_d_channel_min = 1000;
			int r_d_channel_max = -1;
			for(uint16_t i=0; i < r_d_size; ++i) 
			{
				h_r_ds->Fill(r_d_channel[i], h_sc->GetMaximum());
				if (r_d_channel[i] < r_d_channel_min) r_d_channel_min = r_d_channel[i];
				if (r_d_channel[i] > r_d_channel_max) r_d_channel_max = r_d_channel[i];
			}
			h_r_ds->SetTitle(Form("RAW dead (%d-%d);",r_d_channel_min,r_d_channel_max));
			// PlotStyle(h_r_ds);
			h_r_ds->SetFillColorAlpha(kYellow+1, 0.2);
			h_r_ds->SetLineColorAlpha(kYellow+1, 0.2);
			h_r_ds->DrawClone("hist same");
			delete h_r_ds; 
		} else if (r_ds_map.size()==0) {}
		else {
			printf("[Error] Don't expect\n");
			exit(0);
		}


		/* *******************************
		 * 3.2.3 plot ds of rawprime on the same detId
		 * *******************************/
		map<int, cluster> rp_ds_map = rp_d_dict[sc.event][sc.detId];
		// if (!faster) printf("found %ld online (raw') ds in same event and same detId\n", rp_ds_map.size());

		if (rp_ds_map.size()==1)
		{
			cluster& rp_ds = rp_ds_map.begin()->second;
			onlineDeadStripTree->GetEntry(rp_ds.idx);

			TH1F * h_rp_ds = new TH1F(Form("rp_ds_%d_%d", sc.idx, rp_ds.idx), Form("RAW' dead (%d); strip; ADC",rp_ds.idx), 800,0,800 );
			int rp_d_channel_min = 1000;
			int rp_d_channel_max = -1;
			for(uint16_t i=0; i < rp_d_size; ++i) 
			{
				h_rp_ds->Fill(rp_d_channel[i], h_sc->GetMaximum());
				if (rp_d_channel[i] < rp_d_channel_min) rp_d_channel_min = rp_d_channel[i];
				if (rp_d_channel[i] > rp_d_channel_max) rp_d_channel_max = rp_d_channel[i];

			}
			h_rp_ds->SetTitle(Form("RAW' dead (%d-%d);",rp_d_channel_min,rp_d_channel_max));
			// PlotStyle(h_rp_ds);
			h_rp_ds->SetFillColorAlpha(kMagenta, 0.3);
			h_rp_ds->SetLineColorAlpha(kMagenta, 0.3);
			h_rp_ds->DrawClone("hist same");
			delete h_rp_ds; 
		} else if (rp_ds_map.size()==0) {}
		else {
			printf("[Error] Don't expect\n");
			exit(0);
		}

		TLegend* leg = canv2->BuildLegend(.5, .6, .85, .8);
		// formatLegend(leg);
		gPad->Modified();
		gPad->Update();
		delete h_sc;

		
		canv2->SaveAs(Form("%s_%sUnmatchedStripClusters_idx%d.png",expTag.c_str(),prefix.c_str(),sc.idx));

		delete canv2;
		if (!faster || prefix=="Warning2_") printf("===========================================\n");
	   }

	}

	return 0;
}

