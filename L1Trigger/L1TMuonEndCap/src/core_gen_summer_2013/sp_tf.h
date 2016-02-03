// This C++ header file was automatically generated
// by VPPC from a Verilog HDL project.
// VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

// Author    : madorsky
// Timestamp : Fri Feb  1 08:50:46 2013

#ifndef __sp_tf_h_file__
#define __sp_tf_h_file__

#include <memory>
#include <map>

#include <stdio.h>
#include <stdlib.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "L1Trigger/L1TMuon/interface/deprecate/SubsystemCollectorFactory.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include <TMath.h>
//#include <TCanvas.h>
#include <TLorentzVector.h>

#include "TTree.h"
#include "TNtuple.h"

#include <TStyle.h>
#include <TLegend.h>
#include <TF1.h>
#include <TH2.h>
#include <TH1F.h>
#include <TFile.h>
#include "L1Trigger/L1TMuon/interface/deprecate/GeometryTranslator.h"

#include "L1Trigger/L1TMuonEndCap/interface/MuonInternalTrack.h"
#include "L1Trigger/L1TMuonEndCap/interface/MuonInternalTrackFwd.h"

#include "TFile.h"
#include "TH1.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>
#include <iostream>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "Riostream.h"
#include "vppc_sim_lib.h"
#include "sp.h"


/////// FROM MATT'S CODE ///////////

#include "L1Trigger/CSCCommonTrigger/interface/CSCPatternLUT.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

///////////////////////////////////////


/////////////////////  From Matt's Code  ////////////////////////
//#include "L1Trigger/L1TMuonEndCap/plugins/L1TMuonTFAlgorithm.h"


using namespace L1TMuon;
using namespace edm;
using namespace reco;

typedef edm::ParameterSet PSet;

//class sptf : public edm::EDProducer
class sptf : public edm::EDProducer
{
 public:

    sptf(const PSet&);
    //sptf(std::vector<edm::InputTag>, std::vector<edm::InputTag>);
    ~sptf() {}
    sptf() {cout << "Hello there.\n";}
    void produce(edm::Event&, const edm::EventSetup&);
    void beginJob();
    //void beginRun(const edm::Run&, const edm::EventSetup&);
    //void endRun(const edm::Run&, const edm::EventSetup&);
    void endJob();
    //void runEvent();

 private:


    vector<int> the_bxValue;
    vector< vector<int> > the_primSelector;
    vector< vector<int> > the_inputOrder;
    vector< vector<int> > the_bx_jitter;
    vector< vector<int> > the_endcap;
    vector< vector<int> > the_sector;
    vector< vector<int> > the_subsector;
    vector< vector<int> > the_station;
    vector< vector<int> > the_valid;
    vector< vector<int> > the_quality;
    vector< vector<int> > the_pattern;
    vector< vector<int> > the_wiregroup;
    vector< vector<int> > the_cscid;
    vector< vector<int> > the_bend;
    vector< vector<int> > the_halfstrip;

    vector<int> the_emuPhi;
    vector<int> the_emuTheta;
    vector<int> the_emuPhhit;
    vector<int> the_emuPhzvl;
    vector<int> the_emuStrip;
    vector<int> the_emuStraight;
    vector<int> the_emuQuality;
    vector<int> the_emuLayer;

    vector<int> GetPhi() {return the_emuPhi;}
    vector<int> GetTheta() {return the_emuTheta;}
    vector<int> GetPhhit() {return the_emuPhhit;}
    vector<int> GetPhzvl() {return the_emuPhzvl;}
    vector<int> GetStrip() {return the_emuStrip;}
    vector<int> GetStraight() {return the_emuStraight;}
    vector<int> GetLayer() {return the_emuLayer;}
    vector<int> GetQuality() {return the_emuQuality;}

    int count;

    void defparam();
	void build();
	bool built;
	bool glbl_gsr;


    //std::vector<ConvertedHit> ConvHits;
    //edm::ParameterSet LUTparam;
    //CSCSectorReceiverLUT* srLUTs_[5][2];

    //these probably need to be killed wherever they exist due to conflict with local "station" variables.
    //    unsigned _station;
    //    unsigned _cscid;

    static const unsigned _max_ev     = 21;     // maximum events [hack]
    static const unsigned _seg_ch     = 2;     // segments per chamber
    static const unsigned _bw_ph      = 8;     // bit widths of ph and th outputs, reduced precision
    static const unsigned _bw_th      = 7;     // have to be derived from pattern width on top level
    static const unsigned _bw_fph     = 12;     // bit widths of ph and th, full precision
    static const unsigned _bw_fth     = 8;
    static const unsigned _bw_wg      = 7;     // wiregroup input bit width (0..111)
    static const unsigned _bw_ds      = 7;     // bit width of dblstrip input (max 80 for ME234/1 with double-width strips)
	static const unsigned _bw_hs      = 8;     // width of halfstrip input
	static const unsigned _pat_w_st3  = 3;     // pattern half-width for stations 3,4
    static const unsigned _pat_w_st1       = _pat_w_st3 + 1;              // pattern half-width for station 1
    static const unsigned _full_pat_w_st3  = (1 << (_pat_w_st3+1)) - 1;   // number of input bits for stations 3,4
    static const unsigned _full_pat_w_st1  = (1 << (_pat_w_st1+1)) - 1;   // number of input bits for st 1
    static const unsigned _padding_w_st1   = _full_pat_w_st1 / 2;         // width of zero padding for station copies
    static const unsigned _padding_w_st3   = _full_pat_w_st3 / 2;
    static const unsigned _red_pat_w_st3   = _pat_w_st3 * 2 + 1;          // full pattern widths (aka reduced pattern)
    static const unsigned _red_pat_w_st1   = _pat_w_st1 * 2 + 1;
    static const unsigned _th_ch11         = _seg_ch * _seg_ch;           // number of th outputs for ME1/1
    static const unsigned _ph_raw_w        = (1 << _pat_w_st3) * 15;      // strips per section, calculated so ph pattern would cover +/- 8 deg in st 1
    static const unsigned _th_raw_w        = (1 << _bw_th);
    static const unsigned _fold       =         4;     // number of folds for pattern detectors, do not set to 1
    static const unsigned _bw_q       =         4;
    static const unsigned _bw_addr    =         7;
    static const unsigned _max_drift  =         3;     // max possible drifttime
    static const unsigned _bw_phi     =        12;     // bit widths of precise phi and eta outputs
    static const unsigned _bw_eta     =         7;
    static const unsigned _ph_hit_w   =        44;     // (40+4) width of ph raw hits, max coverage +8 to cover possible chamber displacement
    static const unsigned _ph_hit_w20 = _ph_hit_w;     // for 20 deg chambers
    static const unsigned _ph_hit_w10 =        24;     // (20+4) for 10 deg chambers
    static const unsigned _th_hit_w   =        64;     // (56+8) width of th raw hits, max coverage +8 to cover possible chamber displacement
    static const unsigned _th_mem_sz       = (1 << _bw_addr);
    static const unsigned _th_corr_mem_sz  = (1 << _bw_addr);
    static const unsigned _mult_bw         = _bw_fph + 11;  
    static const unsigned _zone_overlap    =  2;
    static const unsigned _bwr         = 6;   // rank width
    static const unsigned _bpow        = 6;   // (1 << _bpow) is count of input ranks
    static const unsigned _cnr         = (1 << _bpow); //internal rank count
    static const unsigned _cnrex       = _ph_raw_w; // actual input rank count, must be even

///		unsigned endcap;
///		unsigned n_strips;
///		unsigned n_wg;
///		// theta range (take +1 because th_coverage contains max th value starting from 0)
///	unsigned th_coverage;
///		// phi range
///	unsigned ph_coverage; //80 : 40;
///		// number of th outputs takes ME1/1 th duplication into account
///	unsigned th_ch;
///		// is this chamber mounted in reverse direction?
///	unsigned ph_reverse;

		// multiplier bit width (phi + factor)
		// ph zone boundaries for chambers that cover more than one zone
// hardcoded boundaries must match boundaries in ph_th_match module

///	unsigned ph_zone_bnd1;
///		unsigned ph_zone_bnd2;
		// sorter parameters
	
			// io
signal_storage qi__storage;  signal_ qi;
		signal_storage wgi__storage;  signal_ wgi;
		signal_storage hstri__storage;  signal_ hstri;
		signal_storage cpati__storage;  signal_ cpati;
			// precise parameters
/*
	 * wire [bw_phi-1:0] phi [2:0];
    wire [bw_eta-1:0] eta [2:0];
    wire [12:0] 	  pt [2:0];
    wire [2:0] 		  sign;
    wire [3:0] 		  modeMem [2:0];
    wire [4:0] 		  etaPT [2:0];
    wire [2:0] 		  FR;
	 */signal_storage csi__storage;  signal_ csi;
		signal_storage pps_csi__storage;  signal_ pps_csi;
		signal_storage seli__storage;  signal_ seli;
		signal_storage addri__storage;  signal_ addri;
		signal_storage r_ini__storage;  signal_ r_ini; // reg data for memory or register
		signal_storage wei__storage;  signal_ wei; // write enable for memory or register
		signal_storage clki__storage;  signal_ clki; // write clock
			// ph_init storage is full-precision now
// chamber index is 0..11 to include ME11a and b
signal_storage ph_init__storage;  signal_ ph_init; // [station][chamber]
			// arrays below contains values for each chamber
// chamber count is ME1=12*2 ME2,3,4=9*3, total 51
signal_storage th_init__storage;  signal_ th_init; // chamber origins in th
		signal_storage ph_disp__storage;  signal_ ph_disp; // chamber displacements in ph
		signal_storage th_disp__storage;  signal_ th_disp; // chamber displacements in th
			// event storage
signal_storage quality__storage;  signal_ quality;
		signal_storage wiregroup__storage;  signal_ wiregroup;
		signal_storage hstrip__storage;  signal_ hstrip;
		signal_storage clctpat__storage;  signal_ clctpat;
		signal_storage v0__storage;  signal_ v0;
		signal_storage v1__storage;  signal_ v1;
		signal_storage v2__storage;  signal_ v2;
		signal_storage v3__storage;  signal_ v3;
		signal_storage v4__storage;  signal_ v4;
		signal_storage v5__storage;  signal_ v5;
		signal_storage pr_cnt__storage;  signal_ pr_cnt;
		signal_storage _event__storage;  signal_ _event;
		signal_storage _bx_jitter__storage;  signal_ _bx_jitter;
		signal_storage _endcap__storage;  signal_ _endcap;
		signal_storage _sector__storage;  signal_ _sector;
		signal_storage _subsector__storage;  signal_ _subsector;
		signal_storage _station__storage;  signal_ _station;
		signal_storage _cscid__storage;  signal_ _cscid;
		signal_storage _bend__storage;  signal_ _bend;
		signal_storage _halfstrip__storage;  signal_ _halfstrip;
		signal_storage _valid__storage;  signal_ _valid;
		signal_storage _quality__storage;  signal_ _quality;
		signal_storage _pattern__storage;  signal_ _pattern;
		signal_storage _wiregroup__storage;  signal_ _wiregroup;
		signal_storage line__storage;  signal_ line;
		signal_storage ev__storage;  signal_ ev;
		signal_storage good_ev__storage;  signal_ good_ev;
		signal_storage tphi__storage;  signal_ tphi;
		signal_storage a__storage;  signal_ a;
		signal_storage b__storage;  signal_ b;
		signal_storage d__storage;  signal_ d;
		signal_storage pts__storage;  signal_ pts;
	
		signal_storage r_outo__storage;  signal_ r_outo; // output data from memory or register
			// ph quality codes output [zone][key_strip]
signal_storage ph_ranko__storage;  signal_ ph_ranko;
		signal_storage ph__storage;  signal_ ph;
		signal_storage th11__storage;  signal_ th11;
		signal_storage th__storage;  signal_ th;
		signal_storage vl__storage;  signal_ vl;
		signal_storage phzvl__storage;  signal_ phzvl;
		signal_storage me11a__storage;  signal_ me11a;
		signal_storage ph_zone__storage;  signal_ ph_zone;
		signal_storage patt_vi__storage;  signal_ patt_vi; // valid
		signal_storage patt_hi__storage;  signal_ patt_hi; // bx index
		signal_storage patt_ci__storage;  signal_ patt_ci; // chamber
		signal_storage patt_si__storage;  signal_ patt_si; // segment
			// numbers of best ranks [zone][num]
signal_storage ph_num__storage;  signal_ ph_num;
			// best ranks [zone][num]
signal_storage ph_q__storage;  signal_ ph_q;
		signal_storage ph_match__storage;  signal_ ph_match; // matching ph
		signal_storage th_match__storage;  signal_ th_match; // matching th, 2 segments 
		signal_storage th_match11__storage;  signal_ th_match11; // matching th for ME11 (station 0 only), 4 segments (due to th duplication)
			// precise phi and theta of best tracks
// [best_track_num]
signal_storage bt_phi__storage;  signal_ bt_phi;
		signal_storage bt_theta__storage;  signal_ bt_theta;
		signal_storage bt_cpattern__storage;  signal_ bt_cpattern;
			// ph and th deltas from best stations
// [best_track_num], last index: [0] - best pair of stations, [1] - second best pair
signal_storage bt_delta_ph__storage;  signal_ bt_delta_ph;
		signal_storage bt_delta_th__storage;  signal_ bt_delta_th;
		signal_storage bt_sign_ph__storage;  signal_ bt_sign_ph;
		signal_storage bt_sign_th__storage;  signal_ bt_sign_th;
			// ranks [best_track_num]
signal_storage bt_rank__storage;  signal_ bt_rank;
			// segment IDs
// [best_track_num][station 0-3]
signal_storage bt_vi__storage;  signal_ bt_vi; // valid
		signal_storage bt_hi__storage;  signal_ bt_hi; // bx index
		signal_storage bt_ci__storage;  signal_ bt_ci; // chamber
		signal_storage bt_si__storage;  signal_ bt_si; // segment
		signal_storage begin_time__storage;  signal_ begin_time;
		signal_storage end_time__storage;  signal_ end_time;
		signal_storage elapsed_time__storage;  signal_ elapsed_time;
	
		signal_storage iadr__storage;  signal_ iadr;
		signal_storage s__storage;  signal_ s;
		signal_storage i__storage;  signal_ i;
		signal_storage pi__storage;  signal_ pi;
		signal_storage j__storage;  signal_ j;
		signal_storage sn__storage;  signal_ sn;
		signal_storage ist__storage;  signal_ ist;
		signal_storage icid__storage;  signal_ icid;
		signal_storage ipr__storage;  signal_ ipr;
		signal_storage code__storage;  signal_ code;
		signal_storage iev__storage;  signal_ iev;
		signal_storage im__storage;  signal_ im;
		signal_storage iz__storage;  signal_ iz;
		signal_storage ir__storage;  signal_ ir;
		signal_storage in__storage;  signal_ in;
		signal_storage best_tracks__storage;  signal_ best_tracks;
		signal_storage stat__storage;  signal_ stat;
		signal_storage good_ev_cnt__storage;  signal_ good_ev_cnt;
		signal_storage found_tr__storage;  signal_ found_tr;
		signal_storage found_cand__storage;  signal_ found_cand;
		signal_storage st__storage;  signal_ st;
		signal_storage st_cnt__storage;  signal_ st_cnt; // station count
		signal_storage iseg__storage;  signal_ iseg;
		signal_storage zi__storage;  signal_ zi;
		signal_storage si__storage;  signal_ si;
		signal_storage ip__storage;  signal_ ip;
		signal_storage ibx__storage;  signal_ ibx;
		signal_storage ich__storage;  signal_ ich;
		signal_storage isg__storage;  signal_ isg;
		signal_storage ii__storage;  signal_ ii;
		signal_storage kp__storage;  signal_ kp;
		unsigned k;
	
	void init ();
    sp uut;
      std::vector<edm::InputTag> _tpinputs;
      std::vector<edm::InputTag>  _convTrkInputs;
};
#endif
