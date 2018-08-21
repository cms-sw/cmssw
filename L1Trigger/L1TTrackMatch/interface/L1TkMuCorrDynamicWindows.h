#ifndef L1TKMUCORRDYNAMICWINDOWS_H
#define L1TKMUCORRDYNAMICWINDOWS_H

#include <iostream>
#include <functional>
#include <array>
#include <vector>
#include <tuple>
#include <string>
#include <utility>
#include <stdlib.h>
#include "TFile.h"
// #include "MuTkTree.h" // my interface to the ntuples
#include "TMath.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"

#include "L1Trigger/L1TTrackMatch/interface/MuMatchWindow.h"
#include "L1Trigger/L1TMuonEndCap/interface/Common.h"

class L1TkMuCorrDynamicWindows{
    
    public:

        typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
        typedef std::vector< L1TTTrackType >       L1TTTrackCollectionType;

        L1TkMuCorrDynamicWindows(std::vector<double>& bounds, TFile* fIn_theta, TFile* fIn_phi);
        ~L1TkMuCorrDynamicWindows(){};
        // void test(double eta, double pt);
        // std::vector<int> find_match(const EMTFTrackCollection& l1mus, const L1TTTrackCollectionType& l1tks l1trks, std::vector<int>* narbitrated = nullptr); // gives a vector with the idxs of muons for each L1TTT
        std::vector<int> find_match(const EMTFTrackCollection& l1mus, const L1TTTrackCollectionType& l1trks); // gives a vector with the idxs of muons for each L1TTT

        // ------------------------------
        static std::vector<double> prepare_corr_bounds(string fname, string hname);

        void set_safety_factor (float sf_l, float sf_h) {
            safety_factor_l_ = sf_l;
            safety_factor_h_ = sf_h;
            // std::cout << "L1TkMuCorrDynamicWindows : safety factor LOW is " << safety_factor_l_ << std::endl;
            // std::cout << "L1TkMuCorrDynamicWindows : safety factor HIGH is " << safety_factor_h_ << std::endl;
        }
        void set_sf_initialrelax (float sf_l, float sf_h) {
            initial_sf_l_ = sf_l;
            initial_sf_h_ = sf_h;
            // std::cout << "L1TkMuCorrDynamicWindows : initial relax safety factor LOW is " << initial_sf_l_ << std::endl;
            // std::cout << "L1TkMuCorrDynamicWindows : initial relax safety factor HIGH is " << initial_sf_h_ << std::endl;
        }
        void set_relaxation_pattern(float pt_start, float pt_end) {
            pt_start_ = pt_start;
            pt_end_   = pt_end;
            // std::cout << "L1TkMuCorrDynamicWindows : set relaxing from " << pt_start_ << " to " << pt_end_ << std::endl;
        }
        void set_safety_factor (float sf) {set_safety_factor(sf,sf);}
        void set_sf_initialrelax (float sf) {set_sf_initialrelax(sf,sf);}
        void set_do_relax_factor (bool val) {
            do_relax_factor_ = val;
            // std::cout << "L1TkMuCorrDynamicWindows : set do_relax to " << std::boolalpha << do_relax_factor_ << std::noboolalpha << std::endl;
        }

        void set_do_trk_qual_presel (bool val) { track_qual_presel_ = val; }

        // setters for trk
        void set_n_trk_par      (int val)    {nTrkPars_       = val;}
        void set_min_trk_p      (float val ) {min_trk_p_      = val;}
        void set_max_trk_aeta   (float val ) {max_trk_aeta_   = val;}
        void set_max_trk_chi2   (float val ) {max_trk_chi2_   = val;}
        void set_min_trk_nstubs (int   val ) {min_trk_nstubs_ = val;}

        // getters for trk
        int   get_n_trk_par      () {return nTrkPars_ ;}
        float get_min_trk_p      () {return min_trk_p_ ;}
        float get_max_trk_aeta   () {return max_trk_aeta_ ;}
        float get_max_trk_chi2   () {return max_trk_chi2_ ;}
        int   get_min_trk_nstubs () {return min_trk_nstubs_ ;}


    private:
        int getBin(double val);

        // resolves ambiguities to give max 1 tkmu per EMTF
        // if a pointer to narbitrated is passed, this vector is filled with the number of tracks arbitrated that were matched to the same EMTF
        // std::vector<int> make_unique_coll(MuTkTree& mtkt, std::vector<int> matches, std::vector<int>* narbitrated = nullptr);
        // std::vector<int> make_unique_coll(const EMTFTrackCollection& l1mus, const L1TTTrackCollectionType& l1trks, std::vector<int> matches, std::vector<int>* narbitrated = nullptr);
        std::vector<int> make_unique_coll(const EMTFTrackCollection& l1mus, const L1TTTrackCollectionType& l1trks, std::vector<int> matches);

        // converters
        double deg_to_rad(double x) {
            return (x * TMath::Pi()/180.) ;
        }

        double eta_to_theta(double x){
            //  give theta in rad 
            return (2. * TMath::ATan(TMath::Exp(-1.*x)));
        }

        double to_mpio2_pio2(double x){
            //  put the angle in radians between -pi/2 and pi/2
            while (x >= 0.5*TMath::Pi())
                x -= TMath::Pi();
            while (x < -0.5*TMath::Pi())
                x += TMath::Pi();
            return x;
        }

        double to_mpi_pi (double x){
            while (x >= TMath::Pi())
                x -= 2.*TMath::Pi();
            while (x < -TMath::Pi())
                x += 2.*TMath::Pi();
            return x;
        }

        double sf_progressive (double x, double xstart, double xstop, double ystart, double ystop)
        {
            if (x < xstart)
                return ystart;
            if (x >= xstart && x < xstop)
                return ystart + (x-xstart)*(ystop-ystart)/(xstop-xstart);
            return ystop;
        }

        int nbins_; // counts the number of MatchWindow = bounds_.size() - 1
        std::vector<double> bounds_; // counts the boundaries of the MatchWindow (in eta/theta)
        std::vector<MuMatchWindow> wdws_theta_;
        std::vector<MuMatchWindow> wdws_phi_;
        float safety_factor_l_; // increase the lower theta/phi threshold by this fractions
        float safety_factor_h_; // increase the upper theta/phi threshold by this fractions
        float initial_sf_l_; // the start of the relaxation
        float initial_sf_h_; // the start of the relaxation
        float pt_start_; // the relaxation of the threshold
        float pt_end_; // the relaxation of the threshold
        bool  do_relax_factor_; // true if applying the linear relaxation
        bool  track_qual_presel_; // if true, apply the track preselection

        // trk configurable params
        int nTrkPars_; // 4
        float min_trk_p_; // 3.5
        float max_trk_aeta_; // 2.5
        float max_trk_chi2_; // 100
        int   min_trk_nstubs_; // 4
};

#endif