#ifndef L1TKMUMANTRA_H
#define L1TKMUMANTRA_H

/*
** class  : GenericDataFormat
** author : L.Cadamuro (UF)
** date   : 4/11/2019
** brief  : very generic structs to be used as inputs to the correlator
**        : to make sure that Mantra can handle muons and tracks from all the detectors
*/

namespace L1TkMuMantraDF {

    struct track_df {
        double pt;     // GeV
        double eta;    // rad, -inf / +inf
        double theta;  // rad, 0 -> +90-90
        double phi;    // rad, -pi / + pi
        int    nstubs; // 
        double chi2;   // 
        int    charge; // -1. +1 
    };

    struct muon_df {
        double pt;     // GeV
        double eta;    // rad, -inf / +inf
        double theta;  // rad, 0 -> +90-90
        double phi;    // rad, -pi / + pi
        int    charge; // -1. +1 
    };
}


/*
** class  : L1TkMuMantra
** author : L.Cadamuro (UF)
** date   : 4/11/2019
** brief  : correlates muons and tracks using pre-encoded windows
*/

#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include "L1Trigger/L1TTrackMatch/interface/MuMatchWindow.h"
// #include "GenericDataFormat.h"
#include "TFile.h"
#include "TMath.h"


class L1TkMuMantra
{
    public:
        L1TkMuMantra(std::vector<double>& bounds, TFile* fIn_theta, TFile* fIn_phi, std::string name);
        ~L1TkMuMantra(){};

        // returns a vector with the same size of muons, each with an index to the matched L1 track, or -1 if no match is found
        std::vector<int> find_match(std::vector<L1TkMuMantraDF::track_df>& tracks, std::vector<L1TkMuMantraDF::muon_df>& muons);

        void test(double eta, double pt);

        void relax_windows   (double& low, double cent, double& high); // will modify low and high

        void set_safety_factor (float sf_l, float sf_h) {
            safety_factor_l_ = sf_l;
            safety_factor_h_ = sf_h;
            if (verbosity_ > 0) std::cout << "L1TkMuMantra : " << name_ << " safety factor LOW is "  << safety_factor_l_ << std::endl;
            if (verbosity_ > 0) std::cout << "L1TkMuMantra : " << name_ << " safety factor HIGH is " << safety_factor_h_ << std::endl;
        }

        int sign(double x){
            if (x == 0)
                return 1;
            return int (x/std::abs(x));
        }

        void setArbitrationType (std::string type); // MaxPt, MinDeltaPt


        // static functions, meant to be used from outside to interface with MAnTra
        static std::vector<double> prepare_corr_bounds(std::string fname, std::string hname);

        // converters
        static double deg_to_rad(double x) {
            return (x * TMath::Pi()/180.) ;
        }

        static double eta_to_theta(double x){
            //  give theta in rad 
            return (2. * TMath::ATan(TMath::Exp(-1.*x)));
        }

        static double to_mpio2_pio2(double x){
            //  put the angle in radians between -pi/2 and pi/2
            while (x >= 0.5*TMath::Pi())
                x -= TMath::Pi();
            while (x < -0.5*TMath::Pi())
                x += TMath::Pi();
            return x;
        }

        static double to_mpi_pi (double x){
            while (x >= TMath::Pi())
                x -= 2.*TMath::Pi();
            while (x < -TMath::Pi())
                x += 2.*TMath::Pi();
            return x;
        }


    private:

        int getBin(double val);

        std::string name_;

        int nbins_; // counts the number of MuMatchWindow = bounds_.size() - 1
        std::vector<double> bounds_; // counts the boundaries of the MuMatchWindow (in eta/theta)
        std::vector<MuMatchWindow> wdws_theta_;
        std::vector<MuMatchWindow> wdws_phi_;

        int    min_nstubs = 4;   // >= min_nstubs
        double max_chi2   = 100; // < max_chi2

        float safety_factor_l_; // increase the lower theta/phi threshold by this fractions w.r.t. the center
        float safety_factor_h_; // increase the upper theta/phi threshold by this fractions w.r.t. the center


        // float initial_sf_l_; // the start of the relaxation
        // float initial_sf_h_; // the start of the relaxation
        // float pt_start_; // the relaxation of the threshold
        // float pt_end_; // the relaxation of the threshold
        // bool  do_relax_factor_; // true if applying the linear relaxation

        enum sortParType {
            kMaxPt,     // pick the highest pt track matched 
            kMinDeltaPt // pick the track with the smallest pt difference w.r.t the muon
        };

        sortParType sort_type_;

        int verbosity_ = 0;
};

#endif // L1TKMUMANTRA_H