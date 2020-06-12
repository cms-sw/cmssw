#include "L1Trigger/L1TTrackMatch/interface/L1TkMuMantra.h"
#include "TH2.h"

using namespace L1TkMuMantraDF ;

L1TkMuMantra::L1TkMuMantra(std::vector<double>& bounds, TFile* fIn_theta, TFile* fIn_phi, std::string name = "mantra") :
    wdws_theta_ (bounds.size()-1, MuMatchWindow()),
    wdws_phi_   (bounds.size()-1, MuMatchWindow())
{

    name_ = name;

    safety_factor_l_ = 0.0;
    safety_factor_h_ = 0.0;

    sort_type_ = kMaxPt;

    // copy boundaries
    nbins_ = bounds.size()-1;
    for (double b : bounds)
        bounds_.push_back(b);

    // now load in memory the TF1 fits

    for (int ib = 0; ib < nbins_; ++ib)
    {
        std::string wdn;
        std::string nml;
        std::string nmc;
        std::string nmh;
        TF1* fl;
        TF1* fc;
        TF1* fh;

        wdn = name_ + std::string("_wdw_theta_") + std::to_string(ib+1);
        nml = std::string("fit_low_")   + std::to_string(ib+1);
        nmc = std::string("fit_cent_")  + std::to_string(ib+1);
        nmh = std::string("fit_high_")  + std::to_string(ib+1);
        
        fl = (TF1*) fIn_theta->Get(nml.c_str());
        fc = (TF1*) fIn_theta->Get(nmc.c_str());
        fh = (TF1*) fIn_theta->Get(nmh.c_str());
        if (fl == nullptr || fc == nullptr || fh == nullptr){
            if (verbosity_ > 0) {
                std::cout << "... fit theta low  : " << fl << std::endl;
                std::cout << "... fit theta cent : " << fc << std::endl;
                std::cout << "... fit theta high : " << fh << std::endl;
            }
            throw std::runtime_error("L1TkMuMantra : Could not init theta");        
        }
        wdws_theta_.at(ib).SetName(wdn);
        wdws_theta_.at(ib).SetLower(fl);
        wdws_theta_.at(ib).SetCentral(fc);
        wdws_theta_.at(ib).SetUpper(fh);

        wdn = name_ + std::string("_wdw_phi_") + std::to_string(ib+1);
        nml = std::string("fit_low_")   + std::to_string(ib+1);
        nmc = std::string("fit_cent_")  + std::to_string(ib+1);
        nmh = std::string("fit_high_")  + std::to_string(ib+1);
        fl = (TF1*) fIn_phi->Get(nml.c_str());
        fc = (TF1*) fIn_phi->Get(nmc.c_str());
        fh = (TF1*) fIn_phi->Get(nmh.c_str());
        if (fl == nullptr || fc == nullptr || fh == nullptr){
            if (verbosity_ > 0) {
                std::cout << "... fit phi low  : " << fl << std::endl;
                std::cout << "... fit phi cent : " << fc << std::endl;
                std::cout << "... fit phi high : " << fh << std::endl;
            }
            throw std::runtime_error("L1TkMuMantra : Could not init phi");        
        }
        wdws_phi_.at(ib).SetName(wdn);
        wdws_phi_.at(ib).SetLower(fl);
        wdws_phi_.at(ib).SetCentral(fc);
        wdws_phi_.at(ib).SetUpper(fh);
    }    
}

int L1TkMuMantra::getBin(double val)
{
    // FIXME: not the most efficient, nor the most elegant implementation for now
    if (val < bounds_.at(0))
        return 0;
    if (val >= bounds_.back())
        return (nbins_-1); // i.e. bounds_size() -2

    for (uint ib = 0; ib < bounds_.size()-1; ++ib)
    {
        if (val >= bounds_.at(ib) && val < bounds_.at(ib+1))
            return ib;
    }

    if (verbosity_ > 0) std::cout << "Something strange happened at val " << val << std::endl;
    return 0;
}

void L1TkMuMantra::test(double eta, double pt)
{
    int ibin = getBin(eta);
   
    std::cout << " ---- eta : " << eta << " pt: " << pt << std::endl; 
    std::cout << " ---- bin " << ibin << std::endl;
    std::cout << " ---- "
        << "- low_phi  : "   << wdws_phi_.at(ibin).bound_low(pt)
        << "- cent_phi : "   << wdws_phi_.at(ibin).bound_cent(pt)
        << "- high_phi : "   << wdws_phi_.at(ibin).bound_high(pt)
        << std::endl;

    std::cout << " ---- "
        << "- low_theta  : "   << wdws_theta_.at(ibin).bound_low(pt)
        << "- cent_theta : "   << wdws_theta_.at(ibin).bound_cent(pt)
        << "- high_theta : "   << wdws_theta_.at(ibin).bound_high(pt)
        << std::endl;

    return;
}

std::vector<int> L1TkMuMantra::find_match(std::vector<track_df>& tracks, std::vector<muon_df>& muons)
{
    std::vector<int> result (muons.size(), -1); // init all TkMu to index -1
    for (uint imu = 0; imu < muons.size(); ++imu)
    {
        muon_df mu = muons.at(imu);
        std::vector<std::pair<double, int>> matched_trks; // sort_par, idx
        for (uint itrk = 0; itrk < tracks.size(); ++itrk)
        {
            // preselection of tracks
            track_df trk = tracks.at(itrk);
            if (trk.chi2   >= max_chi2)   continue; // require trk.chi2 < max_chi2
            if (trk.nstubs < min_nstubs) continue; // require trk.nstubs >= min_nstubs

            // compute delta
            // col_dphicharge = dataframe.loc[ : , 'trk_phi'].subtract(dataframe.loc[ : , phi_name]).multiply(dataframe.loc[ : , 'trk_charge']).apply(to_mpi_pi)
            // col_dthetaendc = dataframe.loc[ : , theta_name].subtract(dataframe.loc[ : , 'trk_theta']).multiply(dataframe.loc[ : , etasign_name])
            
            double dphi_charge = to_mpi_pi((trk.phi - mu.phi) * trk.charge);
            double dtheta_endc = (mu.theta - trk.theta) * (std::abs(mu.theta)/mu.theta); // sign from theta, to avoid division by 0
            if (sign(mu.theta) != sign(trk.theta)){ // crossing the barrel -> remove 180 deg to the theta of the neg candidate to avoid jumps at eta = 0
                dtheta_endc -= TMath::Pi();
            }

            // lookup the values
            int ibin = getBin(std::abs(trk.eta));

            double phi_low   = wdws_phi_.at(ibin).bound_low(trk.pt);
            double phi_cent  = wdws_phi_.at(ibin).bound_cent(trk.pt);
            double phi_high  = wdws_phi_.at(ibin).bound_high(trk.pt);
            relax_windows (phi_low, phi_cent, phi_high); // apply the safety factor
            bool   in_phi = (dphi_charge > phi_low && dphi_charge < phi_high);

            double theta_low   = wdws_theta_.at(ibin).bound_low(trk.pt);
            double theta_cent  = wdws_theta_.at(ibin).bound_cent(trk.pt);
            double theta_high  = wdws_theta_.at(ibin).bound_high(trk.pt);
            relax_windows (theta_low, theta_cent, theta_high);  // apply the safety factor
            bool   in_theta = (dtheta_endc > theta_low && dtheta_endc < theta_high);

            // if (!in_theta && in_phi)
            //     std::cout << "DID NOT MATCH : " << std::endl
            //               << trk.pt << " " << trk.eta << " " << trk.theta << " " << trk.phi << std::endl
            //               << mu.pt << " " << mu.eta << " " << mu.theta << " " << mu.phi << std::endl
            //               << dtheta_endc << std::endl;

            if (in_phi && in_theta){
                double sort_par = 99999;
                if (sort_type_ == kMaxPt)
                    sort_par = trk.pt;
                else if (sort_type_ == kMinDeltaPt){
                    sort_par = (trk.pt > 0 ? std::abs(1. - (mu.pt / trk.pt)) : 0); // trk.pt should always be > 0, but put this protection just in case
                }
                matched_trks.push_back(std::make_pair(sort_par, itrk));
            }
        }

        // choose out of the matched tracks the best one
        if (matched_trks.size() > 0)
        {
            sort (matched_trks.begin(), matched_trks.end());
            int ibest = 99999;
            if (sort_type_ == kMaxPt)
                ibest = matched_trks.rbegin() -> second; // sorted low to high -> take last for highest pT (rbegin)
            else if (sort_type_ == kMinDeltaPt)
                ibest = matched_trks.begin() -> second; // sorted low to high -> take first for min pT distance (begin)
            result.at(imu) = ibest;
        }
    }

    return result;
}

void L1TkMuMantra::relax_windows  (double& low, double cent, double& high)
{
    double delta_high = high - cent;
    double delta_low  = cent - low;

    high = high + safety_factor_h_*delta_high;
    low  = low  - safety_factor_l_*delta_low;

    return;
}

void L1TkMuMantra::setArbitrationType (std::string type)
{
    if (verbosity_ > 0) std::cout << "L1TkMuMantra : setting arbitration type to " << type << std::endl;
    if (type == "MaxPt")
        sort_type_ = kMaxPt;
    else if (type == "MinDeltaPt")
        sort_type_ = kMinDeltaPt;
    else
        throw std::runtime_error("L1TkMuMantra : setArbitrationType : cannot understand the arbitration type passed");
}

std::vector<double> L1TkMuMantra::prepare_corr_bounds(std::string fname, std::string hname)
{
    // find the boundaries of the match windoww
    TFile* fIn  = TFile::Open(fname.c_str());
    TH2* h_test = (TH2*) fIn->Get(hname.c_str());
    if (h_test == nullptr)
    {
        // cout << "Can't find histo to derive bounds" << endl;
        throw std::runtime_error("L1TkMuMantra : Can't find histo to derive bounds");
    }

    int nbds = h_test->GetNbinsY()+1;
    // cout << "... using " << nbds-1 << " eta bins" << endl;
    std::vector<double> bounds (nbds);
    for (int ib = 0; ib < nbds; ++ib)
    {
        bounds.at(ib) = h_test->GetYaxis()->GetBinLowEdge(ib+1);
        // cout << "Low edge " << ib << " is " << bounds.at(ib) << endl;
    }
    fIn->Close();
    return bounds;
}
