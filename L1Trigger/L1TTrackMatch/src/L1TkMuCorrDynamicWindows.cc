#include "L1Trigger/L1TTrackMatch/interface/L1TkMuCorrDynamicWindows.h"

// ROOT includes
#include "TH1.h"
#include "TH2.h"

L1TkMuCorrDynamicWindows::L1TkMuCorrDynamicWindows(std::vector<double>& bounds, TFile* fIn_theta, TFile* fIn_phi) :
    wdws_theta_(bounds.size()-1, MuMatchWindow()),
    wdws_phi_(bounds.size()-1, MuMatchWindow())
{
    set_safety_factor(0.5);
    set_sf_initialrelax(0.0);
    set_relaxation_pattern(2.0, 6.0);
    set_do_relax_factor(true);

    track_qual_presel_ = true;

    nbins_ = bounds.size()-1;
    for (double b : bounds)
        bounds_.push_back(b);

    // now load in memory the TF1 fits

    for (int ib = 0; ib < nbins_; ++ib)
    {
        std::string wdn;
        std::string nml;
        std::string nmh;
        TF1* fl;
        TF1* fh;

        wdn = std::string("wdw_theta_") + std::to_string(ib+1);
        nml = std::string("fit_low_")   + std::to_string(ib+1);
        nmh = std::string("fit_high_")  + std::to_string(ib+1);
        fl = (TF1*) fIn_theta->Get(nml.c_str());
        fh = (TF1*) fIn_theta->Get(nmh.c_str());
        if (fl == nullptr || fh == nullptr)
            throw std::runtime_error("Could not init theta");        
        wdws_theta_.at(ib).SetName(wdn);
        wdws_theta_.at(ib).SetLower(fl);
        wdws_theta_.at(ib).SetUpper(fh);


        wdn = std::string("wdw_phi_") + std::to_string(ib+1);
        nml = std::string("fit_low_")   + std::to_string(ib+1);
        nmh = std::string("fit_high_")  + std::to_string(ib+1);
        fl = (TF1*) fIn_phi->Get(nml.c_str());
        fh = (TF1*) fIn_phi->Get(nmh.c_str());
        if (fl == nullptr || fh == nullptr)
            throw std::runtime_error("Could not init phi");        
        wdws_phi_.at(ib).SetName(wdn);
        wdws_phi_.at(ib).SetLower(fl);
        wdws_phi_.at(ib).SetUpper(fh);
    }
}


int L1TkMuCorrDynamicWindows::getBin(double val)
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

    std::cout << "Something strange happened at val " << val << std::endl;
    return 0;
}

// void L1TkMuCorrDynamicWindows::test(double eta, double pt)
// {
//     int ibin = getBin(eta);
//     std::cout << "- eta : " << eta << " pt: " << pt << std::endl; 
//     std::cout << ">>> bin " << ibin << std::endl;
//     std::cout << ">>> low_phi : "   << wdws_phi_.at(ibin).bound_low(pt)   << " , high_phi : " << wdws_phi_.at(ibin).bound_high(pt) << std::endl;
//     std::cout << ">>> low_theta : " << wdws_theta_.at(ibin).bound_low(pt) << " , high_theta : " << wdws_theta_.at(ibin).bound_high(pt) << std::endl;
//     return;
// }

// std::vector<int> L1TkMuCorrDynamicWindows::find_match(MuTkTree& mtkt, std::vector<int>* narbitrated)
std::vector<int> L1TkMuCorrDynamicWindows::find_match(const EMTFTrackCollection& l1mus, const L1TTTrackCollectionType& l1trks)
{
    // const int nTrkPars_ = 4; // FIXME: make confiugurable from python cfg

    std::vector<int> out (l1trks.size());
    for (auto l1trkit = l1trks.begin(); l1trkit != l1trks.end(); ++l1trkit)
    {
        float trk_pt      = l1trkit->getMomentum(nTrkPars_).perp();
        float trk_p       = l1trkit->getMomentum(nTrkPars_).mag();
        float trk_aeta    = std::abs(l1trkit->getMomentum(nTrkPars_).eta());
        float trk_theta   = to_mpio2_pio2(eta_to_theta(l1trkit->getMomentum(nTrkPars_).eta()));
        float trk_phi     = l1trkit->getMomentum(nTrkPars_).phi();
        int   trk_charge  = (l1trkit->getRInv(nTrkPars_) > 0 ? 1 : -1);

        // porting some selections from the MuonTrackCorr finder
        // https://github.com/cms-l1t-offline/cmssw/blob/l1t-phase2-932-v1.6/L1Trigger/L1TTrackMatch/plugins/L1TkMuonProducer.cc#L264
        // FIXME: make preselections confiuguable
        bool reject_trk = false;
        if (trk_p    < min_trk_p_ )   reject_trk = true;
        if (trk_aeta > max_trk_aeta_) reject_trk = true;
        if (track_qual_presel_)
        {
            float l1tk_chi2 = l1trkit->getChi2(nTrkPars_);
            int l1tk_nstubs = l1trkit->getStubRefs().size();
            if (l1tk_chi2 >= max_trk_chi2_)    reject_trk = true;
            if (l1tk_nstubs < min_trk_nstubs_) reject_trk = true;
        }

        int ibin = getBin(trk_aeta);

        std::vector<std::tuple<float, float, int>> matched; // dtheta, dphi, idx
        // loop on muons to see which match
        // for (uint im = 0; im < *(mtkt.n_EMTF_mu); ++im)
        for (auto l1muit = l1mus.begin(); l1muit != l1mus.end(); ++l1muit)
        {
            // match only muons in the central bx - as the track collection refers anyway to bx 0 only
            if (l1muit->BX() != 0)
                continue;

            // putting everything in rad
            float emtf_theta  = to_mpio2_pio2(eta_to_theta(l1muit->Eta())) ;
            float emtf_phi    = deg_to_rad(l1muit->Phi_glob()) ;

            float dtheta = std::abs(emtf_theta - trk_theta);
            float dphi   = to_mpi_pi(emtf_phi - trk_phi);
            float adphi  = std::abs(dphi);


            double sf_l;
            double sf_h;
            if (do_relax_factor_)
            {
                sf_l = sf_progressive(trk_pt, pt_start_, pt_end_, initial_sf_l_, safety_factor_l_);
                sf_h = sf_progressive(trk_pt, pt_start_, pt_end_, initial_sf_h_, safety_factor_h_);
            }
            else
            {
                sf_l = safety_factor_l_;
                sf_h = safety_factor_h_;
            }

            // double sf_l = sf_progressive(trk_pt, pt_start_, pt_end_, 0.0, safety_factor_l_);
            // double sf_h = sf_progressive(trk_pt, pt_start_, pt_end_, 0.0, safety_factor_h_);


            if (
                // emtf_theta * trk_theta > 0 &&
                dtheta >  (1 - sf_l) * wdws_theta_.at(ibin).bound_low(trk_pt)  &&
                dtheta <= (1 + sf_h) * wdws_theta_.at(ibin).bound_high(trk_pt) &&
                adphi  >  (1 - sf_l) * wdws_phi_.at(ibin).bound_low(trk_pt)    &&
                adphi  <= (1 + sf_h) * wdws_phi_.at(ibin).bound_high(trk_pt)   &&
                dphi*trk_charge < 0                                            && // sign requirement
                // rndm > 0.5
                true
            )
                matched.push_back(std::make_tuple(dtheta, adphi, std::distance(l1mus.begin(), l1muit)));
            // else if (emtf_theta * trk_theta > 0)
            // {
            //     std::cout << "=== DEBUG ===" << std::endl;
            //     if (! (dtheta >  (1 - safety_factor_l) * wdws_theta_.at(ibin).bound_low(trk_pt)  )) std::cout << "FAIL dtheta low -- " << dtheta << " " << wdws_theta_.at(ibin).bound_low(trk_pt) << std::endl;
            //     if (! (dtheta <= (1 + safety_factor_h) * wdws_theta_.at(ibin).bound_high(trk_pt) )) std::cout << "FAIL dtheta high -- " << dtheta << " " << wdws_theta_.at(ibin).bound_high(trk_pt) << std::endl;
            //     if (! (dphi   >  (1 - safety_factor_l) * wdws_phi_.at(ibin).bound_low(trk_pt)    )) std::cout << "FAIL dphi low -- " << dphi << " " << wdws_phi_.at(ibin).bound_low(trk_pt) << std::endl;
            //     if (! (dphi   <= (1 + safety_factor_h) * wdws_phi_.at(ibin).bound_high(trk_pt)   )) std::cout << "FAIL dphi high -- " << dphi << " " << wdws_phi_.at(ibin).bound_high(trk_pt) << std::endl;
            // }
        }

        if (reject_trk)
            matched.clear(); // quick fix - to be optimised to avoid the operations above

        if (matched.size() == 0)
            out.at(std::distance(l1trks.begin(), l1trkit)) = -1;
        else
        {
            std::sort(matched.begin(), matched.end()); // closest in theta, then in phi
            out.at(std::distance(l1trks.begin(), l1trkit)) = std::get<2>(matched.at(0));
        }
    }

    // return out;

    // now convert out to a unique set
    // auto unique_out = make_unique_coll(mtkt, out, narbitrated);
    auto unique_out = make_unique_coll(l1mus, l1trks, out);

    // auto unique_out = out;
    // if (narbitrated) narbitrated->resize(unique_out.size(), 99);
    
    return unique_out;
}

// std::vector<int> L1TkMuCorrDynamicWindows::make_unique_coll(MuTkTree& mtkt, std::vector<int> matches, std::vector<int>* narbitrated)
// std::vector<int> L1TkMuCorrDynamicWindows::make_unique_coll(const EMTFTrackCollection& l1mus, const L1TTTrackCollectionType& l1trks, std::vector<int> matches, std::vector<int>* narbitrated)
std::vector<int> L1TkMuCorrDynamicWindows::make_unique_coll(const EMTFTrackCollection& l1mus, const L1TTTrackCollectionType& l1trks, std::vector<int> matches)
{
    std::vector<int> out (matches.size(), -1);

    // if (narbitrated)
    //     narbitrated->resize(matches.size(), 0);

    std::vector<std::vector<int>> macthed_to_emtf(l1mus.size(), std::vector<int>(0)); // one vector of matched trk idx per EMTF

    for (unsigned int itrack = 0; itrack < matches.size(); ++itrack)
    {
        int iemtf = matches.at(itrack);
        if (iemtf < 0) continue;
        macthed_to_emtf.at(iemtf).push_back(itrack);
    }

    // this sorts by by the trk (a < b if pta < ptb)
    // std::function<bool(int, int, MuTkTree&)> track_less_than_proto = [](int idx1, int idx2, MuTkTree& mtrktr)
    // {
    //     float pt1 = mtrktr.L1TT_trk_pt.At(idx1);
    //     float pt2 = mtrktr.L1TT_trk_pt.At(idx2);
    //     return (pt1 < pt2);
    // };

    std::function<bool(int, int, const L1TTTrackCollectionType&, int)> track_less_than_proto = [](int idx1, int idx2, const L1TTTrackCollectionType& l1trkcoll, int nTrackParams)
    {
        float pt1 = l1trkcoll.at(idx1).getMomentum(nTrackParams).perp();
        float pt2 = l1trkcoll.at(idx2).getMomentum(nTrackParams).perp();
        return (pt1 < pt2);
    };

    // // and binds to accept only 2 params
    // std::function<bool(int,int)> track_less_than = std::bind(track_less_than_proto, std::placeholders::_1, std::placeholders::_2, std::ref(l1mus));
    std::function<bool(int,int)> track_less_than = std::bind(track_less_than_proto, std::placeholders::_1, std::placeholders::_2, l1trks, nTrkPars_);


    for (unsigned int iemtf = 0; iemtf < macthed_to_emtf.size(); ++iemtf)
    {
        std::vector<int>& thisv = macthed_to_emtf.at(iemtf);
        if (thisv.size() == 0) continue;
        
        // std::cout << " === BEFORE === " << std::endl;
        // for (int idx : macthed_to_emtf.at(iemtf))
        //     std::cout << mtkt.L1TT_trk_pt.At(idx) << std::endl;

        std::sort(thisv.begin(), thisv.end(), track_less_than);

        // std::cout << " === AFTER === " << std::endl;
        // for (int idx : macthed_to_emtf.at(iemtf))
        //     std::cout << mtkt.L1TT_trk_pt.At(idx) << std::endl;

        // copy to the output
        int best_trk = thisv.back();
        out.at(best_trk) = iemtf;

        // if (narbitrated)
        //     narbitrated->at(best_trk) = thisv.size();
    }

    return out;
}


std::vector<double> L1TkMuCorrDynamicWindows::prepare_corr_bounds(string fname, string hname)
{
    // find the boundaries of the match windoww
    TFile* fIn = TFile::Open(fname.c_str());
    TH2* h_test = (TH2*) fIn->Get(hname.c_str());
    if (h_test == nullptr)
    {
        // cout << "Can't find histo to derive bounds" << endl;
        throw std::runtime_error("Can't find histo to derive bounds");
    }

    int nbds = h_test->GetNbinsY()+1;
    // cout << "... using " << nbds-1 << " eta bins" << endl;
    vector<double> bounds (nbds);
    for (int ib = 0; ib < nbds; ++ib)
    {
        bounds.at(ib) = h_test->GetYaxis()->GetBinLowEdge(ib+1);
        // cout << "Low edge " << ib << " is " << bounds.at(ib) << endl;
    }
    fIn->Close();
    return bounds;
}
