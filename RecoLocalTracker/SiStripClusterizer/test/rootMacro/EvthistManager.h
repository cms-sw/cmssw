#ifndef EVTHIST_MANAGER_H
#define EVTHIST_MANAGER_H

#include "histManagerBase.h"
#include "TrackAlgo.h"
#include "TMath.h"

namespace trk_cuts {
enum  {nocut=1, chi2, ptRes, nhits};
}
map<int, std::string> trk_cutToname = { {trk_cuts::nocut, "nocut"}, {trk_cuts::chi2, "normalizedchi2<2"}, {trk_cuts::ptRes, "abs(pTErr/pT)<0.10"}, {trk_cuts::nhits, "nhits>=11"}};

namespace jet_cuts{
enum jet_cuts {nocut=1, pt, eta};
}
map<int, std::string> jet_cutToname = { {jet_cuts::nocut, "nocut"}, {jet_cuts::pt, "pt>20"}, {jet_cuts::eta, "|eta|<2.4"}};

constexpr Double_t array_displaced_xy[] = {0,0.5, 1, 1.5,2,3,4,5, 10,20};
constexpr Double_t array_displaced_z[] = {0,1.0,1.5,2,3,4,5,6,7,8,9,10, 15, 20, 30, 40, 50, 70};
constexpr Double_t array_xy[] = {0,0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,1,1.2,1.4,2};
constexpr Double_t array_z[] = {0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,1.2, 1.4,1.6,1.8,2,2.5,3,3.5,4.5,5,6,10};
constexpr Int_t displaced_trk_algo[] = {8, 9, 10, 26};

class EvthistManager
   : public histManagerBase
{
     public:
       EvthistManager(string obj):
       histManagerBase(obj) {

         ///// trk ///
	 hists["trk_pt"] = createhist(Form("%s_track_pt_tot", base_name.c_str()), "track_pt;track_pt;yield", numBins, customBins);
	 hists["trk_eta"] = createhist(Form("%s_track_eta", base_name.c_str()), "track_eta;track_eta;yield", 16, -2.4, 2.4);
	 hists["trk_dxyDdxyerr"] = createhist(Form("%s_track_dxyDdxyerr", base_name.c_str()), "track_dxyDdxyerr;xy/#sigma;yield", 30, -5., 5.);
         hists["trk_dzDdzerr"] = createhist(Form("%s_track_dzDdzerr", base_name.c_str()), "track_dzDdzerr;z/#sigma;yield", 50, 0., 10.);
         hists["trk_chi2"] = createhist(Form("%s_track_chi2", base_name.c_str()), "track_chi2;track_chi2;yield", 100, 0,20);
         hists["trk_nhits"] = createhist(Form("%s_track_nhits", base_name.c_str()), "track_nhits;track_nhits;yield", 100, -0.5, 99.5);
         hists["trk_pterrDpt"] = createhist(Form("%s_track_pterrDpt", base_name.c_str()), "track_pterrDpt;track_pTErr/pt;yield", 50, 0, 1);
         hists["trk_inner_xy"] = createhist(Form("%s_track_inner_xy", base_name.c_str()), ";inner_xy;yield", 200, -10., 10.);
         hists["trk_inner_z"] = createhist(Form("%s_track_inner_z", base_name.c_str()), ";inner_z;yield", 400, -50, 50);

         hists["trk_cutflow"] = createhist(Form("%s_trk_cutflow", base_name.c_str()), "cutflow;;yield", trk_cuts::nhits, trk_cuts::nocut, trk_cuts::nhits+1);
         hists["trk_cutflow"]->GetYaxis()->SetLabelSize(0.025);
         for(int ibin=trk_cuts::nocut; ibin<=trk_cuts::nhits; ibin++)
             hists["trk_cutflow"]->GetXaxis()->SetBinLabel(ibin, trk_cutToname[ibin].c_str());

         for (int i=TrackAlgorithm::undefAlgorithm; i<=TrackAlgorithm::displacedRegionalStep; i++) {
           bool displaced_region = std::find(std::begin(displaced_trk_algo), std::end(displaced_trk_algo), i) != std::end(displaced_trk_algo);
           std::string name = "trk_cutflow_z"+to_string(i);
           //if (displaced_region)
           hists_2d[name] = createhist(Form("%s_%s", base_name.c_str(), name.c_str()), Form("%s;abs(z) coordinate;cutflow", algoNames[i].c_str()), 30,0.,60., trk_cuts::nhits, trk_cuts::nocut, trk_cuts::nhits+1);
           //else
              //hists_2d[name] = createhist(Form("%s_%s", base_name.c_str(), name.c_str()), Form("%s;abs(z) coordinate;cutflow", algoNames[i].c_str()), 30,0,60.,trk_cuts::nhits, trk_cuts::nocut, trk_cuts::nhits+1);
           hists_2d[name]->GetYaxis()->SetLabelSize(0.025);
           for(int ibin=trk_cuts::nocut; ibin<=trk_cuts::nhits; ibin++)
	     hists_2d[name]->GetYaxis()->SetBinLabel(ibin, trk_cutToname[ibin].c_str());

           name = "trk_cutflow_xy"+to_string(i);
           //if (displaced_region)
           hists_2d[name] = createhist(Form("%s_%s", base_name.c_str(), name.c_str()), Form("%s;abs(xy) coordinate;cutflow", algoNames[i].c_str()), 5,0,10, trk_cuts::nhits, trk_cuts::nocut, trk_cuts::nhits+1);
           //else
               //hists_2d[name] = createhist(Form("%s_%s", base_name.c_str(), name.c_str()), Form("%s;abs(xy) coordinate;cutflow", algoNames[i].c_str()), 5,0,10, trk_cuts::nhits, trk_cuts::nocut, trk_cuts::nhits+1);
           hists_2d[name]->GetYaxis()->SetLabelSize(0.025);
           for(int ibin=trk_cuts::nocut; ibin<=trk_cuts::nhits; ibin++)
             hists_2d[name]->GetYaxis()->SetBinLabel(ibin, trk_cutToname[ibin].c_str());

        }
	hists_2d["trk_eta_phi"] = createhist(Form("%s_track_eta_phi", base_name.c_str()), "track_eta_phi;#eta; #phi;yield", 16, -2.4, 2.4, 30, -TMath::Pi(), TMath::Pi());

       ///// jet ////

       hists["jet_pt"]      = createhist(Form("%s_jet_pt_tot", base_name.c_str()), "jet_pt;jet_pt;yield", 20, 20, 100);
       hists["jet_eta"]     = createhist(Form("%s_jet_eta", base_name.c_str()), "jet_eta;jet_eta;yield", 16, -2.4, 2.4);
       hists["jet_cutflow"] = createhist(Form("%s_jet_cutflow", base_name.c_str()), "cutflow;cutflow;yield", jet_cuts::eta, jet_cuts::nocut, jet_cuts::eta+1);
       for(int ibin=jet_cuts::nocut; ibin<=jet_cuts::eta; ibin++)
           hists["jet_cutflow"]->GetXaxis()->SetBinLabel(ibin, jet_cutToname[ibin].c_str());
       };
};


#endif //EVTHIST_MANAGER
