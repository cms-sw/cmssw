#ifndef __L1Analysis_L1AnalysisRecoMuon_H__
#define __L1Analysis_L1AnalysisRecoMuon_H__

#include <TTree.h>
#include <iostream>
#include <vector>

namespace L1Analysis
{
  class L1AnalysisRecoMuon
  {
    
  public :
    void initTree(TTree * tree, const std::string & className);
  
  public:
    L1AnalysisRecoMuon() {}
    void print();
    
    // ---- General L1AnalysisRecoMuon information.    
    int           nMuons;
     std::vector<int>     muon_type;
     std::vector<double>  muons_ch;
     std::vector<double>  muons_pt;
     std::vector<double>  muons_p;
     std::vector<double>  muons_eta;
     std::vector<double>  muons_phi;
     std::vector<double>  muons_validhits;
     std::vector<double>  muons_numberOfMatchedStations;
     std::vector<double>  muons_numberOfValidMuonHits;
     std::vector<double>  muons_normchi2;
     std::vector<double>  muons_imp_point_x;
     std::vector<double>  muons_imp_point_y;
     std::vector<double>  muons_imp_point_z;
     std::vector<double>  muons_imp_point_p;
     std::vector<double>  muons_imp_point_pt;
     std::vector<double>  muons_phi_hb;
     std::vector<double>  muons_z_hb;
     std::vector<double>  muons_r_he_p;
     std::vector<double>  muons_r_he_n;
     std::vector<double>  muons_phi_he_p;
     std::vector<double>  muons_phi_he_n;
     std::vector<double>  muons_tr_ch;
     std::vector<double>  muons_tr_pt;
     std::vector<double>  muons_tr_p;
     std::vector<double>  muons_tr_eta;
     std::vector<double>  muons_tr_phi;
     std::vector<double>  muons_tr_validhits;
     std::vector<double>  muons_tr_validpixhits;
     std::vector<double>  muons_tr_normchi2;
     std::vector<double>  muons_tr_d0;
     std::vector<double>  muons_tr_imp_point_x;
     std::vector<double>  muons_tr_imp_point_y;
     std::vector<double>  muons_tr_imp_point_z;
     std::vector<double>  muons_tr_imp_point_p;
     std::vector<double>  muons_tr_imp_point_pt;
     std::vector<double>  muons_sa_phi_mb2;
     std::vector<double>  muons_sa_z_mb2;
     std::vector<double>  muons_sa_pseta;
     std::vector<double>  muons_sa_normchi2;
     std::vector<double>  muons_sa_validhits;
     std::vector<double>  muons_sa_ch;
     std::vector<double>  muons_sa_pt;
     std::vector<double>  muons_sa_p;
     std::vector<double>  muons_sa_eta;
     std::vector<double>  muons_sa_phi;
     std::vector<double>  muons_sa_outer_pt;
     std::vector<double>  muons_sa_inner_pt;
     std::vector<double>  muons_sa_outer_eta;
     std::vector<double>  muons_sa_inner_eta;
     std::vector<double>  muons_sa_outer_phi;
     std::vector<double>  muons_sa_inner_phi;
     std::vector<double>  muons_sa_outer_x;
     std::vector<double>  muons_sa_outer_y;
     std::vector<double>  muons_sa_outer_z;
     std::vector<double>  muons_sa_inner_x;
     std::vector<double>  muons_sa_inner_y;
     std::vector<double>  muons_sa_inner_z;
     std::vector<double>  muons_sa_imp_point_x;
     std::vector<double>  muons_sa_imp_point_y;
     std::vector<double>  muons_sa_imp_point_z;
     std::vector<double>  muons_sa_imp_point_p;
     std::vector<double>  muons_sa_imp_point_pt;
     std::vector<double>  muons_sa_phi_hb;
     std::vector<double>  muons_sa_z_hb;
     std::vector<double>  muons_sa_r_he_p;
     std::vector<double>  muons_sa_r_he_n;
     std::vector<double>  muons_sa_phi_he_p;
     std::vector<double>  muons_sa_phi_he_n;
};
}


#endif

#ifdef l1ntuple_cxx


void L1Analysis::L1AnalysisRecoMuon::initTree(TTree * tree, const std::string & className)
    {
     SetBranchAddress(tree, "nMuons", className,  &nMuons);
     SetBranchAddress(tree, "muon_type", className,  &muon_type);
     SetBranchAddress(tree, "muons_ch", className,  &muons_ch);
     SetBranchAddress(tree, "muons_pt", className,  &muons_pt);
     SetBranchAddress(tree, "muons_p", className,  &muons_p);
     SetBranchAddress(tree, "muons_eta", className,  &muons_eta);
     SetBranchAddress(tree, "muons_phi", className,  &muons_phi);
     SetBranchAddress(tree, "muons_validhits", className,  &muons_validhits);
     SetBranchAddress(tree, "muons_numberOfMatchedStations", className,  &muons_numberOfMatchedStations);
     SetBranchAddress(tree, "muons_numberOfValidMuonHits", className,  &muons_numberOfValidMuonHits);
     SetBranchAddress(tree, "muons_normchi2", className,  &muons_normchi2);
     SetBranchAddress(tree, "muons_imp_point_x", className,  &muons_imp_point_x);
     SetBranchAddress(tree, "muons_imp_point_y", className,  &muons_imp_point_y);
     SetBranchAddress(tree, "muons_imp_point_z", className,  &muons_imp_point_z);
     SetBranchAddress(tree, "muons_imp_point_p", className,  &muons_imp_point_p);
     SetBranchAddress(tree, "muons_imp_point_pt", className,  &muons_imp_point_pt);
     SetBranchAddress(tree, "muons_phi_hb", className,  &muons_phi_hb);
     SetBranchAddress(tree, "muons_z_hb", className,  &muons_z_hb);
     SetBranchAddress(tree, "muons_r_he_p", className,  &muons_r_he_p);
     SetBranchAddress(tree, "muons_r_he_n", className,  &muons_r_he_n);
     SetBranchAddress(tree, "muons_phi_he_p", className,  &muons_phi_he_p);
     SetBranchAddress(tree, "muons_phi_he_n", className,  &muons_phi_he_n);
     SetBranchAddress(tree, "muons_tr_ch", className,  &muons_tr_ch);
     SetBranchAddress(tree, "muons_tr_pt", className,  &muons_tr_pt);
     SetBranchAddress(tree, "muons_tr_p", className,  &muons_tr_p);
     SetBranchAddress(tree, "muons_tr_eta", className,  &muons_tr_eta);
     SetBranchAddress(tree, "muons_tr_phi", className,  &muons_tr_phi);
     SetBranchAddress(tree, "muons_tr_validhits", className,  &muons_tr_validhits);
     SetBranchAddress(tree, "muons_tr_validpixhits", className,  &muons_tr_validpixhits);
     SetBranchAddress(tree, "muons_tr_normchi2", className,  &muons_tr_normchi2);
     SetBranchAddress(tree, "muons_tr_d0", className,  &muons_tr_d0);
     SetBranchAddress(tree, "muons_tr_imp_point_x", className,  &muons_tr_imp_point_x);
     SetBranchAddress(tree, "muons_tr_imp_point_y", className,  &muons_tr_imp_point_y);
     SetBranchAddress(tree, "muons_tr_imp_point_z", className,  &muons_tr_imp_point_z);
     SetBranchAddress(tree, "muons_tr_imp_point_p", className,  &muons_tr_imp_point_p);
     SetBranchAddress(tree, "muons_tr_imp_point_pt", className,  &muons_tr_imp_point_pt);
     SetBranchAddress(tree, "muons_sa_phi_mb2", className,  &muons_sa_phi_mb2);
     SetBranchAddress(tree, "muons_sa_z_mb2", className,  &muons_sa_z_mb2);
     SetBranchAddress(tree, "muons_sa_pseta", className,  &muons_sa_pseta);
     SetBranchAddress(tree, "muons_sa_normchi2", className,  &muons_sa_normchi2);
     SetBranchAddress(tree, "muons_sa_validhits", className,  &muons_sa_validhits);
     SetBranchAddress(tree, "muons_sa_ch", className,  &muons_sa_ch);
     SetBranchAddress(tree, "muons_sa_pt", className,  &muons_sa_pt);
     SetBranchAddress(tree, "muons_sa_p", className,  &muons_sa_p);
     SetBranchAddress(tree, "muons_sa_eta", className,  &muons_sa_eta);
     SetBranchAddress(tree, "muons_sa_phi", className,  &muons_sa_phi);
     SetBranchAddress(tree, "muons_sa_outer_pt", className,  &muons_sa_outer_pt);
     SetBranchAddress(tree, "muons_sa_inner_pt", className,  &muons_sa_inner_pt);
     SetBranchAddress(tree, "muons_sa_outer_eta", className,  &muons_sa_outer_eta);
     SetBranchAddress(tree, "muons_sa_inner_eta", className,  &muons_sa_inner_eta);
     SetBranchAddress(tree, "muons_sa_outer_phi", className,  &muons_sa_outer_phi);
     SetBranchAddress(tree, "muons_sa_inner_phi", className,  &muons_sa_inner_phi);
     SetBranchAddress(tree, "muons_sa_outer_x", className,  &muons_sa_outer_x);
     SetBranchAddress(tree, "muons_sa_outer_y", className,  &muons_sa_outer_y);
     SetBranchAddress(tree, "muons_sa_outer_z", className,  &muons_sa_outer_z);
     SetBranchAddress(tree, "muons_sa_inner_x", className,  &muons_sa_inner_x);
     SetBranchAddress(tree, "muons_sa_inner_y", className,  &muons_sa_inner_y);
     SetBranchAddress(tree, "muons_sa_inner_z", className,  &muons_sa_inner_z);
     SetBranchAddress(tree, "muons_sa_imp_point_x", className,  &muons_sa_imp_point_x);
     SetBranchAddress(tree, "muons_sa_imp_point_y", className,  &muons_sa_imp_point_y);
     SetBranchAddress(tree, "muons_sa_imp_point_z", className,  &muons_sa_imp_point_z);
     SetBranchAddress(tree, "muons_sa_imp_point_p", className,  &muons_sa_imp_point_p);
     SetBranchAddress(tree, "muons_sa_imp_point_pt", className,  &muons_sa_imp_point_pt);
     SetBranchAddress(tree, "muons_sa_phi_hb", className,  &muons_sa_phi_hb);
     SetBranchAddress(tree, "muons_sa_z_hb", className,  &muons_sa_z_hb);
     SetBranchAddress(tree, "muons_sa_r_he_p", className,  &muons_sa_r_he_p);
     SetBranchAddress(tree, "muons_sa_r_he_n", className,  &muons_sa_r_he_n);
     SetBranchAddress(tree, "muons_sa_phi_he_p", className,  &muons_sa_phi_he_p);
     SetBranchAddress(tree, "muons_sa_phi_he_n", className,  &muons_sa_phi_he_n);
    }

void L1Analysis::L1AnalysisRecoMuon::print()
{

}

#endif


