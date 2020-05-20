#ifndef L1Trigger_TrackFindingTracklet_interface_HistImp_h
#define L1Trigger_TrackFindingTracklet_interface_HistImp_h

#include <TFile.h>
#include <TH1F.h>
#include <TEfficiency.h>

#include "L1Trigger/TrackFindingTracklet/interface/HistBase.h"

namespace trklet {

  class Settings;
  class Globals;

  class HistImp : public HistBase {
  public:
    HistImp();

    ~HistImp() = default;

    void open() override;
    void close() override;

    void bookLayerResidual() override;
    void bookDiskResidual() override;
    void bookTrackletParams() override;
    void bookSeedEff() override;

    void fillTrackletParams(const Settings *settings,
                            Globals *globals,
                            int seedIndex,
                            int iSector,
                            double rinv,
                            double irinv,
                            double phi0,
                            double iphi0,
                            double eta,
                            double ieta,
                            double z0,
                            double iz0,
                            int tp) override;

    void FillLayerResidual(
        int layer, int seed, double phiresid, double iphiresid, double zresid, double izresid, bool match) override;

    void FillDiskResidual(
        int disk, int seed, double phiresid, double iphiresid, double rresid, double irresid, bool match) override;

    //Efficiency for finding seed
    void fillSeedEff(int seedIndex, double etaTP, bool eff) override;

  private:
    TFile *h_file_;

    //Layer residuales
    TH1F *h_layerresid_phi_L3_L1L2_;
    TH1F *h_layerresid_phi_L3_L1L2_match_;
    TH1F *h_layerresid_phif_L3_L1L2_;
    TH1F *h_layerresid_phif_L3_L1L2_match_;
    TH1F *h_layerresid_z_L3_L1L2_;
    TH1F *h_layerresid_z_L3_L1L2_match_;
    TH1F *h_layerresid_zf_L3_L1L2_;
    TH1F *h_layerresid_zf_L3_L1L2_match_;

    //Disk residuals
    TH1F *h_diskresid_phi_D1_L1L2_;
    TH1F *h_diskresid_phi_D1_L1L2_match_;
    TH1F *h_diskresid_phif_D1_L1L2_;
    TH1F *h_diskresid_phif_D1_L1L2_match_;
    TH1F *h_diskresid_r_D1_L1L2_;
    TH1F *h_diskresid_r_D1_L1L2_match_;
    TH1F *h_diskresid_rf_D1_L1L2_;
    TH1F *h_diskresid_rf_D1_L1L2_match_;

    //Tracklet parameters
    TH1F *h_rinv_L1L2_;
    TH1F *h_irinv_L1L2_;
    TH1F *h_rinv_matched_L1L2_;
    TH1F *h_irinv_matched_L1L2_;
    TH1F *h_rinvres_L1L2_;
    TH1F *h_irinvres_L1L2_;
    TH1F *h_phi0_L1L2_;
    TH1F *h_iphi0_L1L2_;
    TH1F *h_phi0_matched_L1L2_;
    TH1F *h_iphi0_matched_L1L2_;
    TH1F *h_phi0global_L1L2_;
    TH1F *h_iphi0global_L1L2_;
    TH1F *h_phi0global_matched_L1L2_;
    TH1F *h_iphi0global_matched_L1L2_;
    TH1F *h_phi0res_L1L2_;
    TH1F *h_iphi0res_L1L2_;
    TH1F *h_eta_L1L2_;
    TH1F *h_ieta_L1L2_;
    TH1F *h_eta_matched_L1L2_;
    TH1F *h_ieta_matched_L1L2_;
    TH1F *h_etares_L1L2_;
    TH1F *h_ietares_L1L2_;
    TH1F *h_z0_L1L2_;
    TH1F *h_iz0_L1L2_;
    TH1F *h_z0_matched_L1L2_;
    TH1F *h_iz0_matched_L1L2_;
    TH1F *h_z0res_L1L2_;
    TH1F *h_iz0res_L1L2_;

    //seeding efficiency
    TEfficiency *h_eff_eta_L1L2seed_;
    TEfficiency *h_eff_eta_L2L3seed_;
    TEfficiency *h_eff_eta_L3L4seed_;
    TEfficiency *h_eff_eta_L5L6seed_;
    TEfficiency *h_eff_eta_D1D2seed_;
    TEfficiency *h_eff_eta_D3D4seed_;
    TEfficiency *h_eff_eta_D1L1seed_;
    TEfficiency *h_eff_eta_D1L2seed_;
  };

};  // namespace trklet
#endif
