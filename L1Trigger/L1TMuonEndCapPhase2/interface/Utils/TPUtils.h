#ifndef L1Trigger_L1TMuonEndCapPhase2_TPUtils_h
#define L1Trigger_L1TMuonEndCapPhase2_TPUtils_h

namespace emtf::phase2::tp {

    // _______________________________________________________________________
    // radians <-> degrees
    float deg_to_rad(float deg);

    float rad_to_deg(float rad);

    // _______________________________________________________________________
    // phi range: [-180..180] or [-pi..pi]
    float wrap_phi_deg(float);

    float wrap_phi_rad(float);

    // _______________________________________________________________________
    // theta
    float calc_theta_rad_from_eta(float);

    float calc_theta_deg_from_eta(float);

    float calc_theta_deg_from_int(int);

    float calc_theta_rad_from_int(int);

    int calc_theta_int(int, float);

    // _______________________________________________________________________
    // phi
    float calc_phi_glob_deg_from_loc(int, float);

    float calc_phi_glob_rad_from_loc(int, float);

    float calc_phi_loc_deg_from_int(int);

    float calc_phi_loc_rad_from_int(int);

    float calc_phi_loc_deg_from_glob(int, float);

    int calc_phi_int(int, float);

}

#endif // namespace L1Trigger_L1TMuonEndCapPhase2_TPUtils_h
