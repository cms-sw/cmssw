#include <cmath>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/TPUtils.h"

namespace emtf::phase2::tp {

    // _______________________________________________________________________
    // radians <-> degrees
    float deg_to_rad(float deg) {
        constexpr float factor = M_PI / 180.;
        return deg * factor;
    }

    float rad_to_deg(float rad) {
        constexpr float factor = 180. / M_PI;

        return rad * factor;
    }

    // _______________________________________________________________________
    // phi range: [-180..180] or [-pi..pi]
    float wrap_phi_deg(float deg) {
        float twopi = 360.;
        float recip = 1.0 / twopi;

        return deg - (std::round(deg * recip) * twopi);
    }

    float wrap_phi_rad(float rad) {
        const float twopi = M_PI * 2.;
        const float recip = 1.0 / twopi;

        return rad - (std::round(rad * recip) * twopi);
    }

    // _______________________________________________________________________
    // theta
    float calc_theta_rad_from_eta(float eta) {
        float theta = std::atan2(1.0, std::sinh(eta));  // cot(theta) = sinh(eta)

        return theta;
    }

    float calc_theta_deg_from_eta(float eta) {
        float theta = rad_to_deg(calc_theta_rad_from_eta(eta));

        return theta;
    }

    float calc_theta_deg_from_int(int theta_int) {
        float theta = static_cast<float>(theta_int);

        theta = theta * (45.0 - 8.5) / 128. + 8.5;

        return theta;
    }

    float calc_theta_rad_from_int(int theta_int) {
        float theta = deg_to_rad(calc_theta_deg_from_int(theta_int));

        return theta;
    }

    int calc_theta_int(int endcap, float theta) {  // theta in deg [0..180], endcap [-1, +1]
        theta = (endcap == -1) ? (180. - theta) : theta;
        theta = (theta - 8.5) * 128. / (45.0 - 8.5);

        int theta_int = static_cast<int>(std::round(theta));

        theta_int = (theta_int <= 0) ? 1 : theta_int;  // protect against invalid value

        return theta_int;
    }

    // _______________________________________________________________________
    // phi
    float calc_phi_glob_deg_from_loc(int sector, float loc) {  // loc in deg, sector [1..6]
        float glob = loc + 15. + (60. * (sector - 1));

        glob = (glob >= 180.) ? (glob - 360.) : glob;

        return glob;
    }

    float calc_phi_glob_rad_from_loc(int sector, float loc) {  // loc in rad, sector [1..6]
        float glob = deg_to_rad(calc_phi_glob_deg_from_loc(sector, rad_to_deg(loc)));

        return glob;
    }

    float calc_phi_loc_deg_from_int(int phi_int) {
        float loc = static_cast<float>(phi_int);

        loc = (loc / 60.) - 22.;

        return loc;
    }

    float calc_phi_loc_rad_from_int(int phi_int) {
        float loc = deg_to_rad(calc_phi_loc_deg_from_int(phi_int));

        return loc;
    }

    float calc_phi_loc_deg_from_glob(int sector, float glob) {  // glob in deg [-180..180], sector [1..6]
        glob = wrap_phi_deg(glob);

        float loc = glob - 15. - (60. * (sector - 1));

        return loc;
    }

    int calc_phi_int(int sector, float glob) {  // glob in deg [-180..180], sector [1..6]
        float loc = calc_phi_loc_deg_from_glob(sector, glob);

        loc = ((loc + 22.) < 0.) ? (loc + 360.) : loc;
        loc = (loc + 22.) * 60.;

        int phi_int = static_cast<int>(std::round(loc));

        return phi_int;
    }

}
