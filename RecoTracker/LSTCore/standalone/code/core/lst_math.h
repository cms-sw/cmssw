#ifndef lst_math_h
#define lst_math_h
#include <tuple>
#include <vector>
#include <cmath>

namespace lst_math {
  inline float Phi_mpi_pi(float x) {
    if (std::isnan(x)) {
      std::cout << "MathUtil::Phi_mpi_pi() function called with NaN" << std::endl;
      return x;
    }

    while (x >= M_PI)
      x -= 2. * M_PI;

    while (x < -M_PI)
      x += 2. * M_PI;

    return x;
  };

  inline float ATan2(float y, float x) {
    if (x != 0)
      return atan2(y, x);
    if (y == 0)
      return 0;
    if (y > 0)
      return M_PI / 2;
    else
      return -M_PI / 2;
  };

  inline float ptEstimateFromRadius(float radius) { return 2.99792458e-3 * 3.8 * radius; };

  class Helix {
  public:
    std::vector<float> center_;
    float radius_;
    float phi_;
    float lam_;
    float charge_;

    Helix(std::vector<float> c, float r, float p, float l, float q) {
      center_ = c;
      radius_ = r;
      phi_ = Phi_mpi_pi(p);
      lam_ = l;
      charge_ = q;
    }

    Helix(float pt, float eta, float phi, float vx, float vy, float vz, float charge) {
      // Radius based on pt
      radius_ = pt / (2.99792458e-3 * 3.8);
      phi_ = phi;
      charge_ = charge;

      // reference point vector which for sim track is the vertex point
      float ref_vec_x = vx;
      float ref_vec_y = vy;
      float ref_vec_z = vz;

      // The reference to center vector
      float inward_radial_vec_x = charge_ * radius_ * sin(phi_);
      float inward_radial_vec_y = charge_ * radius_ * -cos(phi_);
      float inward_radial_vec_z = 0;

      // Center point
      float center_vec_x = ref_vec_x + inward_radial_vec_x;
      float center_vec_y = ref_vec_y + inward_radial_vec_y;
      float center_vec_z = ref_vec_z + inward_radial_vec_z;
      center_.push_back(center_vec_x);
      center_.push_back(center_vec_y);
      center_.push_back(center_vec_z);

      // Lambda
      lam_ = copysign(M_PI / 2. - 2. * atan(exp(-abs(eta))), eta);
    }

    const std::vector<float> center() { return center_; }
    const float radius() { return radius_; }
    const float phi() { return phi_; }
    const float lam() { return lam_; }
    const float charge() { return charge_; }

    std::tuple<float, float, float, float> get_helix_point(float t) {
      float x = center()[0] - charge() * radius() * sin(phi() - (charge()) * t);
      float y = center()[1] + charge() * radius() * cos(phi() - (charge()) * t);
      float z = center()[2] + radius() * tan(lam()) * t;
      float r = sqrt(x * x + y * y);
      return std::make_tuple(x, y, z, r);
    }

    float infer_t(const std::vector<float> point) {
      // Solve for t based on z position
      float t = (point[2] - center()[2]) / (radius() * tan(lam()));
      return t;
    }

    float compare_radius(const std::vector<float> point) {
      float t = infer_t(point);
      auto [x, y, z, r] = get_helix_point(t);
      float point_r = sqrt(point[0] * point[0] + point[1] * point[1]);
      return (point_r - r);
    }

    float compare_xy(const std::vector<float> point) {
      float t = infer_t(point);
      auto [x, y, z, r] = get_helix_point(t);
      float xy_dist = sqrt(pow(point[0] - x, 2) + pow(point[1] - y, 2));
      return xy_dist;
    }
  };

  class Hit {
  public:
    float x_, y_, z_;
    // Derived quantities
    float r3_, rt_, phi_, eta_;
    int idx_;

    // Default constructor
    Hit() : x_(0), y_(0), z_(0), idx_(-1) { setDerivedQuantities(); }

    // Parameterized constructor
    Hit(float x, float y, float z, int idx = -1) : x_(x), y_(y), z_(z), idx_(idx) { setDerivedQuantities(); }

    // Copy constructor
    Hit(const Hit& hit) : x_(hit.x_), y_(hit.y_), z_(hit.z_), idx_(hit.idx_) { setDerivedQuantities(); }

    // Getters for derived quantities
    float phi() const { return phi_; }
    float eta() const { return eta_; }
    float rt() const { return rt_; }
    float x() const { return x_; }
    float y() const { return y_; }

  private:
    void setDerivedQuantities() {
      // Setting r3
      r3_ = sqrt(x_ * x_ + y_ * y_ + z_ * z_);

      // Setting rt
      rt_ = sqrt(x_ * x_ + y_ * y_);

      // Setting phi
      phi_ = Phi_mpi_pi(M_PI + ATan2(-y_, -x_));

      // Setting eta
      eta_ = ((z_ > 0) - (z_ < 0)) * std::acosh(r3_ / rt_);
    }
  };

  inline Hit getCenterFromThreePoints(Hit& hitA, Hit& hitB, Hit& hitC) {
    //       C
    //
    //
    //
    //    B           d <-- find this point that makes the arc that goes throw a b c
    //
    //
    //     A

    // Steps:
    // 1. Calculate mid-points of lines AB and BC
    // 2. Find slopes of line AB and BC
    // 3. construct a perpendicular line between AB and BC
    // 4. set the two equations equal to each other and solve to find intersection

    float xA = hitA.x();
    float yA = hitA.y();
    float xB = hitB.x();
    float yB = hitB.y();
    float xC = hitC.x();
    float yC = hitC.y();

    float x_mid_AB = (xA + xB) / 2.;
    float y_mid_AB = (yA + yB) / 2.;

    //float x_mid_BC = (xB + xC) / 2.;
    //float y_mid_BC = (yB + yC) / 2.;

    bool slope_AB_inf = (xB - xA) == 0;
    bool slope_BC_inf = (xC - xB) == 0;

    bool slope_AB_zero = (yB - yA) == 0;
    bool slope_BC_zero = (yC - yB) == 0;

    float slope_AB = slope_AB_inf ? 0 : (yB - yA) / (xB - xA);
    float slope_BC = slope_BC_inf ? 0 : (yC - yB) / (xC - xB);

    float slope_perp_AB = slope_AB_inf or slope_AB_zero ? 0. : -1. / (slope_AB);
    //float slope_perp_BC = slope_BC_inf or slope_BC_zero ? 0. : -1. / (slope_BC);

    if ((slope_AB - slope_BC) == 0) {
      std::cout << " slope_AB_zero: " << slope_AB_zero << std::endl;
      std::cout << " slope_BC_zero: " << slope_BC_zero << std::endl;
      std::cout << " slope_AB_inf: " << slope_AB_inf << std::endl;
      std::cout << " slope_BC_inf: " << slope_BC_inf << std::endl;
      std::cout << " slope_AB: " << slope_AB << std::endl;
      std::cout << " slope_BC: " << slope_BC << std::endl;
      std::cout << "MathUtil::getCenterFromThreePoints() function the three points are in straight line!" << std::endl;
      return Hit();
    }

    float x =
        (slope_AB * slope_BC * (yA - yC) + slope_BC * (xA + xB) - slope_AB * (xB + xC)) / (2. * (slope_BC - slope_AB));
    float y = slope_perp_AB * (x - x_mid_AB) + y_mid_AB;

    return Hit(x, y, 0);
  };
}  // namespace lst_math
#endif
