#include <cmath>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

GENERATE_SOA_LAYOUT(SoATemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(double, v_x),
                    SOA_COLUMN(double, v_y),
                    SOA_COLUMN(double, v_z),

                    SOA_ELEMENT_METHODS(

                        SOA_HOST_DEVICE void normalise() {
                          float norm_position = square_norm_position();
                          if (norm_position > 0.0f) {
                            x() /= norm_position;
                            y() /= norm_position;
                            z() /= norm_position;
                          }
                          double norm_velocity = square_norm_velocity();
                          if (norm_velocity > 0.0f) {
                            v_x() /= norm_velocity;
                            v_y() /= norm_velocity;
                            v_z() /= norm_velocity;
                          }
                        }),

                    SOA_CONST_ELEMENT_METHODS(
                        SOA_HOST_DEVICE float square_norm_position()
                            const { return sqrt(x() * x() + y() * y() + z() * z()); }

                        SOA_HOST_DEVICE double square_norm_velocity()
                            const { return sqrt(v_x() * v_x() + v_y() * v_y() + v_z() * v_z()); }

                        template <typename T1, typename T2>
                        SOA_HOST_DEVICE static auto time(T1 pos, T2 vel) {
                          if (vel != 0)
                            return pos / vel;
                          return 0.;
                        }),

                    SOA_SCALAR(int, detectorType))

using SoA = SoATemplate<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

// clang-format off
GENERATE_SOA_LAYOUT(PositionLayout,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z))

GENERATE_SOA_LAYOUT(VelocityLayout,
                    SOA_COLUMN(float, vx),
                    SOA_COLUMN(float, vy),
                    SOA_COLUMN(float, vz))

GENERATE_SOA_BLOCKS(PointsLayout,
                    SOA_BLOCK(position, PositionLayout),
                    SOA_BLOCK(velocity, VelocityLayout),
                    SOA_VIEW_METHODS(
                        SOA_HOST_DEVICE void update_position(uint32_t i, float time) {
                            auto pos = this->position()[i];
                            auto vel = this->velocity()[i];
                            pos.x() += vel.vx() * time;
                            pos.y() += vel.vy() * time;
                            pos.z() += vel.vz() * time;
                        }
                    ),
                    SOA_CONST_VIEW_METHODS(
                        SOA_HOST_DEVICE auto distance2(uint32_t i, uint32_t j) const {
                            auto pi = this->position()[i];
                            auto pj = this->position()[j];
                            return (pi.x() - pj.x()) * (pi.x() - pj.x()) + 
                                   (pi.y() - pj.y()) * (pi.y() - pj.y()) + 
                                   (pi.z() - pj.z()) * (pi.z() - pj.z());
                        }
                    )
)
// clang-format on

using Points = PointsLayout<>;
using PointsView = Points::View;
using PointsConstView = Points::ConstView;
