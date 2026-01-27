#include <cmath>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

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
                          };
                          double norm_velocity = square_norm_velocity();
                          if (norm_velocity > 0.0f) {
                            v_x() /= norm_velocity;
                            v_y() /= norm_velocity;
                            v_z() /= norm_velocity;
                          };
                        }),

                    SOA_CONST_ELEMENT_METHODS(
                        SOA_HOST_DEVICE float square_norm_position()
                            const { return sqrt(x() * x() + y() * y() + z() * z()); };

                        SOA_HOST_DEVICE double square_norm_velocity()
                            const { return sqrt(v_x() * v_x() + v_y() * v_y() + v_z() * v_z()); };

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
