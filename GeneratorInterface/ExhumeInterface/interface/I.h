#ifndef GeneratorInterface_ExhumeInterface_I_h
#define GeneratorInterface_ExhumeInterface_I_h
#include <complex>
namespace Exhume {
const std::complex<double> _i_sq_ = -1.0;
const std::complex<double> _sqrt_i_sq_ = sqrt(_i_sq_);
//#define I _sqrt_i_sq_
const std::complex<double> I = _sqrt_i_sq_;
}
#endif
