#include <cmath>
#include "DataFormats/Math/interface/approx_asin.h"

// 2asin(cd/2)/c
float arc(float c, float d) {
  float z = 0.5f * c * d;
  z *= z;
  float x = d;

  return ((((4.2163199048E-2f * z + 2.4181311049E-2f) * z + 4.5470025998E-2f) * z + 7.4953002686E-2f) * z +
          1.6666752422E-1f) *
             z * x +
         x;
}

#include <iostream>
int main() {
  float d = 10;
  for (float c = 0.1; c > 1.e-20; c *= 0.75)
    std::cout << c << " " << 2. * asin(0.5 * double(c) * double(d)) / double(c) << " "
              << 2.f * std::asin(0.5f * c * d) / c << " " << 2.f * unsafe_asin07<5>(0.5f * c * d) / c << " "
              << arc(c, d) << std::endl;

  for (float c = -3.15; c < 3.15; c += 0.1)
    std::cout << c << " " << std::sin(c) << ' ' << std::cos(c) << ' ' << unsafe_asin07<5>(std::sin(c)) << ' '
              << unsafe_acos07<5>(std::cos(c)) << ' ' << unsafe_asin71<5>(std::sin(c)) << ' '
              << unsafe_acos71<5>(std::cos(c)) << ' ' << unsafe_asin<5>(std::sin(c)) << ' '
              << unsafe_acos<5>(std::cos(c)) << ' ' << std::endl;

  for (float c = -1; c < 1.05; c += 0.1) {
    auto d = std::acos(c) - unsafe_acos<11>(c);
    auto e = std::asin(c) - unsafe_asin<11>(c);
    std::cout << c << ' ' << d << ' ' << e << std::endl;
  }

  return 0;
}
