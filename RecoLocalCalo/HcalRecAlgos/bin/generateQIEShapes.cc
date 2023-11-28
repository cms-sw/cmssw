#include <iostream>
#include <sstream>
#include <vector>
#include <cstdint>

//
// Pregenerate QIE Shapes using hardcoded arrays
// This is taken directly from CondFormats/HcalObjects/srcHcalQIEData.cc
// This generation is running upon conditions retrieval typically for the cpu workload
//
// For the GPU workload, it is better to put generated values into constant memory.
// Either this or just use global memory (for global mem, we need getters...).
// Choosign constant memory as thsese
// values are statically known and never change. Any change in any case requires
// recompilation!
//

const float binMin[32] = {-1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                          16, 18, 20, 22, 24, 26, 28, 31, 34, 37, 40, 44, 48, 52, 57, 62};

const float binMin2[64] = {-0.5,  0.5,   1.5,   2.5,   3.5,   4.5,   5.5,   6.5,   7.5,   8.5,   9.5,
                           10.5,  11.5,  12.5,  13.5,  14.5,  // 16 bins with width 1x
                           15.5,  17.5,  19.5,  21.5,  23.5,  25.5,  27.5,  29.5,  31.5,  33.5,  35.5,
                           37.5,  39.5,  41.5,  43.5,  45.5,  47.5,  49.5,  51.5,  53.5,  // 20 bins with width 2x
                           55.5,  59.5,  63.5,  67.5,  71.5,  75.5,  79.5,  83.5,  87.5,  91.5,  95.5,
                           99.5,  103.5, 107.5, 111.5, 115.5, 119.5, 123.5, 127.5, 131.5, 135.5,  // 21 bins with width 4x
                           139.5, 147.5, 155.5, 163.5, 171.5, 179.5, 187.5};  // 7 bins with width 8x

constexpr uint32_t nbins_qie8 = 32;
constexpr uint32_t nbins_qie11 = 64;

void dump(std::vector<float> const& vec, std::string const& name) {
  std::stringstream str;
  str << "float const " << name << "[" << vec.size() << "] = {";
  uint32_t counter = 0;
  for (auto const& value : vec) {
    if (counter % 8 == 0)
      str << std::endl;
    if (counter == vec.size() - 1)
      str << value;
    else
      str << value << ", ";
    counter++;
  }
  str << "};";
  std::cout << str.str() << std::endl;
}

void generate(uint32_t const nbins, float const* initValues, std::vector<float>& values) {
  // preset the first range
  for (uint32_t adc = 0; adc < nbins; adc++)
    values[adc] = initValues[adc];

  // do the rest
  int scale = 1;
  for (uint32_t range = 1; range < 4; range++) {
    int factor = nbins == 32 ? 5 : 8;
    scale *= factor;

    auto const index_offset = range * nbins;
    uint32_t const overlap = nbins == 32 ? 2 : 3;
    values[index_offset] = values[index_offset - overlap];

    for (uint32_t i = 1; i < nbins; i++)
      values[index_offset + i] = values[index_offset + i - 1] + scale * (values[i] - values[i - 1]);
  }

  values[nbins * 4] = 2 * values[nbins * 4 - 1] - values[nbins * 4 - 2];
}

int main(int argc, char* argv[]) {
  //
  // run 128 bins
  //
  std::vector<float> valuesqie8(nbins_qie8 * 4 + 1), valuesqie11(nbins_qie11 * 4 + 1);
  generate(nbins_qie8, binMin, valuesqie8);
  generate(nbins_qie11, binMin2, valuesqie11);

  dump(valuesqie8, std::string{"qie8shape"});
  dump(valuesqie11, std::string{"qie11shape"});

  return 0;
}
