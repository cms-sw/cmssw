#include "CondFormats/HGCalObjects/interface/HGCalDenseIndexerBase.h"
#include <iostream>
#include <iterator>
#include <set>
#include <cassert>

int main() {
  //init an indexer for a MH (HD) module
  //which has 6 ROCs, each with 2 halfs, each with 37 channels
  std::vector<uint32_t> rans{{6, 2, 37}};
  HGCalDenseIndexerBase di(rans);

  //a lambda to print arrays
  auto parray = [](auto v) { std::copy(std::begin(v), std::end(v), std::ostream_iterator<uint32_t>(std::cout, " ")); };

  //test coding/decoding different values
  std::set<uint32_t> allidx;
  for (uint32_t i = 0; i < rans[0]; i++) {
    for (uint32_t j = 0; j < rans[1]; j++) {
      for (uint32_t k = 0; k < rans[2]; k++) {
        std::vector<uint32_t> vals{{i, j, k}};
        uint32_t rtn = di.denseIndex(vals);
        allidx.insert(rtn);
        auto decoded_vals = di.unpackDenseIndex(rtn);

        if (vals == decoded_vals)
          continue;
        std::cout << "Dense indexing failed @ ";
        parray(vals);
        std::cout << " -> " << rtn << " -> ";
        parray(decoded_vals);
        std::cout << std::endl;
      }
    }
  }

  //check that all values were unique
  assert(allidx.size() == di.getMaxIndex());

  //check that values were sequential (last value==size)
  assert((*allidx.end()) == di.getMaxIndex());

  return 0;
}
