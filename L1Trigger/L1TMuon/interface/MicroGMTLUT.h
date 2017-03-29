#ifndef __l1microgmtlut_h
#define __l1microgmtlut_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>
#include <vector>

#include "CondFormats/L1TObjects/interface/LUT.h"
#include "../interface/MicroGMTConfiguration.h"

namespace l1t {
  class MicroGMTLUT : public LUT {
    public:
      MicroGMTLUT() : m_totalInWidth(0), m_outWidth(0), m_initialized(false) {};
      MicroGMTLUT(l1t::LUT* lut);
      virtual ~MicroGMTLUT() {};

      // should be implemented in each daughter!
      // This function is the minimum that should be provided
      virtual int lookupPacked(int input) const;

      // populates the m_contents map.
      void initialize();

      int checkedInput(unsigned in, unsigned maxWidth) const;

      // I/O functions
      void save(std::ofstream& output);
      int load(const std::string& inFileName);

    protected:
      unsigned m_totalInWidth;
      unsigned m_outWidth;
      std::vector<MicroGMTConfiguration::input_t> m_inputs;
      bool m_initialized;
  };
}

#endif /* defined(__l1microgmtlut_h) */
