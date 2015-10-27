#ifndef __l1microgmtlut_h
#define __l1microgmtlut_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>
#include <vector>

#include "../interface/MicroGMTConfiguration.h"

namespace l1t {
  class MicroGMTLUT {
    public:
      MicroGMTLUT() : m_totalInWidth(0), m_outWidth(0), m_initialized(false) {};
      virtual ~MicroGMTLUT() {};

      // should be implemented in each daughter!
      // This function is the minimum that should be provided
      virtual int lookupPacked(int input) const;

      // populates the m_contents map.
      void initialize();

      int checkedInput(unsigned in, unsigned maxWidth) const;

      // I/O functions
      void save(std::ofstream& output);
      void load(const std::string& inFileName);
      // content to file
      void contentsToStream(std::stringstream& stream);
      void headerToStream(std::stringstream& stream) const;

    protected:
      unsigned m_totalInWidth;
      unsigned m_outWidth;
      std::vector<MicroGMTConfiguration::input_t> m_inputs;
      std::map<int, int> m_contents;
      std::string m_fname;
      bool m_initialized;
  };
}

#endif /* defined(__l1microgmtlut_h) */