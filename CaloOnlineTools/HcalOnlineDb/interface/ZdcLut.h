#ifndef ZdcLut_h
#define ZdcLut_h

/**

   \class ZdcLut
   \brief Generation of ZDC Lookup tables and associate helper methods
   \brief Adopted to CMSSW HCAL LUT manager specs
   \brief by Gena Kukartsev, Brown University, Dec 08, 2009
   \author Elijah Dunn

*/

#include <iostream>
#include <vector>
#include <string>

struct ZDC_channels {
  std::vector<int> LUT;
};

struct ZDC_fibers {
  std::vector<ZDC_channels> channel;
};

struct ZDC_sides {
  std::vector<ZDC_fibers> fiber;
};

class ZdcLut {
public:
  ZdcLut();
  ~ZdcLut();

  //get_lut returns a specific lut based on side, fiber, and fiber_channel
  //std::vector <int> get_lut(int side_num, int fiber_num, int channel_num){ return side[side_num].fiber[fiber_num].channel[channel_num].LUT; }
  std::vector<int> get_lut(int emap_side, int emap_htr_fiber, int emap_fi_ch);

  std::vector<int> get_lut(std::string zdc_section, int zdc_side, int zdc_channel);

  int simple_loop(void);

private:
  //variable
  std::vector<ZDC_sides> side;
};

#endif
