#ifndef CSCBadStrips_h
#define CSCBadStrips_h

#include <vector>

class CSCBadStrips{
 public:
  CSCBadStrips();
  ~CSCBadStrips();
  
  struct BadChamber{
    int chamber_index;
    int pointer;
    int bad_channels;
  };
  struct BadChannel{
    short int layer;
    short int channel;
    short int flag1;
    short int flag2;
    short int flag3;
  };

  int numberOfBadChannels;

  typedef std::vector<BadChamber> BadChamberContainer;
  typedef std::vector<BadChannel> BadChannelContainer;

  BadChamberContainer chambers;
  BadChannelContainer channels;
};

#endif
