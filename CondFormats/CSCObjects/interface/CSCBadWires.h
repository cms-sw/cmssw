#ifndef CSCBadWires_h
#define CSCBadWires_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

class CSCBadWires {
public:
  CSCBadWires();
  ~CSCBadWires();

  struct BadChamber {
    int chamber_index;
    int pointer;
    int bad_channels;

    COND_SERIALIZABLE;
  };
  struct BadChannel {
    short int layer;
    short int channel;
    short int flag1;
    short int flag2;
    short int flag3;

    COND_SERIALIZABLE;
  };

  int numberOfBadChannels;

  typedef std::vector<BadChamber> BadChamberContainer;
  typedef std::vector<BadChannel> BadChannelContainer;

  BadChamberContainer chambers;
  BadChannelContainer channels;

  COND_SERIALIZABLE;
};

#endif
