#ifndef CSCDeadNoisy_h
#define CSCDeadNoisy_h

#include <vector>

class CSCDeadNoisy{
 public:
  CSCDeadNoisy();
  ~CSCDeadNoisy();
  
  struct BadChambers{
    int chamber_index;
    int pointer;
    int bad_channels;
  };
  struct Item{
    short int layer;
    short int channel;
    short int flag1;
    short int flag2;
    short int flag3;
  };

  typedef std::vector<BadChambers> BadChambersContainer;
  typedef std::vector<Item> BadChannelsContainer;

  BadChambersContainer chambers;
  BadChannelsContainer channels;
};

#endif
