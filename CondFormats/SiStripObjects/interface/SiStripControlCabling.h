#ifndef SISTRIPCONTROLCABLING_H
#define SISTRIPCONTROLCABLING_H
#include <map>
#include <vector>
class SiStripControlCabling{
public:
  SiStripControlCabling();
  virtual ~SiStripControlCabling();
  struct RingItem{
    short id;//Ring identifier

    struct I2CItem{
      short channel;//I2C channel
      int DetId; //Module identifier as in offline sw
      //      short apvpair; [0,2] //this info should be added if the apvpair cannot be inferred from the addresses below
      short addressApv1; //address APV1
      short addressApv2; //address APV2
    };

    //map of CCU [0,] to I2C channels
    map<short, vector<I2CItem> > //CCU 

  };
  //map of fecs [0,] to Rings
  std::map<short, std::vector<RingItem> > Fecs;

private:

};
#endif
