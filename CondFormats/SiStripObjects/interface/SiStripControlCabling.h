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
      //      short addressApv1; //address APV1
      //      short addressApv2; //address APV2
      std::vector<short> apvAdds;
    };

    //map of CCU [0,] to I2C channels
    //FIXME
    std::map<short, std::vector<I2CItem> > ccuMap;
  };
  //map of fecs [0,] to Rings
  std::map<short, std::vector<RingItem> > Fecs;

//   typedef std::map<short, std::vector<RingItem> > ItemMap;
//   typedef std::vector<RingItem>::iterator ItemIterator;
//   typedef std::map<short, std::vector<RingItem::I2CItem> >::iterator CCUIterator;
//   typedef std::vector<RingItem::I2CItem>::iterator I2CIterator;

private:

};
#endif
