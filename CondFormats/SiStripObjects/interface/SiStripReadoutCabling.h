#ifndef SISTRIPREADOUTCABLING_H
#define SISTRIPREADOUTCABLING_H
#include <map>
#include <vector>
class SiStripReadOutCabling{
public:
  SiStripReadOutCabling();
  virtual ~SiStripReadOutCabling();
  struct Item{
    short fedchannelid;//[0,95]
    int detid; //detid
    short apvpair; //apvpair[0,2]
  };
  //map of fed_id[0,449] to SiStripReadOutCabling::Item
  std::map<short, std::vector<Item> > cablingmap;
  //  typedef std::vector<Item>::const_iterator ItemIterator;  

private:

};
#endif
