#ifndef CSCobject_h
#define CSCobject_h

#include <vector>
#include <map>

class CSCobject{
 public:
  CSCobject();
  ~CSCobject();

  std::map< int, std::vector<std::vector<float> > > obj;
};

#endif
