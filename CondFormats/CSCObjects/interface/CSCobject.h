#ifndef CSCobject_h
#define CSCobject_h

#include <vector>
#include <map>
using namespace std;

class CSCobject{
 public:
  CSCobject();
  ~CSCobject();

  map< int,vector<vector<float> > > obj;
};

#endif
