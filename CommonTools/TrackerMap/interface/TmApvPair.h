#include <map>
#include <string>
using namespace std;

class TmModule;

class TmApvPair  {
 public:
  TmApvPair(int ident);
  ~TmApvPair();
  int red,green,blue;
  float value;
  int count;
  string text;
  string name;
  int histNumber;
  int idex;//Fed and position in fed
  TmModule * mod;
  int mpos;//ApvPair position in module
  int getFedCh(){int res = (int) (idex/1000); return idex - res*1000;}
  int getFedId(){int res = (int) (idex/1000); return  res;}
};

class SvgApvPair {
 public:
  static map<const int  , TmApvPair*> apvMap;
};
class ModApvPair {
 public:
  static multimap<const int  , TmApvPair*> apvModuleMap;
};
class SvgFed {
 public:
  static map<const int  , int> fedMap;
};
