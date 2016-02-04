#include <map>
#include <string>

class TmModule;

class TmApvPair  {
 public:
  TmApvPair(int ident,int crate);
  ~TmApvPair();
  int red,green,blue;
  float value;
  std::string text;
  int count;
  int idex;//Fed and position in fed
  int crate;
  TmModule * mod;
  int mpos;//ApvPair position in module
  int getFedCh(){int res = (int) (idex/1000); return idex - res*1000;}
  int getFedId(){int res = (int) (idex/1000); return  res;}
};
