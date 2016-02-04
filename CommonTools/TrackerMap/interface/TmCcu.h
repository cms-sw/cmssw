#include <map>
#include <string>


class TmCcu  {
 public:
  TmCcu(int crate,int slot,int ring, int addr);
  ~TmCcu();
  int red,green,blue;
  float value;
  std::string text;
  int count;
  int idex;
  int crate;
  int nmod;//number of modules connected to this ccu
  std::string cmodid;//list of modules connected to this ccu
  int layer;//tracker layer of connected modules
  int mpos;//ccu position in ring
  int getCcuCrate(){int res = (int) (idex/10000000); return res;}
  int getCcuSlot(){int res = (int) (idex/100000); int res1=(int)(res/100);return res - res1*100;}
  int getCcuRing(){int res = (int) (idex%100000); int res1=(int)(res/1000);return res1 ;}
  int getCcuAddr(){int res = (int) (idex%1000); return  res;}
};
