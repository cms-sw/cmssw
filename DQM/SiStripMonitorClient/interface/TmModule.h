#include <map>
#include <string>
using namespace std;


class TmModule  {
 public:
  TmModule(int idc, int iring, int ilayer);
  virtual ~TmModule();
  float posx, posy, posz;
  float length, width, thickness, widthAtHalfLength;
  int red,green,blue;
  float value;
  int count;	
  string text;
  string name;
  int histNumber;
  int getId(){return idModule; }
  int getKey(){return layer*100000+ring*1000+idModule; }
  bool notInUse(){return notused;}
  void setUsed(){notused=false;}
  int idModule;
  int ring;
  int layer;
  unsigned int idex;
  bool notused;

  void setQPointArray(int ar){histNumber = ar;};
  int getQPointArray(){return histNumber;};
};

class SvgModuleMap {
 public:
  static map<const int  , TmModule *> smoduleMap;
 };


class IdModuleMap {
 public:
  static map<const int  , TmModule *> imoduleMap;
 };


