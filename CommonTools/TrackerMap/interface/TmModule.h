#include <map>
#include <string>


class TmModule  {
 public:
  TmModule(int idc, int iring, int ilayer);
  virtual ~TmModule();
  float posx, posy, posz;
  float length, width, thickness, widthAtHalfLength;
  int red,green,blue;
  float value;
  int count;	
  std::string text;
  std::string name;
  std::string capvids;
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



