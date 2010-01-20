#include <map>
#include <string>

using namespace std;

class TmModule;

class TmPsu  {
 public:
  TmPsu(int dcs,int branch, int rack, int crate,int board);
  ~TmPsu();
  int id;
  int idex;
  string psId;//ex: TECminus_5_6_4_2_3...
  int getPsuDcs(){int res = (int) (id%100000); return  (int)(id - res)/100000;}
  int getPsuBranch(){int res1 = (int)(id%100000); int res = (int)(res1%1000); return (int) (res1 -res)/1000;}
  int getPsuRack(){int res = (int) (idex%1000); return (idex - res)/1000;}
  int getPsuCrate(){int res1 = (int) (idex%1000); int res=(int)(res1%100);return (int)(res1 - res)/100;}
  int getPsuBoard(){int res2 = (int) (idex%1000); int res1=(int)(res2%100);return res1;}
    
  
  int red,green,blue;
  int redHV2,greenHV2,blueHV2;
  int redHV3,greenHV3,blueHV3;
  float value;
  float valueHV3;
  float valueHV2;
  int count;
  int countHV2;
  int countHV3;
  int nmod;
  int nmodHV2;
  int nmodHV3;
  string cmodid_LV;//list of modules connected to the LV channels of this psu
  string cmodid_HV2;
  string cmodid_HV3;
  string text;
  string textHV2; 
  string textHV3;
 
 
  };
  
  
   
   
