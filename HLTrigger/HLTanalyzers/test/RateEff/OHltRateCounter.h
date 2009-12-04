//////////////////////////////////////////////////////////
//
// Class to store  and process rate counts
//
//////////////////////////////////////////////////////////

#ifndef OHltRateCounter_h
#define OHltRateCounter_h

#include <vector>
#include <libconfig.h++>
#include <TMath.h>

using namespace std;
using namespace libconfig;

class OHltRateCounter {

 public:

  OHltRateCounter(unsigned int size);
  virtual ~OHltRateCounter(){};

  bool isNewRunLS(int Run,int LumiBlock);
  void addRunLS(int Run,int LumiBlock);
  void incrRunLSCount(int Run,int LumiBlock,int iTrig, int incr=1);
  void incrRunLSTotCount(int Run,int LumiBlock, int incr=1);
  int getIDofRunLSCounter(int Run,int LumiBlock);

  // Helper functions
  static inline float eff(int a, int b){ 
    if (b==0.){return -1.;}
    float af = float(a),bf = float(b),effi = af/bf;
    return effi;
  }
  static inline float effErr(int a, int b){
    if (b==0.){return -1.;}
    float af = float(a),bf = float(b),r = af/bf;
    float unc = sqrt(af + (r*r*bf) )/bf;
    return unc;
  }
  static inline float effErrb(int a, int b){
    if (b==0.){return -1.;}
    float af = float(a),bf = float(b),r = af/bf;
    float unc = sqrt(af - (r*r*bf) )/bf;
    return unc;
  }
  static inline float eff(float a, float b){ 
    if (b==0.){return -1.;}
    float af = float(a),bf = float(b),effi = af/bf;
    return effi;
  }
  static inline float effErr(float a, float b){
    if (b==0.){return -1.;}
    float af = float(a),bf = float(b),r = af/bf;
    float unc = sqrt(af + (r*r*bf) )/bf;
    return unc;
  }
  static inline float effErrb(float a, float b){
    if (b==0.){return -1.;}
    float af = float(a),bf = float(b),r = af/bf;
    float unc = sqrt(af - (r*r*bf) )/bf;
    return unc;
  }
  
  
  // Data
  vector<int> iCount;
  vector<int> sPureCount;
  vector<int> pureCount;
  vector< vector<int> > overlapCount;
  vector<int> prescaleCount;

  vector< vector<int> > perLumiSectionCount;
  vector<int> perLumiSectionTotCount;
  vector<int> runID;
  vector<int> lumiSection;

};
#endif
