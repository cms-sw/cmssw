//////////////////////////////////////////////////////////
//
// Class to print effs
//
//////////////////////////////////////////////////////////

#ifndef OHltEffPrinter_h
#define OHltEffPrinter_h

#include <vector>
#include "OHltMenu.h"
#include "OHltConfig.h"
#include "OHltTree.h"

using namespace std;

class OHltEffPrinter {
 public:
  
  OHltEffPrinter(){};
  virtual ~OHltEffPrinter(){};

  void SetupAll(vector<float> Eff,vector<float> EffErr,vector<float> spureEff,
		vector<float> spureEffErr,vector<float> pureEff,
		vector<float> pureEffErr,vector< vector<float> >coMa, float DenEff);

  void printEffASCII(OHltConfig *cfg,OHltMenu *menu);

  vector<float> Eff;
  vector<float> EffErr;
  vector<float> spureEff;
  vector<float> spureEffErr;
  vector<float> pureEff;
  vector<float> pureEffErr;
  vector< vector<float> >coMa;
  float DenEff;
  
};

#endif
