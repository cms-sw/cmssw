#ifndef __UETable_h__
#define __UETable_h__

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>

class UETable{
 public:
  UETable(){};
  float getUE(int i){return values[i];}
  unsigned int getNp(int i){return np[i];}
  unsigned int getNi0(int i){return ni0[i];}
  unsigned int getNi1(int i){return ni1[i];}
  unsigned int getNi2(int i){return ni2[i];}
  float getEtaEdge(int i){return edgeEta[i];}


  void pushUE(float v){values.push_back(v);}
  void pushNp(unsigned int v){np.push_back(v);}
  void pushNi0(unsigned int v){ni0.push_back(v);}
  void pushNi1(unsigned int v){ni1.push_back(v);}
  void pushNi2(unsigned int v){ni2.push_back(v);}
  void pushEtaEdge(float v){edgeEta.push_back(v);}

  std::vector<float> values;
  std::vector<unsigned int> np;
  std::vector<unsigned int> ni0;
  std::vector<unsigned int> ni1;
  std::vector<unsigned int> ni2;
  std::vector<float> edgeEta;

  COND_SERIALIZABLE;
};

#endif
