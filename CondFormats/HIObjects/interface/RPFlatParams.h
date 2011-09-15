#ifndef __RPFlatParams_h__
#define __RPFlatParams_h__

#include <vector>
class RPFlatParams{
 public:
  struct EP {
    float x[50];
    float y[50];
    float xSub1[50];
    float ySub1[50];
    float xSub2[50];
    float ySub2[50];
    int RPNameIndx[50];
  };
  RPFlatParams(){}
  virtual ~RPFlatParams(){}
  std::vector<EP> m_table;
};

#endif

