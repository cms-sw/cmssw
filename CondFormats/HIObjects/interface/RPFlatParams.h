#ifndef __RPFlatParams_h__
#define __RPFlatParams_h__

#include <vector>
class RPFlatParams{
 public:
  struct EP {
    double x[50];
    double y[50];
    double xSub1[50];
    double ySub1[50];
    double xSub2[50];
    double ySub2[50];
    int RPNameIndx[50];
  };
  RPFlatParams(){}
  virtual ~RPFlatParams(){}
  std::vector<EP> m_table;
};

#endif

