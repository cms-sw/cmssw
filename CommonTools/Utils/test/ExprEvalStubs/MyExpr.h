#include <vector>
#include <algorithm>
#include <numeric>
#include<memory>

#include "Cand.h"

struct MyExpr {
  using Coll = std::vector<std::unique_ptr<Cand const>>; 
  using Res = std::vector<bool>;
  virtual void eval(Coll const &, Res &)=0;
  
};
