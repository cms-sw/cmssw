#ifndef PayObj_H
#define PayObj_H

#include <vector>
class PayObj {
  public:
    explicit PayObj(unsigned int i):data(){
      data.reserve(i);
      for(unsigned int j=0; j<i; j++){
	data.push_back(j);
      }
    }
    PayObj():data(){}
    virtual ~PayObj(){}
    bool operator==(const PayObj& rhs) const {
      return data==rhs.data;
    }
    bool operator!=(const PayObj& rhs) const {
      return data!=rhs.data;
    }
    std::vector<int> data;
};
#endif
