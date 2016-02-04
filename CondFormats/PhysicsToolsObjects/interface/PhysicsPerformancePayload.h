#ifndef PhysicsPerformancePayload_h
#define PhysicsPerformancePayload_h
//
// File: CondFormats/PhysicsPerformancePayload/interface/PhysicsPerformancePayload.h
//
// Zongru Wan, Kansas State University
//

#include <vector>

class PhysicsPerformancePayload
{
 public:
  PhysicsPerformancePayload(){}
  PhysicsPerformancePayload(int stride, std::vector<float> table);
  int stride(){return stride_;}

  typedef std::vector<float> Row;
  
  Row getRow(int n) const;
  int nRows() const ;
  
  std::vector<float> payload() const {return table_;}


  virtual ~PhysicsPerformancePayload() {}

  
 protected:
  int stride_;
  std::vector<float> table_;
};

#endif

