#include "CondCore/RegressionTest/interface/PrimitivePayload.h"
#include "CondCore/RegressionTest/interface/ArrayPayload.h"

struct Data {
  Data();
  Data( int seed );
  int m_i;
  std::string m_s;
  std::vector<int> m_a; 
  bool operator ==(const Data& rhs) const;
  bool operator !=(const Data& rhs) const;
};

class RegressionTestPayload : public PrimitivePayload, public ArrayPayload {
public:
  RegressionTestPayload();
  RegressionTestPayload( int seed );
  typedef Data T_Data;
  bool operator ==(const RegressionTestPayload& rhs) const;
  bool operator !=(const RegressionTestPayload& rhs) const;
public:
  int m_i;
  Data m_data0;
  T_Data m_data1;

};

