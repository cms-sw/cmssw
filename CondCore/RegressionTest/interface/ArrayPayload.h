#include <string>
#include <vector>
#include <map>
#include <list>
#include <set>
#include <bitset>

struct Param {
  Param();
  Param( int seed );
  int p_i;
  std::string p_s;
  bool operator ==(const Param& rhs) const;
  bool operator !=(const Param& rhs) const;
};

class ArrayPayload {
  public:
  ArrayPayload();
  ArrayPayload( int seed );
  bool operator ==(const ArrayPayload& rhs) const;
  bool operator !=(const ArrayPayload& rhs) const;
  private:
  int m_i;
  // c-arrays
  int m_ai0[4];
  int m_ai1[112];
  int m_ai2[3][2];
  int m_ai3[2][80];
  std::string m_as0[4];
  std::string m_as1[112];
  Param m_ap0[4];
  Param m_ap1[112];
  // stl containers
  std::pair<unsigned int,unsigned int> m_p0;
  std::pair<int,std::string> m_p1;
  std::pair<int,Param> m_p2;
  std::vector<int> m_vec0;
  std::vector<std::string> m_vec1;
  std::map<unsigned int,unsigned int> m_map0;
  std::map<std::string,std::string> m_map1;
  std::list<int> m_list;
  std::set<std::string> m_set;
  std::bitset<128> m_bitset;
  std::vector<Param> m_vec2;
  std::map<int,Param> m_map2;
  // blob streaming
  std::vector<Param> m_vec3;
};
