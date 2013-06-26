#ifndef RecoLocalMuon_RPCCluster_h
#define RecoLocalMuon_RPCCluster_h
#include <boost/cstdint.hpp>
class RPCCluster{
 public:
  RPCCluster();
  RPCCluster(int fs,int ls, int bx);
  ~RPCCluster();

  int firstStrip() const;
  int lastStrip() const;
  int clusterSize() const;
  int bx() const;

  void merge(const RPCCluster& cl);

  bool operator<(const RPCCluster& cl) const;
  bool operator==(const RPCCluster& cl) const;
  bool isAdjacent(const RPCCluster& cl) const;

 private:
  uint16_t fstrip;
  uint16_t lstrip;
  int16_t bunchx;
};
#endif
