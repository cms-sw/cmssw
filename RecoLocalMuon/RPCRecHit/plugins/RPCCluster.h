#include <cstdint>
#ifndef RecoLocalMuon_RPCCluster_h
#define RecoLocalMuon_RPCCluster_h
class RPCCluster {
public:
  RPCCluster();
  RPCCluster(int fs, int ls, int bx);
  ~RPCCluster();

  int firstStrip() const;
  int lastStrip() const;
  int clusterSize() const;
  int bx() const;

  bool hasTime() const;
  float time() const;
  float timeRMS() const;

  bool hasY() const;
  float y() const;
  float yRMS() const;

  void addTime(const float time);
  void addY(const float y);
  void merge(const RPCCluster& cl);

  bool operator<(const RPCCluster& cl) const;
  bool operator==(const RPCCluster& cl) const;
  bool isAdjacent(const RPCCluster& cl) const;

private:
  uint16_t fstrip;
  uint16_t lstrip;
  int16_t bunchx;

  float sumTime, sumTime2;
  uint16_t nTime;

  float sumY, sumY2;
  uint16_t nY;
};
#endif
