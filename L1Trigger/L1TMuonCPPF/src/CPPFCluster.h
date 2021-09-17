#include <cstdint>
#ifndef L1Trigger_CPPFCluster_h
#define L1Trigger_CPPFCluster_h
class CPPFCluster {
public:
  CPPFCluster();
  CPPFCluster(int fs, int ls, int bx);
  ~CPPFCluster();

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
  void merge(const CPPFCluster& cl);

  bool operator<(const CPPFCluster& cl) const;
  bool operator==(const CPPFCluster& cl) const;
  bool isAdjacent(const CPPFCluster& cl) const;

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
