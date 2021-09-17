#include <cstdint>
#ifndef RecoLocalMuon_GEMRecHit_GEMCluster_h
#define RecoLocalMuon_GEMRecHit_GEMCluster_h

class GEMCluster {
public:
  GEMCluster();
  GEMCluster(int fs, int ls, int bx);
  ~GEMCluster();

  int firstStrip() const;
  int lastStrip() const;
  int clusterSize() const;
  int bx() const;

  void merge(const GEMCluster& cl);

  bool operator<(const GEMCluster& cl) const;
  bool operator==(const GEMCluster& cl) const;
  bool isAdjacent(const GEMCluster& cl) const;

private:
  uint16_t fstrip;
  uint16_t lstrip;
  int16_t bunchx;
};
#endif
