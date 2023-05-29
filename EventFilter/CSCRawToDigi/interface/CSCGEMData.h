#ifndef EventFilter_CSCRawToDigi_CSCGEMData_h
#define EventFilter_CSCRawToDigi_CSCGEMData_h

#include <vector>
#ifndef LOCAL_UNPACK
#include <atomic>
#endif

class GEMPadDigiCluster;

class CSCGEMData {
public:
  /// default constructor
  CSCGEMData(int ntbins = 12, int gem_fibers_mask = 0xf);
  // length is in 16-bit words
  CSCGEMData(const unsigned short *c04buf, int length, int gem_fibers_mask = 0xf);

  // std::vector<int> BXN() const;
  std::vector<GEMPadDigiCluster> digis(int gem_chamber) const;
  std::vector<GEMPadDigiCluster> etaDigis(int gem_chamber, int eta, int correctionToALCTbx) const;
  int sizeInWords() const { return size_; }
  int numGEMs() const {
    return 2;  // !!! TODO actual number of GEM chambers in readout
  }
  int gemFibersMask() const { return gems_enabled_; }
  int numGEMEnabledFibers() const { return ngems_; }
  int nTbins() const { return ntbins_; }
  void print() const;
  bool check() const { return ((theData[0] == 0x6C04) && (theData[size_ - 1] == 0x6D04)); }

  /// turns on the debug flag for this class
  static void setDebug(bool debugValue) { debug = debugValue; }

  /// Add and pack GEMPadDigiCluster digis
  void addEtaPadCluster(const GEMPadDigiCluster &digi, int gem_chamber, int eta_roll);

  unsigned short *data() { return theData; }

private:
  int getPartitionNumber(int address, int nPads) const;
  int getPartitionStripNumber(int address, int nPads, int etaPart) const;

#ifdef LOCAL_UNPACK
  static bool debug;
#else
  static std::atomic<bool> debug;
#endif

  int ntbins_;
  int gems_enabled_;
  int ngems_;
  int size_;
  unsigned short theData[8 * 2 * 32 + 2];
};

#endif
