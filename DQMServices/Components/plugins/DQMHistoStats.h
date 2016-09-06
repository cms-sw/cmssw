#ifndef DQMHistoStats_H
#define DQMHistoStats_H

struct Dimension{
  int nBin = 0;
  double low = 0, up = 0;
  double mean = 0, meanError = 0; 
  double rms = 0, rmsError = 0; 
  double underflow = 0, overflow = 0;  
};

class HistoEntry {
 public:
  std::string path;

  std::string name;
  const char *type;
  size_t bin_count = 0;
  size_t bin_size = 0;
  size_t extra = 0;
  size_t total = 0;
  double entries = 0;
  int maxBin = 0, minBin = 0;
  double maxValue = 0, minValue = 0;
  int dimNumber = 1;
  Dimension dimX, dimY, dimZ; 

  bool operator<(const HistoEntry &rhs) const { return path < rhs.path; }
};

typedef std::set<HistoEntry> HistoStats;

#endif
