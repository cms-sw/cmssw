#ifndef DQM_OBJECTS_H
#define DQM_OBJECTS_H

#include <string>
#include <cstdint>

class TObject;

// These are duplicated from other parts of DQM, to avoid a dependency.
static const uint32_t DQM_PROP_TYPE_MASK = 0x000000ff;
static const uint32_t DQM_PROP_TYPE_SCALAR = 0x0000000f;
static const uint32_t DQM_PROP_REPORT_ERROR = 0x00000100;
static const uint32_t DQM_PROP_REPORT_WARN = 0x00000200;
static const uint32_t DQM_PROP_REPORT_OTHER = 0x00000400;
static const uint32_t DQM_PROP_REPORT_ALARM = (DQM_PROP_REPORT_ERROR | DQM_PROP_REPORT_WARN | DQM_PROP_REPORT_OTHER);

struct VisDQMRenderInfo {
  std::string drawOptions;
  bool blacklisted;
};

struct VisDQMAxisInfo {
  std::string type;
  double min;
  double max;
};

enum VisDQMReference { DQM_REF_OBJECT, DQM_REF_OVERLAY, DQM_REF_OVERLAY_RATIO, DQM_REF_STACKED };

enum VisDQMStripTrend {
  DQM_TREND_OBJECT,
  DQM_TREND_LS_VALUE,
  DQM_TREND_NUM_ENTRIES,
  DQM_TREND_NUM_BINS,
  DQM_TREND_NUM_BYTES,
  DQM_TREND_X_MEAN,
  DQM_TREND_X_MIN,
  DQM_TREND_X_MAX,
  DQM_TREND_X_MEAN_RMS,
  DQM_TREND_X_MEAN_MIN_MAX,
  DQM_TREND_X_NUM_BINS,
  DQM_TREND_Y_MEAN,
  DQM_TREND_Y_MIN,
  DQM_TREND_Y_MAX,
  DQM_TREND_Y_MEAN_RMS,
  DQM_TREND_Y_MEAN_MIN_MAX,
  DQM_TREND_Y_NUM_BINS,
  DQM_TREND_Z_MEAN,
  DQM_TREND_Z_MIN,
  DQM_TREND_Z_MAX,
  DQM_TREND_Z_MEAN_RMS,
  DQM_TREND_Z_MEAN_MIN_MAX,
  DQM_TREND_Z_NUM_BINS
};

struct VisDQMImgInfo {
  std::string imgspec;
  std::string drawOptions;
  std::string reflabel1;
  std::string reflabel2;
  std::string reflabel3;
  std::string reflabel4;
  VisDQMReference reference;
  VisDQMStripTrend trend;
  int width;
  int height;
  int showstats;
  int showerrbars;
  double ktest;
  VisDQMAxisInfo xaxis;
  VisDQMAxisInfo yaxis;
  VisDQMAxisInfo zaxis;
  std::string refnorm;
  std::string json;
};

struct VisDQMObject {
  uint32_t flags;
  uint64_t version;
  std::string dirname;
  std::string objname;
  uint32_t tag;  // Unused.
  std::string name;
  TObject *object;
  VisDQMImgInfo image;
};

#endif  // DQM_OBJECTS_H
