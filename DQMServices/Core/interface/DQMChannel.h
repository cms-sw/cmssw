#ifndef DQMSERVICES_CORE_DQM_CHANNEL_H
#define DQMSERVICES_CORE_DQM_CHANNEL_H

struct DQMChannel {
  int binx;       //< bin # in x-axis (or bin # for 1D histogram)
  int biny;       //< bin # in y-axis (for 2D or 3D histograms)
  int binz;       //< bin # in z-axis (for 3D histograms)
  float content;  //< bin content
  float RMS;      //< RMS of bin content

  int getBin() { return getBinX(); }
  int getBinX() { return binx; }
  int getBinY() { return biny; }
  int getBinZ() { return binz; }
  float getContents() { return content; }
  float getRMS() { return RMS; }

  DQMChannel(int bx, int by, int bz, float data, float rms) {
    binx = bx;
    biny = by;
    binz = bz;
    content = data;
    RMS = rms;
  }

  DQMChannel() {
    binx = 0;
    biny = 0;
    binz = 0;
    content = 0;
    RMS = 0;
  }
};

#endif  // DQMSERVICES_CORE_DQM_CHANNEL_H
