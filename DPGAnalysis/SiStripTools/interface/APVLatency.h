#ifndef DPGAnalysis_SiStripTools_APVLatency_H
#define DPGAnalysis_SiStripTools_APVLatency_H

class APVLatency {

 public:

  APVLatency(): _latency(-1) {};
  ~APVLatency() {};

  void put(const int latency) {_latency = latency;}
  const int get() const {return _latency;}

 private:

  int _latency;

};

#endif // DPGAnalysis_SiStripTools_APVLatency_H
