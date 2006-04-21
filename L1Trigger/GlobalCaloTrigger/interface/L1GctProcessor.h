
#ifndef L1GCTPROCESSOR_H_
#define L1GCTPROCESSOR_H_


class L1GctProcessor {

 public:

  L1GctProcessor() {};
  virtual ~L1GctProcessor() {};
  ///
  /// clear internal buffers
  virtual void reset() = 0;
  ///
  /// set the input buffers
  virtual void fetchInput() = 0;
  /// 
  /// process the data and set outputs
  virtual void process() = 0;

};

#endif
