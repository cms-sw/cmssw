
#ifndef L1GCTPROCESSOR_H_
#define L1GCTPROCESSOR_H_


class L1GctProcessor {

 public:

  L1GctProcessor() {};
  virtual ~L1GctProcessor() {};

  virtual void reset() = 0;

  virtual void fetchInput() = 0;

  virtual void process() = 0;

};

#endif
