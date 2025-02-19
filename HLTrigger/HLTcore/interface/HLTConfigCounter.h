#ifndef HLTConfigCounter_H
#define HLTConfigCounter_H

class   HLTConfigCounter {

  public:

  HLTConfigCounter(): count_(0) {}

    long increment() { count_++; return count_; }
    long decrement() { count_--; return count_; }

  private:

    long count_;

};

#endif
