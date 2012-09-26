#ifndef HLTConfigCounter_H
#define HLTConfigCounter_H

class   HLTConfigCounter {

  public:

  HLTConfigCounter(): count_(0) {}

    unsigned long increment() { count_++; return count_; }
    unsigned long decrement() { count_--; return count_; }

  private:

    unsigned long count_;

};

#endif
