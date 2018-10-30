#ifndef Common_MergeableCounter_h
#define Common_MergeableCounter_h

namespace edm {

  struct MergeableCounter {
    MergeableCounter() : value() {}
    ~MergeableCounter() {}
    bool mergeProduct(MergeableCounter const & newThing);
    void swap(MergeableCounter& iOther);
    int value;
  };

}

#endif
