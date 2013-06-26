#ifndef Common_MergeableCounter_h
#define Common_MergeableCounter_h

namespace edm {

  struct MergeableCounter {
    ~MergeableCounter() {}
    bool mergeProduct(MergeableCounter const & newThing);
    int value;
  };

}

#endif
