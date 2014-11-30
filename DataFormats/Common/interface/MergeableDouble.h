#ifndef Common_MergeableDouble_h
#define Common_MergeableDouble_h

namespace edm {

  struct MergeableDouble {
    ~MergeableDouble() {}
    bool mergeProduct(MergeableDouble const & newThing);
    double value;
  };

}

#endif
