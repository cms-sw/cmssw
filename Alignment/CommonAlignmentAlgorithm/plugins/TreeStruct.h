#ifndef Alignment_CommonAlignmentAlgorithm_plugins_TreeStruct_h
#define Alignment_CommonAlignmentAlgorithm_plugins_TreeStruct_h

/// structure to store algorithm results in a TTree

struct TreeStruct 
{
  TreeStruct() : delta(0.f), error(0.f), paramIndex(0) {}
  TreeStruct(int ind) : delta(0.f), error(0.f), paramIndex(ind) {}
  TreeStruct(float del, float err, int ind) : delta(del), error(err), paramIndex(ind) {}
  
  float delta;     /// parameter from alignment algorithm (change wrt. start)
  float error;     /// error from alignment algorithm
  int   paramIndex;/// internal param. index (same index => same delta)
  /// List of leaves to pass as 3rd argument to TTree::Branch(...) if 2nd argument
  /// is a pointer to TreeStruct - keep in synch with data members above!
  static const char* LeafList() {return "delta/F:error/F:paramIndex/I";}
};

#endif
