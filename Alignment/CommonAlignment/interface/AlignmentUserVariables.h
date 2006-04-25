
#ifndef Alignment_CommonAlignment_ALIGNMENTUSERVARIABLES_H
#define Alignment_CommonAlignment_ALIGNMENTUSERVARIABLES_H

/** (Abstract) Base class for alignment algorithm user variables */

class AlignmentUserVariables {

  public:

  // derived class must implement clone method
  // (should be simply copy constructor)
  virtual AlignmentUserVariables* clone(void) const =0;

};

#endif

