#ifndef Alignment_CommonAlignment_AlignmentUserVariables_h
#define Alignment_CommonAlignment_AlignmentUserVariables_h

/// (Abstract) Base class for alignment algorithm user variables

class AlignmentUserVariables 
{

public:
  virtual ~AlignmentUserVariables() {}
  // derived class must implement clone method
  // (should be simply copy constructor)
  virtual AlignmentUserVariables* clone( void ) const = 0;

};

#endif

