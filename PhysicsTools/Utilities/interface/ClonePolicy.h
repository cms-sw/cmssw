#ifndef Candidate_ClonePolicy_h
#define Candidate_ClonePolicy_h

template<typename T>
struct ClonePolicy{
  static T * clone( const T & t ) {
    return t.clone();
  }
};

#endif
