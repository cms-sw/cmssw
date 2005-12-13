#ifndef Candidate_CopyPolicy_h
#define Candidate_CopyPolicy_h

template<typename T>
struct CopyPolicy{
  static const T & clone( const T & t ) {
    return t;
  }
};

#endif
