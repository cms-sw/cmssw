#ifndef Candidate_NewPolicy_h
#define Candidate_NewPolicy_h

template<typename T>
struct NewPolicy{
  static T * clone( const T & t ) {
    return new T( t );
  }
};

#endif
