#ifndef Common_NewPolicy_h
#define Common_NewPolicy_h

template<typename T>
struct NewPolicy{
  static T * clone( const T & t ) {
    return new T( t );
  }
};

#endif
