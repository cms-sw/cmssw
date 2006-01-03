#ifndef Common_CopyPolicy_h
#define Common_CopyPolicy_h

template<typename T>
struct CopyPolicy{
  static const T & clone( const T & t ) {
    return t;
  }
};

#endif
