#ifndef CommonTools_Utils_RefSelector_h
#define CommonTools_Utils_RefSelector_h

template<typename S>
struct RefSelector {
  RefSelector( const S & sel ) : sel_( sel ) { }
  template<typename Ref>
  bool operator()( const Ref & ref ) const {
    return sel_( * ref );
  }
private:
  S sel_;
};

#endif
