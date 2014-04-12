#ifndef PtSorter_h
#define PtSorter_h

class PtSorter {
public:
  template <class T> bool operator() ( const T& a, const T& b ) {
    return ( a.pt() > b.pt() );
  }
};





#endif
