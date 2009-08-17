#ifndef Grid1D_H
#define Grid1D_H
#include <algorithm>

class Grid1D {
public:

  typedef double Scalar;

  Grid1D() {}

  Grid1D( Scalar lower, Scalar upper, int nodes) : 
    lower_(lower), upper_(upper), nodes_(nodes) {
    step_ = (upper - lower) / (nodes-1);
  }


  Scalar step() const {return step_;}
  Scalar lower() const {return lower_;}
  Scalar upper() const {return upper_;}
  int nodes() const {return nodes_;}
  int cells() const {return nodes()-1;}

  Scalar node( int i) const { return i*step() + lower();}

  Scalar closestNode( Scalar a) const {
    Scalar b = (a-lower())/step();
    Scalar c = floor(b);
    Scalar tmp = (b-c < 0.5) ? std::max(c,0.) : std::min(c+1.,static_cast<Scalar>(nodes()-1));
    return tmp*step()+lower();
  }

  /// returns valid index, or -1 if the value is outside range +/- one cell.
  int index( Scalar a) const {
    int ind = static_cast<int>((a-lower())/step());
    // FIXME: this causes an exception to be thrown later. Should be tested
    // more carefully before release
    //  if (ind < -1 || ind > cells()) {
    //     std::cout << "**** ind = " << ind << " cells: " << cells() << std::endl;
    //    return -1;
    //  }
    return std::max(0, std::min( cells()-1, ind));
  }

private:

  Scalar step_;
  Scalar lower_;
  Scalar upper_;
  int    nodes_;

};

#endif
