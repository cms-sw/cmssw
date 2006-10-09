#ifndef Grid1D_H
#define Grid1D_H

template <class T>
class Grid1D {
public:

  typedef T Scalar;

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
    Scalar tmp = (b-c < 0.5) ? max(c,0.) : min(c+1.,static_cast<Scalar>(nodes()-1));
    return tmp*step()+lower();
  }

  int index( Scalar a) const {
    return max(0, min( cells()-1, static_cast<int>((a-lower())/step())));
  }

private:

  Scalar step_;
  Scalar lower_;
  Scalar upper_;
  int    nodes_;

};

#endif
