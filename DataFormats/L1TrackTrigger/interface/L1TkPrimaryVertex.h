#ifndef L1TkTrigger_L1TkPrimaryVertex_h
#define L1TkTrigger_L1TkPrimaryVertex_h


// Nov 12, 2013
// First version of a class for L1-zvertex

class L1TkPrimaryVertex {

 public:

 L1TkPrimaryVertex() : zvertex(-999), sum(-999) {}

 ~L1TkPrimaryVertex() { }

 L1TkPrimaryVertex(float z, float s) : zvertex(z), sum(s) { }


    float getZvertex() const { return zvertex ; } 
    float getSum() const { return sum ; }

 private:
   float zvertex;
   float sum;

};

#include <vector>

typedef std::vector<L1TkPrimaryVertex> L1TkPrimaryVertexCollection ;


#endif
