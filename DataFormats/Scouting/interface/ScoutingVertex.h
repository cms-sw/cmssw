#ifndef DataFormats_ScoutingVertex_h
#define DataFormats_ScoutingVertex_h

#include <vector>

//class for holding vertex information, for use in data scouting 
//IMPORTANT: the content of this class should be changed only in backwards compatible ways!
class ScoutingVertex
{
 public:
  //constructor with values for all data fields
 ScoutingVertex(float x, float y, float z, float zError, float xError, float yError, int tracksSize, float chi2, float ndof):
  x_(x), y_(y), z_(z), zError_(zError), xError_(xError), yError_(yError), tracksSize_(tracksSize), chi2_(chi2), ndof_(ndof) {}
  //default constructor
 ScoutingVertex(): x_(0), y_(0), z_(0), zError_(0), xError_(0), yError_(0), tracksSize_(0), chi2_(0), ndof_(0) {}
  
  //accessor functions
  float x() const { return x_; }
  float y() const { return y_; }
  float z() const { return z_; }
  float zError() const { return zError_; }
  float xError() const { return xError_; }
  float yError() const { return yError_; }
  int tracksSize() const { return tracksSize_; }
  float chi2() const { return chi2_; }
  float ndof() const { return ndof_;}

 private:
  float x_;
  float y_;
  float z_;
  float zError_;
  float xError_;
  float yError_;
  int tracksSize_; 
  float chi2_;
  float ndof_;
};

typedef std::vector<ScoutingVertex> ScoutingVertexCollection;

#endif
