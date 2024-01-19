#ifndef DataFormats_Run3ScoutingVertex_h
#define DataFormats_Run3ScoutingVertex_h

#include <vector>

//class for holding vertex information, for use in data scouting
//IMPORTANT: the content of this class should be changed only in backwards compatible ways!
class Run3ScoutingVertex {
public:
  //constructor with values for all data fields
  Run3ScoutingVertex(float x,
                     float y,
                     float z,
                     float xError,
                     float yError,
                     float zError,
                     float xyCov,
                     float xzCov,
                     float yzCov,
                     int tracksSize,
                     float chi2,
                     int ndof,
                     bool isValidVtx)
      : x_(x),
        y_(y),
        z_(z),
        zError_(zError),
        xError_(xError),
        yError_(yError),
        xyCov_(xyCov),
        xzCov_(xzCov),
        yzCov_(yzCov),
        tracksSize_(tracksSize),
        chi2_(chi2),
        ndof_(ndof),
        isValidVtx_(isValidVtx) {}
  //default constructor
  Run3ScoutingVertex()
      : x_(0),
        y_(0),
        z_(0),
        zError_(0),
        xError_(0),
        yError_(0),
        xyCov_(0),
        xzCov_(0),
        yzCov_(0),
        tracksSize_(0),
        chi2_(0),
        ndof_(0),
        isValidVtx_(false) {}

  //accessor functions
  float x() const { return x_; }
  float y() const { return y_; }
  float z() const { return z_; }
  float xError() const { return xError_; }
  float yError() const { return yError_; }
  float zError() const { return zError_; }
  float xyCov() const { return xyCov_; }
  float xzCov() const { return xyCov_; }
  float yzCov() const { return yzCov_; }
  int tracksSize() const { return tracksSize_; }
  float chi2() const { return chi2_; }
  int ndof() const { return ndof_; }
  bool isValidVtx() const { return isValidVtx_; }

private:
  float x_;
  float y_;
  float z_;
  float zError_;
  float xError_;
  float yError_;
  float xyCov_;
  float xzCov_;
  float yzCov_;
  int tracksSize_;
  float chi2_;
  int ndof_;
  bool isValidVtx_;
};

typedef std::vector<Run3ScoutingVertex> Run3ScoutingVertexCollection;

#endif
