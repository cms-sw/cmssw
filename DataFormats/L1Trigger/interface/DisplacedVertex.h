#ifndef DataFormats_L1TVertex_DisplacedVertex_h
#define DataFormats_L1TVertex_DisplacedVertex_h
#include <vector>
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include <ap_int.h>

namespace l1t {

  class DisplacedTrueVertex {
  public:
    DisplacedTrueVertex(float d_T, float R_T, float cos_T, float x, float y, float z, float openingAngle, float parentPt)
        : d_T_(d_T), R_T_(R_T), cos_T_(cos_T), x_(x), y_(y), z_(z), openingAngle_(openingAngle), parentPt_(parentPt) {}
    DisplacedTrueVertex() {}
    ~DisplacedTrueVertex() {}
    float d_T() const { return d_T_; }
    float R_T() const { return R_T_; }
    float cos_T() const { return cos_T_; }
    float x() const { return x_; }
    float y() const { return y_; }
    float z() const { return z_; }
    float openingAngle() const { return openingAngle_; }
    float parentPt() const { return parentPt_; }

  private:
    float d_T_;
    float R_T_;
    float cos_T_;
    float x_;
    float y_;
    float z_;
    float openingAngle_;
    float parentPt_;
  };
  typedef std::vector<DisplacedTrueVertex> DisplacedTrueVertexCollection;

  class DisplacedTrackVertex {
    
  public:
    DisplacedTrackVertex(int firstIndexTrk,
                         int secondIndexTrk,
                         int inTraj,
                         float d_T,
                         float R_T,
                         float cos_T,
                         float del_Z,
                         float x,
                         float y,
                         float z,
                         float openingAngle,
                         float parentPt,
                         bool isReal)
        : firstIndexTrk_(firstIndexTrk),
          secondIndexTrk_(secondIndexTrk),
          inTraj_(inTraj),
          d_T_(d_T),
          R_T_(R_T),
          cos_T_(cos_T),
          del_Z_(del_Z),
          x_(x),
          y_(y),
          z_(z),
          openingAngle_(openingAngle),
          parentPt_(parentPt),
          isReal_(isReal) {}
    DisplacedTrackVertex() {}
    ~DisplacedTrackVertex() {}
    void setScore(float score) { score_ = score; }
    float d_T() const { return d_T_; }
    float R_T() const { return R_T_; }
    float cos_T() const { return cos_T_; }
    float x() const { return x_; }
    float y() const { return y_; }
    float z() const { return z_; }
    float openingAngle() const { return openingAngle_; }
    float parentPt() const { return parentPt_; }
    int firstIndexTrk() const { return firstIndexTrk_; }
    int secondIndexTrk() const { return secondIndexTrk_; }
    int inTraj() const { return inTraj_; }
    float del_Z() const { return del_Z_; }
    bool isReal() const { return isReal_; }
    float score() const { return score_; }

  private:
    int firstIndexTrk_;
    int secondIndexTrk_;
    int inTraj_;
    float d_T_;
    float R_T_;
    float cos_T_;
    float del_Z_;
    float x_;
    float y_;
    float z_;
    float openingAngle_;
    float parentPt_;
    bool isReal_;
    float score_;
  };

  typedef std::vector<DisplacedTrackVertex> DisplacedTrackVertexCollection;
}  // namespace l1t

#endif
