#ifndef MuonReco_MuonMETCorrectionData_h
#define MuonReco_MuonMETCorrectionData_h

#include <cmath>

namespace reco {
  class MuonMETCorrectionData {
  public:
    enum Type {
      NotUsed = 0,
      CombinedTrackUsed = 1,
      GlobalTrackUsed = 1,
      InnerTrackUsed = 2,
      TrackUsed = 2,
      OuterTrackUsed = 3,
      StandAloneTrackUsed = 3,
      TreatedAsPion = 4,
      MuonP4V4QUsed = 5,
      MuonCandidateValuesUsed = 5
    };

    MuonMETCorrectionData() : type_(0), corrX_(0), corrY_(0) {}
    MuonMETCorrectionData(Type type, float corrX, float corrY) : type_(type), corrX_(corrX), corrY_(corrY) {}

    Type type() { return Type(type_); }
    float corrX() { return corrX_; }
    float corrY() { return corrY_; }
    float x() { return corrX_; }
    float y() { return corrY_; }
    float pt() { return sqrt(x() * x() + y() * y()); }

  protected:
    int type_;
    float corrX_;
    float corrY_;
  };

}  // namespace reco

#endif  //MuonReco_MuonMETCorrectionData_h
