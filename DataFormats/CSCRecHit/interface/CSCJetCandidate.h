#ifndef DataFormats_CSCJetCandidate_h
#define DataFormats_CSCJetCandidate_h

#include <vector>
#include <memory>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Common/interface/SortedCollection.h"

#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include "DataFormats/Candidate/interface/LeafCandidate.h"

namespace reco {

  class CSCJetCandidate : public LeafCandidate {
  public:
    //default constructor
    CSCJetCandidate()
        : x_(0.),
          y_(0.),
          z_(0.),
          tPeak_(-999.),
          tWire_(-999.),
          quality_(0),
          chamber_(0),
          station_(0),
          nStrips_(0),
          hitWire_(0),
          wgroupsBX_(0),
          nWireGroups_(0) {}

    CSCJetCandidate(const double phi,
                    const double eta,
                    const float x,
                    const float y,
                    const float z,
                    const float tPeak,
                    const float tWire,
                    const int quality,
                    const int chamber,
                    const int station,
                    const int nStrips,
                    const int hitWire,
                    const int wgroupsBX,
                    const int nWireGroups);

    //destructor
    ~CSCJetCandidate() override;

    float xPos() const { return x_; }
    float yPos() const { return y_; }
    float zPos() const { return z_; }
    float tPeak() const { return tPeak_; }
    float tWire() const { return tWire_; }
    int quality() const { return quality_; }
    int chamber() const { return chamber_; }
    int station() const { return station_; }
    int nStrips() const { return nStrips_; }
    int hitWire() const { return hitWire_; }
    int wgroupsBX() const { return wgroupsBX_; }
    int nWireGroups() const { return nWireGroups_; }

  private:
    float x_;
    float y_;
    float z_;
    float tPeak_;
    float tWire_;
    int quality_;
    int chamber_;
    int station_;
    int nStrips_;
    int hitWire_;
    int wgroupsBX_;
    int nWireGroups_;
  };

  typedef std::vector<CSCJetCandidate> CSCJetCandidateCollection;
}  // namespace reco
#endif
