#ifndef DataFormats_CSCJetCandidate_H
#define DataFormats_CSCJetCandidate_H

#include <vector>
#include <memory>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Common/interface/SortedCollection.h"

#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include "DataFormats/Candidate/interface/LeafCandidate.h"

namespace reco {

  class CSCJetCandidate : public LeafCandidate{
  public:

    //default constructor
    CSCJetCandidate(): x_(0.),y_(0.),z_(0.),tPeak_(-999.),tWire_(-999.) {}
    
    CSCJetCandidate(const double phi,const double eta,const double x, const double y,const double z, const double tPeak, const double tWire);

    //destructor
    ~CSCJetCandidate() override;

    double xPos() const {return x_;}
    double yPos() const {return y_;}
    double zPos() const {return z_;}
    double tPeak() const {return tPeak_;}
    double tWire() const {return tWire_;}

   private:
    double x_;      
    double y_;
    double z_;      
    double tPeak_;      
    double tWire_;      
  };

   typedef std::vector<CSCJetCandidate> CSCJetCandidateCollection;
}
#endif
