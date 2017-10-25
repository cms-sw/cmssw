#ifndef DTSEGMENTEXTENDEDCAND_H
#define DTSEGMENTEXTENDEDCAND_H

/** \class DTSegmentExtendedCand
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 */

/* Base Class Headers */
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"

/* Collaborating Class Declarations */
#include "DataFormats/DTRecHit/interface/DTSLRecCluster.h"

/* C++ Headers */
#include <vector>

/* ====================================================================== */

/* Class DTSegmentExtendedCand Interface */

class DTSegmentExtendedCand  : public DTSegmentCand {

  public:
    struct DTSLRecClusterForFit ;

/* Constructor */ 
    DTSegmentExtendedCand(DTSegmentCand* cand ): DTSegmentCand(*cand),
                                                  theClus(std::vector<DTSLRecClusterForFit>()) {
                                                  }

/* Destructor */ 
    ~DTSegmentExtendedCand() override {}

/* Operations */ 
    void addClus(const DTSegmentExtendedCand::DTSLRecClusterForFit& clus) {
      theClus.push_back(clus);
    }

    std::vector<DTSegmentExtendedCand::DTSLRecClusterForFit> clusters() const {
      return theClus;
    }

    //DTSegmentCand* cand() const { return theCand; }

    bool isCompatible(const DTSegmentExtendedCand::DTSLRecClusterForFit& clus) ;

    unsigned int nHits() const override ;

    bool good() const override ;

    struct DTSLRecClusterForFit {
      public: 
        DTSLRecClusterForFit(const DTSLRecCluster& c,
                             const LocalPoint& p,
                             const LocalError& e) : clus(c), pos(p), err(e) {}
        DTSLRecCluster clus;
        LocalPoint pos;
        LocalError err;
    };

  private:
    //DTSegmentCand* theCand;
    std::vector<DTSLRecClusterForFit> theClus;
    //double theChi2;

  protected:

};
#endif // DTSEGMENTEXTENDEDCAND_H

