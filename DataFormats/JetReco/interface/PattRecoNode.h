/** \class reco::PattRecoNode
 *
 * \short Tree nodes for storing FFTJet preclusters
 *
 * This is a pure storage class with limited functionality.
 * Applications should use fftjet::SparseClusteringTree::Node
 *
 * \author Igor Volobouev, TTU, June 16, 2010
 * \version   $Id: PattRecoNode.h,v 1.1 2010/11/22 23:27:56 igv Exp $
 ************************************************************/

#ifndef DataFormats_JetReco_PattRecoNode_h
#define DataFormats_JetReco_PattRecoNode_h

namespace reco {
    template<class Cluster>
    class PattRecoNode
    {
    public:
        inline PattRecoNode() : originalLevel_(0), nodeMask_(0), parent_(0) {}

        inline PattRecoNode(const Cluster& j, const unsigned level,
                            const unsigned mask, const unsigned parent)
            : jet_(j), originalLevel_(level),
              nodeMask_(mask), parent_(parent) {}

        inline const Cluster& getCluster() const {return jet_;}
        inline unsigned originalLevel() const {return originalLevel_;}
        inline unsigned mask() const {return nodeMask_;}
        inline unsigned parent() const {return parent_;}

    private:
        Cluster jet_;
        unsigned originalLevel_;
        unsigned nodeMask_;
        unsigned parent_;
    };
}

#endif // JetReco_PattRecoNode_h
