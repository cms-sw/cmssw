/** \class reco::PattRecoTree
 *
 * \short Class for storing FFTJet sparse clustering trees
 *
 * This is a pure storage class with limited functionality.
 * Applications should use fftjet::SparseClusteringTree
 *
 * \author Igor Volobouev, TTU, June 16, 2010
 * \version   $Id: PattRecoTree.h,v 1.1 2010/11/22 23:27:57 igv Exp $
 ************************************************************/

#ifndef DataFormats_JetReco_PattRecoTree_h
#define DataFormats_JetReco_PattRecoTree_h

#include "DataFormats/JetReco/interface/PattRecoNode.h"

namespace reco {
    template<typename ScaleType, class Cluster>
    class PattRecoTree
    {
    public:
        typedef PattRecoNode<Cluster> Node;

        inline PattRecoTree() : sparse_(false) {}

        // Inspectors
        inline bool isSparse() const {return sparse_;}
        inline const std::vector<Node>& getNodes() const {return nodes_;}
        inline const std::vector<ScaleType>& getScales() const
        {return scales_;}

        // Modifiers
        inline void setSparse(const bool b) {sparse_ = b;}

        inline void clear()
        {nodes_.clear(); scales_.clear(); sparse_ = false;}

        inline void reserveNodes(const unsigned n) {nodes_.reserve(n);}
        inline void reserveScales(const unsigned n) {scales_.reserve(n);}
        inline void addNode(const Node& node) {nodes_.push_back(node);}
        inline void addScale(const double s)
        {scales_.push_back(static_cast<ScaleType>(s));}

    private:
        std::vector<Node> nodes_;
        std::vector<ScaleType> scales_;
        bool sparse_;
    };
}

#endif // JetReco_PattRecoTree_h
