//
// Code converting data between FFTJet clustering tree objects and
// event-storable entities. The data can be stored in either single
// or double precision. If the data is stored in double precision,
// the clustering tree can be restored in its original form.
//
// Written by:
//
// Terence Libeiro
// Igor Volobouev
//
// June 2010

#ifndef RecoJets_FFTJetAlgorithms_clusteringTreeConverters_h
#define RecoJets_FFTJetAlgorithms_clusteringTreeConverters_h

#include <cfloat>

#include "fftjet/Peak.hh"
#include "fftjet/AbsClusteringTree.hh"
#include "fftjet/SparseClusteringTree.hh"

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/JetReco/interface/PattRecoTree.h"
#include "DataFormats/JetReco/interface/PattRecoPeak.h"

namespace fftjetcms {
    // The function below makes PattRecoTree storable in the event
    // record out of a sparse clustering tree of fftjet::Peak objects
    template<class Real>
    void sparsePeakTreeToStorable(
        const fftjet::SparseClusteringTree<fftjet::Peak,long>& in,
        bool writeOutScaleInfo,
        reco::PattRecoTree<Real,reco::PattRecoPeak<Real> >* out);

    // The function below restores sparse clustering tree of fftjet::Peak
    // objects using the PattRecoTree. The "completeEventScale" parameter
    // should be set to the appropriate scale in case the complete event
    // was written out and its scale not stored in the event. If the complete
    // event was not written out, the scale must be set to 0.
    template<class Real>
    void sparsePeakTreeFromStorable(
        const reco::PattRecoTree<Real,reco::PattRecoPeak<Real> >& in,
        const std::vector<double>* scaleSetIfNotAdaptive,
        double completeEventScale,
        fftjet::SparseClusteringTree<fftjet::Peak,long>* out);

    // The function below makes PattRecoTree storable in the event
    // record out of a dense clustering tree of fftjet::Peak objects
    template<class Real>
    void densePeakTreeToStorable(
        const fftjet::AbsClusteringTree<fftjet::Peak,long>& in,
        bool writeOutScaleInfo,
        reco::PattRecoTree<Real,reco::PattRecoPeak<Real> >* out);

    // The function below restores dense clustering tree of fftjet::Peak
    // objects using the PattRecoTree. The "completeEventScale" parameter
    // should be set to the appropriate scale in case the complete event
    // was written out and its scale not stored in the event. If the complete
    // event was not written out, the scale must be set to 0.
    template<class Real>
    void densePeakTreeFromStorable(
        const reco::PattRecoTree<Real,reco::PattRecoPeak<Real> >& in,
        const std::vector<double>* scaleSetIfNotAdaptive,
        double completeEventScale,
        fftjet::AbsClusteringTree<fftjet::Peak,long>* out);
}

////////////////////////////////////////////////////////////////////////
//
//  Implementation follows
//
////////////////////////////////////////////////////////////////////////

namespace fftjetcms {
    // The function below makes PattRecoTree storable in the event
    // record out of a sparse clustering tree of fftjet::Peak objects
    template<class Real>
    void sparsePeakTreeToStorable(
        const fftjet::SparseClusteringTree<fftjet::Peak,long>& sparseTree,
        const bool writeOutScaleInfo,
        reco::PattRecoTree<Real,reco::PattRecoPeak<Real> >* tree)
    {
        typedef fftjet::SparseClusteringTree<fftjet::Peak,long> SparseTree;
        typedef reco::PattRecoPeak<Real> StoredPeak;
        typedef reco::PattRecoNode<StoredPeak> StoredNode;
        typedef reco::PattRecoTree<Real,StoredPeak> StoredTree;

        assert(tree);

        tree->clear();
        tree->setSparse(true);

        const unsigned nNodes = sparseTree.size();
        double hessian[3] = {0., 0., 0.};

        // Do not write out the meaningless top node
        tree->reserveNodes(nNodes - 1);
        for (unsigned i=1; i<nNodes; ++i)
        {
            const SparseTree::Node& node(sparseTree.getNode(i));
            const fftjet::Peak& peak(node.getCluster());
            peak.hessian(hessian);
            StoredNode sn(StoredPeak(peak.eta(),
                                     peak.phi(),
                                     peak.magnitude(),
                                     hessian,
                                     peak.driftSpeed(),
                                     peak.magSpeed(),
                                     peak.lifetime(),
                                     peak.scale(),
                                     peak.nearestNeighborDistance(),
                                     peak.clusterRadius(),
                                     peak.clusterSeparation()),
                          node.originalLevel(),
                          node.mask(),
                          node.parent());
            tree->addNode(sn);
        }

        // Do we want to write out the scales? We will use the following
        // convention: if the tree is using an adaptive algorithm, the scales
        // will be written out. If not, they are not going to change from
        // event to event. In this case the scales would waste disk space
        // for no particularly good reason, so they will not be written out.
        if (writeOutScaleInfo)
        {
            // Do not write out the meaningless top-level scale
            const unsigned nScales = sparseTree.nLevels();
            tree->reserveScales(nScales - 1);
            for (unsigned i=1; i<nScales; ++i)
                tree->addScale(sparseTree.getScale(i));
        }
    }

    // The function below restores sparse clustering tree of fftjet::Peak
    // objects using the PattRecoTree
    template<class Real>
    void sparsePeakTreeFromStorable(
        const reco::PattRecoTree<Real,reco::PattRecoPeak<Real> >& in,
        const std::vector<double>* scaleSetIfNotAdaptive,
        const double completeEventScale,
        fftjet::SparseClusteringTree<fftjet::Peak,long>* out)
    {
        typedef fftjet::SparseClusteringTree<fftjet::Peak,long> SparseTree;
        typedef reco::PattRecoPeak<Real> StoredPeak;
        typedef reco::PattRecoNode<StoredPeak> StoredNode;
        typedef reco::PattRecoTree<Real,StoredPeak> StoredTree;

        if (!in.isSparse())
            throw cms::Exception("FFTJetBadConfig")
                << "can't restore sparse clustering tree"
                << " from densely stored record" << std::endl;

        assert(out);
        out->clear();

        const std::vector<StoredNode>& nodes(in.getNodes());
        const unsigned n = nodes.size();
        out->reserveNodes(n + 1U);

        double hessian[3] = {0., 0., 0.};

        for (unsigned i=0; i<n; ++i)
        {
            const StoredNode& snode(nodes[i]);
            const StoredPeak& p(snode.getCluster());
            p.hessian(hessian);
            const SparseTree::Node node(
                fftjet::Peak(p.eta(), p.phi(), p.magnitude(),
                             hessian, p.driftSpeed(),
                             p.magSpeed(), p.lifetime(),
                             p.scale(), p.nearestNeighborDistance(),
                             1.0, 0.0, 0.0,
                             p.clusterRadius(), p.clusterSeparation()),
                snode.originalLevel(),
                snode.mask());
            out->addNode(node, snode.parent());
        }

        const std::vector<Real>& storedScales(in.getScales());
        if (!storedScales.empty())
        {
            const unsigned nsc = storedScales.size();
            out->reserveScales(nsc + 1U);
            out->addScale(DBL_MAX);
            const Real* scales = &storedScales[0];
            for (unsigned i=0; i<nsc; ++i)
                out->addScale(scales[i]);
        }
        else if (scaleSetIfNotAdaptive && !scaleSetIfNotAdaptive->empty())
        {
            const unsigned nsc = scaleSetIfNotAdaptive->size();
            // There may be the "complete event" scale added at the end.
            // Reserve a sufficient number of scales to take this into
            // account.
            if (completeEventScale)
                out->reserveScales(nsc + 2U);
            else
                out->reserveScales(nsc + 1U);
            out->addScale(DBL_MAX);
            const double* scales = &(*scaleSetIfNotAdaptive)[0];
            for (unsigned i=0; i<nsc; ++i)
                out->addScale(scales[i]);
            if (completeEventScale)
                out->addScale(completeEventScale);
        }
        else
        {
            throw cms::Exception("FFTJetBadConfig")
                << "can't restore sparse clustering tree scales"
                << std::endl;
        }
    }

    // The function below makes PattRecoTree storable in the event
    // record out of a dense clustering tree of fftjet::Peak objects
    template<class Real>
    void densePeakTreeToStorable(
        const fftjet::AbsClusteringTree<fftjet::Peak,long>& in,
        bool writeOutScaleInfo,
        reco::PattRecoTree<Real,reco::PattRecoPeak<Real> >* out)
    {
        typedef fftjet::AbsClusteringTree<fftjet::Peak,long> DenseTree;
        typedef reco::PattRecoPeak<Real> StoredPeak;
        typedef reco::PattRecoNode<StoredPeak> StoredNode;
        typedef reco::PattRecoTree<Real,StoredPeak> StoredTree;
        
        
        assert(out);
        out->clear();
        out->setSparse(false);
        
        
        const unsigned nLevels = in.nLevels();
        double hessian[3] = {0., 0., 0.};

        
        // Do not write out the meaningless top node
        out->reserveNodes(in.nClusters() - 1);
        
        
        for (unsigned i=1; i<nLevels; ++i)
        {
            
            const unsigned int nclus = in.nClusters(i);
            DenseTree::NodeId id(i,0);
            for (;id.second<nclus; ++id.second) 
            {
           
           
                const fftjet::Peak& peak(in.getCluster(id));
                peak.hessian(hessian);
                StoredNode sn(StoredPeak(peak.eta(),
                                         peak.phi(),
                                         peak.magnitude(),
                                         hessian,
                                         peak.driftSpeed(),
                                         peak.magSpeed(),
                                         peak.lifetime(),
                                         peak.scale(),
                                         peak.nearestNeighborDistance(),
                                         peak.clusterRadius(),
                                         peak.clusterSeparation()),
                              i,
                              0,
                              0);
                
                out->addNode(sn);
                
            }
            
        }
        
    
        // Do we want to write out the scales? We will use the following
        // convention: if the tree is using an adaptive algorithm, the scales
        // will be written out. If not, they are not going to change from
        // event to event. In this case the scales would waste disk space
        // for no particularly good reason, so they will not be written out.
        if (writeOutScaleInfo)
        {
            // Do not write out the meaningless top-level scale
            const unsigned nScales = in.nLevels();
            
            out->reserveScales(nScales - 1);
            
            for (unsigned i=1; i<nScales; ++i)
                out->addScale(in.getScale(i));
            
        }
    }

    // The function below restores dense clustering tree of fftjet::Peak
    // objects using the PattRecoTree
    template<class Real>
    void densePeakTreeFromStorable(
        const reco::PattRecoTree<Real,reco::PattRecoPeak<Real> >& in,
        const std::vector<double>* scaleSetIfNotAdaptive,
        const double completeEventScale,
        fftjet::AbsClusteringTree<fftjet::Peak,long>* out)
    {
        typedef fftjet::AbsClusteringTree<fftjet::Peak,long> DenseTree;
        typedef reco::PattRecoPeak<Real> StoredPeak;
        typedef reco::PattRecoNode<StoredPeak> StoredNode;
        typedef reco::PattRecoTree<Real,StoredPeak> StoredTree;

        if (in.isSparse())
            throw cms::Exception("FFTJetBadConfig")
                << "can't restore dense clustering tree"
                << " from sparsely stored record" << std::endl;

        assert(out);
        out->clear();

        const std::vector<StoredNode>& nodes(in.getNodes());
        const unsigned n = nodes.size();
        double hessian[3] = {0., 0., 0.};
                
        const std::vector<Real>& scales (in.getScales());
        unsigned int scnum     = 0;
        std::vector<fftjet::Peak> clusters;
        const unsigned scsize1 = scales.size();
        unsigned scsize2 = scaleSetIfNotAdaptive ? scaleSetIfNotAdaptive->size() : 0;
        if (scsize2 && completeEventScale) ++scsize2;
        const unsigned scsize  = (scsize1==0?scsize2:scsize1);

        if (scsize == 0)  
            throw cms::Exception("FFTJetBadConfig")
                << " No scales passed to the function densePeakTreeFromStorable()"
                << std::endl;

        // to check whether the largest level equals the size of scale vector
        const double* sc_not_ad = scsize2 ? &(*scaleSetIfNotAdaptive)[0] : 0;

        unsigned templevel = n ? nodes[0].originalLevel() : 1;
        for (unsigned i=0; i<n; ++i)
        {
            const StoredNode& snode(nodes[i]);
            const StoredPeak& p(snode.getCluster());
            p.hessian(hessian);
        
            const unsigned levelNumber = snode.originalLevel();

            if (templevel != levelNumber) 
            {
                if (scnum >= scsize)
                    throw cms::Exception("FFTJetBadConfig")
                        << "bad scales, please check the scales"
                        << std::endl;
                const double scale = ( (scsize1==0) ? sc_not_ad[scnum] : scales[scnum] );
                out->insert(scale, clusters, 0L);
                clusters.clear();
                templevel = levelNumber;
                ++scnum;
            }

            fftjet::Peak apeak(p.eta(), p.phi(), p.magnitude(),
                               hessian, p.driftSpeed(),
                               p.magSpeed(), p.lifetime(),
                               p.scale(), p.nearestNeighborDistance(),
                               1.0, 0.0, 0.0,
                               p.clusterRadius(), p.clusterSeparation());
            clusters.push_back(apeak);
               
            if (i==(n-1) && levelNumber!=scsize) 
                throw cms::Exception("FFTJetBadConfig")
                    << "bad scales, please check the scales"
                    << std::endl;
        }

        const double scale = scsize1 ? scales[scnum] : completeEventScale ? 
            completeEventScale : sc_not_ad[scnum];
        out->insert(scale, clusters, 0L);
    }
}

#endif // RecoJets_FFTJetAlgorithms_clusteringTreeConverters_h
