/** \class PFClusterRefCandidateMerger
 *
 * Framework module that produces a merged collection
 * of candidates with a PFCluster compoment
 *
 * \author Hartmut Stadie
 *
 * $Id: PFClusterCandidateProducer.cc,v 1.1 2011/01/26 16:00:11 srappocc Exp $
 *
 */

#include "CommonTools/RecoAlgos/src/PFClusterToRefCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"

typedef Merger<reco::RecoPFClusterRefCandidateCollection>
        PFClusterRefCandidateMerger;

DEFINE_FWK_MODULE(PFClusterRefCandidateMerger);
