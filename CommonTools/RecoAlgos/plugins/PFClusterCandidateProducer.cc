/** \class PFClusterRefCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Track compoment
 *
 * \author Steven Lowette
 *
 * $Id: PFClusterCandidateProducer.cc,v 1.1 2011/01/26 16:00:11 srappocc Exp $
 *
 */

#include "CommonTools/RecoAlgos/src/PFClusterToRefCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
          edm::View<reco::PFCluster>,
          reco::RecoPFClusterRefCandidateCollection,
          AnySelector,
          converter::helper::CandConverter<reco::PFCluster>::type
        > PFClusterRefCandidateProducer;

DEFINE_FWK_MODULE(PFClusterRefCandidateProducer);
