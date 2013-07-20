/** \class PFClusterRefCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Track compoment
 *
 * \author Steven Lowette
 *
 * $Id: PFClusterCandidateProducer.cc,v 1.2 2013/02/28 00:17:19 wmtan Exp $
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
