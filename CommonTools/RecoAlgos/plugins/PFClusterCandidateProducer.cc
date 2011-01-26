/** \class PFClusterRefCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Track compoment
 *
 * \author Steven Lowette
 *
 * $Id: PFClusterRefCandidateProducer.cc,v 1.1 2009/11/26 11:49:29 lowette Exp $
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
