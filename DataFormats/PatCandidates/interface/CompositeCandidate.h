//
// $Id: CompositeCandidate.h,v 1.5 2008/10/08 19:04:42 gpetrucc Exp $
//

#ifndef DataFormats_PatCandidates_CompositeCandidate_h
#define DataFormats_PatCandidates_CompositeCandidate_h

/**
  \class    pat::CompositeCandidate CompositeCandidate.h "DataFormats/PatCandidates/interface/CompositeCandidate.h"
  \brief    Analysis-level particle class

   CompositeCandidate implements an analysis-level particle class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: CompositeCandidate.h,v 1.5 2008/10/08 19:04:42 gpetrucc Exp $
*/

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"

// Define typedefs for convenience
namespace pat {
  class CompositeCandidate;
  typedef std::vector<CompositeCandidate>              CompositeCandidateCollection; 
  typedef edm::Ref<CompositeCandidateCollection>       CompositeCandidateRef; 
  typedef edm::RefVector<CompositeCandidateCollection> CompositeCandidateRefVector; 
}

namespace pat {


  typedef reco::CompositeCandidate CompositeCandidateType;


  class CompositeCandidate : public PATObject<reco::CompositeCandidate> {

    public:

      CompositeCandidate();
      CompositeCandidate(const CompositeCandidateType & aCompositeCandidate);
      virtual ~CompositeCandidate();

      virtual CompositeCandidate * clone() const { return new CompositeCandidate(*this); }

  };


}

#endif
