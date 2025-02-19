#ifndef PhysicsTools_PatUtils_PATJetCorrExtractor_h
#define PhysicsTools_PatUtils_PATJetCorrExtractor_h

/** \class PATJetCorrExtractor
 *
 * Retrieve jet energy correction factor for pat::Jets (of either PF-type or Calo-type)
 *
 * NOTE: this specialization of the "generic" template defined in
 *         JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h
 *       is to be used for pat::Jets only
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.5 $
 *
 * $Id: PATJetCorrExtractor.h,v 1.5 2012/02/13 14:12:12 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>
#include <vector>

namespace
{
  std::string format_vstring(const std::vector<std::string>& v)
  {
    std::string retVal;
  
    retVal.append("{ ");

    unsigned numEntries = v.size();
    for ( unsigned iEntry = 0; iEntry < numEntries; ++iEntry ) {
      retVal.append(v[iEntry]);
      if ( iEntry < (numEntries - 1) ) retVal.append(", ");
    }
    
    retVal.append(" }");
    
    return retVal;
  }
}

class PATJetCorrExtractor
{
 public:

  reco::Candidate::LorentzVector operator()(const pat::Jet& jet, const std::string& jetCorrLabel, 
					    const edm::Event* evt = 0, const edm::EventSetup* es = 0, 
					    double jetCorrEtaMax = 9.9, 
					    const reco::Candidate::LorentzVector* rawJetP4_specified = 0)
  {
    reco::Candidate::LorentzVector corrJetP4;

    try {
      corrJetP4 = jet.correctedP4(jetCorrLabel);
    } catch( cms::Exception e ) {
      throw cms::Exception("InvalidRequest") 
	<< "The JEC level " << jetCorrLabel << " does not exist !!\n"
	<< "Available levels = " << format_vstring(jet.availableJECLevels()) << ".\n";
    }

    return corrJetP4;
  }
};

#endif


