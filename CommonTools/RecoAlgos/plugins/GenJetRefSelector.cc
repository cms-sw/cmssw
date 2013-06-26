/* \class GenJetRefSelector
 *
 * Selects GenJet with a configurable string-based cut.
 * Saves references to the selected tracks
 *
 * \author: Attilio Santocchia, INFN
 *
 * usage:
 *
 * module bestGenJets = GenJetRefSelector {
 *   src = myGenJets
 *   string cut = "pt > 20 & abs( eta ) < 2"
 * }
 *
 * for more details about the cut syntax, see the documentation
 * page below:
 *
 *   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
 *
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

 typedef SingleObjectSelector<
           reco::GenJetCollection, 
           StringCutObjectSelector<reco::GenJet>,
           reco::GenJetRefVector
         > GenJetRefSelector;

DEFINE_FWK_MODULE( GenJetRefSelector );
