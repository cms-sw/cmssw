/* \class JPTJetSelector
 *
 * Selects track with a configurable string-based cut.
 * Saves clones of the selected tracks 
 *
 * \author: Kalanand Mishra, Fermilab
 *
 * usage:
 *
 * module bestJPTJets = JPTJetSelector {
 *   src = ktJets
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
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"

 typedef SingleObjectSelector<
           reco::JPTJetCollection, 
           StringCutObjectSelector<reco::JPTJet> 
         > JPTJetSelector;

DEFINE_FWK_MODULE( JPTJetSelector );


