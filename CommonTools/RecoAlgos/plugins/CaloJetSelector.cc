/* \class CaloJetSelector
 *
 * Selects track with a configurable string-based cut.
 * Saves clones of the selected tracks 
 *
 * \author: Luca Lista, INFN
 *
 * usage:
 *
 * module bestCaloJets = CaloJetSelector {
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
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

 typedef SingleObjectSelector<
           reco::CaloJetCollection, 
           StringCutObjectSelector<reco::CaloJet> 
         > CaloJetSelector;

DEFINE_FWK_MODULE( CaloJetSelector );
