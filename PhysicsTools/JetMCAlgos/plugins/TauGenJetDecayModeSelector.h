#ifndef PhysicsTools_JetMCAlgos_TauGenJetDecayModeSelector_h
#define PhysicsTools_JetMCAlgos_TauGenJetDecayModeSelector_h

/** \class TauGenJetDecayModeSelector
 *
 * Selection of GenJet objects representing generated tau-decays 
 * by tau decay mode
 *
 * \author Christian Veelken, UC Davis
 *
 * \version $Revision: 1.3 $
 *
 * $Id: TauGenJetDecayModeSelector.h,v 1.3 2010/02/11 00:13:19 wmtan Exp $
 *
 */

#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

class TauGenJetDecayModeSelectorImp
{
 public:
  // constructor 
  explicit TauGenJetDecayModeSelectorImp(const edm::ParameterSet&);
  
  bool operator()(const reco::GenJet&) const;

 private:
  typedef std::vector<std::string> vstring;
  vstring selectedTauDecayModes_;
};

typedef SingleObjectSelector<
            reco::GenJetCollection,
            TauGenJetDecayModeSelectorImp
        > TauGenJetDecayModeSelector;

#endif
