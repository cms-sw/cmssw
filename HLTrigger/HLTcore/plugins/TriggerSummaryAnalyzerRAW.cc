/** \class TriggerSummaryAnalyzerRAW
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/TriggerSummaryAnalyzerRAW.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
TriggerSummaryAnalyzerRAW::TriggerSummaryAnalyzerRAW(const edm::ParameterSet& ps) : 
  inputTag_(ps.getParameter<edm::InputTag>("inputTag")),
  inputToken_(consumes<trigger::TriggerEventWithRefs>(inputTag_))
{ }

TriggerSummaryAnalyzerRAW::~TriggerSummaryAnalyzerRAW()
{
}

//
// member functions
//

void TriggerSummaryAnalyzerRAW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltTriggerSummaryRAW"));
  descriptions.add("triggerSummaryAnalyzerRAW", desc);
}

// ------------ method called to produce the data  ------------
void
TriggerSummaryAnalyzerRAW::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace l1extra;
   using namespace trigger;

   LogVerbatim("TriggerSummaryAnalyzerRAW") << endl;
   LogVerbatim("TriggerSummaryAnalyzerRAW") << "TriggerSummaryAnalyzerRAW: content of TriggerEventWithRefs: " << inputTag_.encode();

   Handle<TriggerEventWithRefs> handle;
   iEvent.getByToken(inputToken_,handle);
   if (handle.isValid()) {
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "Used Processname: " << handle->usedProcessName() << endl;
     const size_type nFO(handle->size());
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "Number of TriggerFilterObjects: " << nFO << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "The TriggerFilterObjects: #, tag" << endl;
     for (size_type iFO=0; iFO!=nFO; ++iFO) {
       LogVerbatim("TriggerSummaryAnalyzerRAW") << iFO << " " << handle->filterTag(iFO).encode() << endl;
       LogVerbatim("TriggerSummaryAnalyzerRAW") << "  # of objects:";

       const unsigned int nPhotons(handle->photonSlice(iFO).second-
				   handle->photonSlice(iFO).first);
       if (nPhotons>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " Photons: " << nPhotons;

       const unsigned int nElectrons(handle->electronSlice(iFO).second-
				     handle->electronSlice(iFO).first);
       if (nElectrons>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " Electrons: " << nElectrons;

       const unsigned int nMuons(handle->muonSlice(iFO).second-
				 handle->muonSlice(iFO).first);
       if (nMuons>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " Muons: " << nMuons;

       const unsigned int nJets(handle->jetSlice(iFO).second-
				handle->jetSlice(iFO).first);
       if (nJets>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " Jets: " << nJets;

       const unsigned int nComposites(handle->compositeSlice(iFO).second-
				      handle->compositeSlice(iFO).first);
       if (nComposites>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " Composites: " << nComposites;

       const unsigned int nBaseMETs(handle->basemetSlice(iFO).second-
				    handle->basemetSlice(iFO).first);
       if (nBaseMETs>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " BaseMETs: " << nBaseMETs;

       const unsigned int nCaloMETs(handle->calometSlice(iFO).second-
				    handle->calometSlice(iFO).first);
       if (nCaloMETs>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " CaloMETs: " << nCaloMETs;

       const unsigned int nPixTracks(handle->pixtrackSlice(iFO).second-
				     handle->pixtrackSlice(iFO).first);
       if (nPixTracks>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " PixTracks: " << nPixTracks;

       const unsigned int nL1EM(handle->l1emSlice(iFO).second-
				handle->l1emSlice(iFO).first);
       if (nL1EM>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1EM: " << nL1EM;

       const unsigned int nL1Muon(handle->l1muonSlice(iFO).second-
				  handle->l1muonSlice(iFO).first);
       if (nL1Muon>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1Muon: " << nL1Muon;

       const unsigned int nL1Jet(handle->l1jetSlice(iFO).second-
				 handle->l1jetSlice(iFO).first);
       if (nL1Jet>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1Jet: " << nL1Jet;

       const unsigned int nL1EtMiss(handle->l1etmissSlice(iFO).second-
				    handle->l1etmissSlice(iFO).first);
       if (nL1EtMiss>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1EtMiss: " << nL1EtMiss;

       const unsigned int nL1HfRings(handle->l1hfringsSlice(iFO).second-
				    handle->l1hfringsSlice(iFO).first);
       if (nL1HfRings>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " L1HfRings: " << nL1HfRings;

       const unsigned int nPFJets(handle->pfjetSlice(iFO).second-
				handle->pfjetSlice(iFO).first);
       if (nPFJets>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " PFJets: " << nPFJets;

       const unsigned int nPFTaus(handle->pftauSlice(iFO).second-
				handle->pftauSlice(iFO).first);
       if (nPFTaus>0) LogVerbatim("TriggerSummaryAnalyzerRAW") << " PFTaus: " << nPFTaus;

       LogVerbatim("TriggerSummaryAnalyzerRAW") << endl;
     }
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "Elements in linearised collections of Refs: " << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  Photons:    " << handle->photonSize()    << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  Electrons:  " << handle->electronSize()  << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  Muons:      " << handle->muonSize()      << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  Jets:       " << handle->jetSize()       << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  Composites: " << handle->compositeSize() << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  BaseMETs:   " << handle->basemetSize()   << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  CaloMETs:   " << handle->calometSize()   << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  Pixtracks:  " << handle->pixtrackSize()  << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1EM:       " << handle->l1emSize()      << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1Muon:     " << handle->l1muonSize()    << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1Jet:      " << handle->l1jetSize()     << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1EtMiss:   " << handle->l1etmissSize()  << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  L1HfRings:  " << handle->l1hfringsSize() << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  PFJets:     " << handle->pfjetSize()     << endl;
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "  PFTaus:     " << handle->pftauSize()     << endl;
   } else {
     LogVerbatim("TriggerSummaryAnalyzerRAW") << "Handle invalid! Check InputTag provided." << endl;
   }
   LogVerbatim("TriggerSummaryAnalyzerRAW") << endl;
   
   return;
}
