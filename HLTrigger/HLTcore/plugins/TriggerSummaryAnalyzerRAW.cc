/** \class TriggerSummaryAnalyzerRAW
 *
 * See header file for documentation
 *
 *  $Date: 2012/01/30 09:40:35 $
 *  $Revision: 1.11 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/TriggerSummaryAnalyzerRAW.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

//
// constructors and destructor
//
TriggerSummaryAnalyzerRAW::TriggerSummaryAnalyzerRAW(const edm::ParameterSet& ps) : 
  inputTag_(ps.getParameter<edm::InputTag>("inputTag"))
{ }

TriggerSummaryAnalyzerRAW::~TriggerSummaryAnalyzerRAW()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
TriggerSummaryAnalyzerRAW::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace l1extra;
   using namespace trigger;

   cout << endl;
   cout << "TriggerSummaryAnalyzerRAW: content of TriggerEventWithRefs: " << inputTag_.encode();

   Handle<TriggerEventWithRefs> handle;
   iEvent.getByLabel(inputTag_,handle);
   if (handle.isValid()) {
     cout << "Used Processname: " << handle->usedProcessName() << endl;
     const size_type nFO(handle->size());
     cout << "Number of TriggerFilterObjects: " << nFO << endl;
     cout << "The TriggerFilterObjects: #, tag" << endl;
     for (size_type iFO=0; iFO!=nFO; ++iFO) {
       cout << iFO << " " << handle->filterTag(iFO).encode() << endl;
       cout << "  # of objects:";

       const unsigned int nPhotons(handle->photonSlice(iFO).second-
				   handle->photonSlice(iFO).first);
       if (nPhotons>0) cout << " Photons: " << nPhotons;

       const unsigned int nElectrons(handle->electronSlice(iFO).second-
				     handle->electronSlice(iFO).first);
       if (nElectrons>0) cout << " Electrons: " << nElectrons;

       const unsigned int nMuons(handle->muonSlice(iFO).second-
				 handle->muonSlice(iFO).first);
       if (nMuons>0) cout << " Muons: " << nMuons;

       const unsigned int nJets(handle->jetSlice(iFO).second-
				handle->jetSlice(iFO).first);
       if (nJets>0) cout << " Jets: " << nJets;

       const unsigned int nComposites(handle->compositeSlice(iFO).second-
				      handle->compositeSlice(iFO).first);
       if (nComposites>0) cout << " Composites: " << nComposites;

       const unsigned int nBaseMETs(handle->basemetSlice(iFO).second-
				    handle->basemetSlice(iFO).first);
       if (nBaseMETs>0) cout << " BaseMETs: " << nBaseMETs;

       const unsigned int nCaloMETs(handle->calometSlice(iFO).second-
				    handle->calometSlice(iFO).first);
       if (nCaloMETs>0) cout << " CaloMETs: " << nCaloMETs;

       const unsigned int nPixTracks(handle->pixtrackSlice(iFO).second-
				     handle->pixtrackSlice(iFO).first);
       if (nPixTracks>0) cout << " PixTracks: " << nPixTracks;

       const unsigned int nL1EM(handle->l1emSlice(iFO).second-
				handle->l1emSlice(iFO).first);
       if (nL1EM>0) cout << " L1EM: " << nL1EM;

       const unsigned int nL1Muon(handle->l1muonSlice(iFO).second-
				  handle->l1muonSlice(iFO).first);
       if (nL1Muon>0) cout << " L1Muon: " << nL1Muon;

       const unsigned int nL1Jet(handle->l1jetSlice(iFO).second-
				 handle->l1jetSlice(iFO).first);
       if (nL1Jet>0) cout << " L1Jet: " << nL1Jet;

       const unsigned int nL1EtMiss(handle->l1etmissSlice(iFO).second-
				    handle->l1etmissSlice(iFO).first);
       if (nL1EtMiss>0) cout << " L1EtMiss: " << nL1EtMiss;

       const unsigned int nL1HfRings(handle->l1hfringsSlice(iFO).second-
				    handle->l1hfringsSlice(iFO).first);
       if (nL1HfRings>0) cout << " L1HfRings: " << nL1HfRings;

       const unsigned int nPFJets(handle->pfjetSlice(iFO).second-
				handle->pfjetSlice(iFO).first);
       if (nPFJets>0) cout << " PFJets: " << nPFJets;

       const unsigned int nPFTaus(handle->pftauSlice(iFO).second-
				handle->pftauSlice(iFO).first);
       if (nPFTaus>0) cout << " PFTaus: " << nPFTaus;

       cout << endl;
     }
     cout << "Elements in linearised collections of Refs: " << endl;
     cout << "  Photons:    " << handle->photonSize()    << endl;
     cout << "  Electrons:  " << handle->electronSize()  << endl;
     cout << "  Muons:      " << handle->muonSize()      << endl;
     cout << "  Jets:       " << handle->jetSize()       << endl;
     cout << "  Composites: " << handle->compositeSize() << endl;
     cout << "  BaseMETs:   " << handle->basemetSize()   << endl;
     cout << "  CaloMETs:   " << handle->calometSize()   << endl;
     cout << "  Pixtracks:  " << handle->pixtrackSize()  << endl;
     cout << "  L1EM:       " << handle->l1emSize()      << endl;
     cout << "  L1Muon:     " << handle->l1muonSize()    << endl;
     cout << "  L1Jet:      " << handle->l1jetSize()     << endl;
     cout << "  L1EtMiss:   " << handle->l1etmissSize()  << endl;
     cout << "  L1HfRings:  " << handle->l1hfringsSize() << endl;
     cout << "  PFJets:     " << handle->pfjetSize()     << endl;
     cout << "  PFTaus:     " << handle->pftauSize()     << endl;
   } else {
     cout << "Handle invalid! Check InputTag provided." << endl;
   }
   cout << endl;
   
   return;
}
