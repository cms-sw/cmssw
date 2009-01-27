#include "RecoTauTag/RecoTau/interface/CMSInsideOutJetProducer.h"


namespace {
  const bool debug = false;

  template <class T>  
  void dumpJets (const T& fJets) {
    for (unsigned i = 0; i < fJets.size(); ++i) {
      std::cout << "Jet # " << i << std::endl << fJets[i].print();
    }
  }

  void copyVariables (const ProtoJet& fProtojet, reco::Jet* fJet) {
    fJet->setJetArea (fProtojet.jetArea ());
    fJet->setPileup (fProtojet.pileup ());
    fJet->setNPasses (fProtojet.nPasses ());
  }

  void copyConstituents (const JetReco::InputCollection& fConstituents, const edm::View <Candidate>& fInput, reco::Jet* fJet) {
    // put constituents
    for (unsigned iConstituent = 0; iConstituent < fConstituents.size (); ++iConstituent) {
      fJet->addDaughter (fInput.ptrAt (fConstituents[iConstituent].index ()));
    }
  }

}

CMSInsideOutProducer::CMSInsideOutProducer(const edm::ParameterSet& conf)
 : alg_( conf.getParameter<double>("seedObjectPt"),    
         conf.getParameter<double>("growthParameter"), 
         conf.getParameter<double>("maxSize"),         
         conf.getParameter<double>("minSize")),        
   mSrc(conf.getParameter<edm::InputTag>( "src" )),
   mVerbose (conf.getUntrackedParameter<bool>("verbose", false)),
   mEtInputCut (conf.getParameter<double>("inputEtMin")),
   mEInputCut (conf.getParameter<double>("inputEMin"))
{
   //setup product
   produces<PFJetCollection>();
}

void CMSInsideOutProducer::produce(edm::Event& event, const edm::EventSetup& fSetup)
{
   edm::Handle<edm::View<Candidate> > inputHandle;
   event.getByLabel( mSrc, inputHandle);
   // convert to input collection
   JetReco::InputCollection input;
   input.reserve (inputHandle->size());
   for (unsigned int i = 0; i < inputHandle->size(); ++i) {
      if ((mEtInputCut <= 0 || (*inputHandle)[i].et()     > mEtInputCut) &&
          (mEInputCut  <= 0 || (*inputHandle)[i].energy() > mEInputCut )) {
         input.push_back (JetReco::InputItem (&((*inputHandle)[i]), i));
      }
   }
   // run algorithm
   vector <ProtoJet> output;
   if (input.empty ()) {
      edm::LogInfo ("Empty Event") << "empty input for jet algorithm: bypassing..." << std::endl;
   }
   else {
      alg_.run (input, &output);
   }

   reco::Jet::Point vertex (0,0,0); // do not have true vertex yet, use default

   auto_ptr<PFJetCollection> jets (new PFJetCollection);
   jets->reserve(output.size());
   for (unsigned iJet = 0; iJet < output.size (); ++iJet) {
      ProtoJet* protojet = &(output [iJet]);
      const JetReco::InputCollection& constituents = protojet->getTowerList();
      PFJet::Specific specific;
      JetMaker::makeSpecific (constituents, &specific);
      jets->push_back (PFJet (protojet->p4(), vertex, specific));
      Jet* newJet = &(jets->back());
      copyConstituents (constituents, *inputHandle, newJet);
      copyVariables (*protojet, newJet);
   }
   if (mVerbose) dumpJets (*jets);
   event.put(jets);
} // end produce(...)


