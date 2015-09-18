#ifndef CorrectedJetProducer_h
#define CorrectedJetProducer_h

#include <sstream>
#include <string>
#include <vector>

#include "CommonTools/Utils/interface/PtComparator.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

namespace edm
{
  class ParameterSet;
}

namespace reco
{
  template<class T>
  class CorrectedJetProducer : public edm::global::EDProducer<>
  {
  public:
    typedef std::vector<T> JetCollection;
    explicit CorrectedJetProducer (const edm::ParameterSet& fParameters);
    virtual ~CorrectedJetProducer () {}
    virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  private:
    const edm::EDGetTokenT<JetCollection> mInput;
    const std::vector <edm::EDGetTokenT<reco::JetCorrector> > mCorrectorTokens;
    const bool mVerbose;
  };
}

// ---------- implementation ----------

namespace reco {

  template<class T>
  CorrectedJetProducer<T>::CorrectedJetProducer(const edm::ParameterSet& fConfig)
    : mInput(consumes<JetCollection>(fConfig.getParameter <edm::InputTag> ("src")))
    , mCorrectorTokens(edm::vector_transform(fConfig.getParameter<std::vector<edm::InputTag> >("correctors"), [this](edm::InputTag const & tag){return consumes<reco::JetCorrector>(tag);}))
    , mVerbose (fConfig.getUntrackedParameter <bool> ("verbose", false))
  {
    std::string alias = fConfig.getUntrackedParameter <std::string> ("alias", "");
    if (alias.empty ())
      produces <JetCollection>();
    else
      produces <JetCollection>().setBranchAlias(alias);
  }

  template<class T>
  void CorrectedJetProducer<T>::produce(edm::StreamID, edm::Event& fEvent, const edm::EventSetup& fSetup) const
  {
    // FIXME - use something more efficient instead of an std::vector
    std::vector<reco::JetCorrector const *> correctors(mCorrectorTokens.size(), nullptr);

    // look for correctors
    for (unsigned i = 0; i < mCorrectorTokens.size(); i++)
      {
        edm::Handle <reco::JetCorrector> handle;
        fEvent.getByToken (mCorrectorTokens [i], handle);
        correctors[i] = handle.product();
      }
    edm::Handle<JetCollection> jets;                         //Define Inputs
    fEvent.getByToken (mInput, jets);                        //Get Inputs
    std::auto_ptr<JetCollection> result (new JetCollection); //Corrected jets
    typename JetCollection::const_iterator jet;
    for (jet = jets->begin(); jet != jets->end(); jet++)
      {
	const T* referenceJet = &*jet;
	int index = jet-jets->begin();
	edm::RefToBase<reco::Jet> jetRef(edm::Ref<JetCollection>(jets,index));
	T correctedJet = *jet; //copy original jet
	if (mVerbose)
	  std::cout<<"CorrectedJetProducer::produce-> original jet: "
		   <<jet->print()<<std::endl;
	for (unsigned i = 0; i < mCorrectorTokens.size(); ++i)
	  {
	    if ( !(correctors[i]->vectorialCorrection()) ) {
	      // Scalar correction
              double scale = 1.;
              if (!(correctors[i]->refRequired()))
	        scale = correctors[i]->correction (*referenceJet);
              else
                scale = correctors[i]->correction (*referenceJet,jetRef);
	      if (mVerbose)
		std::cout<<"CorrectedJetProducer::produce-> Corrector # "
			 <<i<<", correction factor: "<<scale<<std::endl;
	      correctedJet.scaleEnergy (scale); // apply scalar correction
	      referenceJet = &correctedJet;
	    } else {
	      // Vectorial correction
	      reco::JetCorrector::LorentzVector corrected;
	      double scale = correctors[i]->correction (*referenceJet, jetRef, corrected);
	      if (mVerbose)
		std::cout<<"CorrectedJetProducer::produce-> Corrector # "
			 <<i<<", correction factor: "<<scale<<std::endl;
	      correctedJet.setP4( corrected ); // apply vectorial correction
	      referenceJet = &correctedJet;
	    }
	  }
	if (mVerbose)
	  std::cout<<"CorrectedJetProducer::produce-> corrected jet: "
		   <<correctedJet.print ()<<std::endl;
	result->push_back (correctedJet);
      }
    NumericSafeGreaterByPt<T> compJets;
    // reorder corrected jets
    std::sort (result->begin (), result->end (), compJets);
    // put corrected jet collection into event
    fEvent.put(result);
  }

}

#endif
