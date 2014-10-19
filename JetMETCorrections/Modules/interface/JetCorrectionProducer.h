#ifndef JetCorrectionProducer_h
#define JetCorrectionProducer_h

#include <sstream>
#include <string>
#include <vector>

#include "CommonTools/Utils/interface/PtComparator.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "FWCore/Framework/interface/EDProducer.h"
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

namespace cms
{
  template<class T>
  class JetCorrectionProducer : public edm::EDProducer
  {
  public:
    typedef std::vector<T> JetCollection;
    explicit JetCorrectionProducer (const edm::ParameterSet& fParameters);
    virtual ~JetCorrectionProducer () {}
    virtual void produce(edm::Event&, const edm::EventSetup&);
  private:
    edm::EDGetTokenT<JetCollection> mInput;
    std::vector <edm::EDGetTokenT<reco::JetCorrector> > mCorrectorTokens;
    // cache
    std::vector <const reco::JetCorrector*> mCorrectors;
    bool mVerbose;
  };
}

// ---------- implementation ----------

namespace cms {

  template<class T>
  JetCorrectionProducer<T>::JetCorrectionProducer(const edm::ParameterSet& fConfig)
    : mInput(consumes<JetCollection>(fConfig.getParameter <edm::InputTag> ("src")))
    , mCorrectorTokens(edm::vector_transform(fConfig.getParameter<std::vector<edm::InputTag> >("correctors"), [this](edm::InputTag const & tag){return consumes<reco::JetCorrector>(tag);}))
    , mCorrectors(mCorrectorTokens.size(), 0)
    , mVerbose (fConfig.getUntrackedParameter <bool> ("verbose", false))
  {
    std::string alias = fConfig.getUntrackedParameter <std::string> ("alias", "");
    if (alias.empty ())
      produces <JetCollection>();
    else
      produces <JetCollection>().setBranchAlias(alias);
  }

  template<class T>
  void JetCorrectionProducer<T>::produce(edm::Event& fEvent,
					 const edm::EventSetup& fSetup)
  {
    // look for correctors
    for (unsigned i = 0; i < mCorrectorTokens.size(); i++)
      {
        edm::Handle <reco::JetCorrector> handle;
        fEvent.getByToken (mCorrectorTokens [i], handle);
        mCorrectors [i] = &*handle;
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
	  std::cout<<"JetCorrectionProducer::produce-> original jet: "
		   <<jet->print()<<std::endl;
	for (unsigned i = 0; i < mCorrectors.size(); ++i)
	  {
	    if ( !(mCorrectors[i]->vectorialCorrection()) ) {
	      // Scalar correction
              double scale = 1.;
              if (!(mCorrectors[i]->refRequired()))
	        scale = mCorrectors[i]->correction (*referenceJet);
              else
                scale = mCorrectors[i]->correction (*referenceJet,jetRef);
	      if (mVerbose)
		std::cout<<"JetCorrectionProducer::produce-> Corrector # "
			 <<i<<", correction factor: "<<scale<<std::endl;
	      correctedJet.scaleEnergy (scale); // apply scalar correction
	      referenceJet = &correctedJet;
	    } else {
	      // Vectorial correction
	      reco::JetCorrector::LorentzVector corrected;
	      double scale = mCorrectors[i]->correction (*referenceJet, jetRef, corrected);
	      if (mVerbose)
		std::cout<<"JetCorrectionProducer::produce-> Corrector # "
			 <<i<<", correction factor: "<<scale<<std::endl;
	      correctedJet.setP4( corrected ); // apply vectorial correction
	      referenceJet = &correctedJet;
	    }
	  }
	if (mVerbose)
	  std::cout<<"JetCorrectionProducer::produce-> corrected jet: "
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
