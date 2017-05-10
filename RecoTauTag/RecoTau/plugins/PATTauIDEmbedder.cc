#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/PATTauDiscriminator.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "FWCore/Utilities/interface/transform.h"

class PATTauIDEmbedder : public edm::EDProducer
{
public:

	explicit PATTauIDEmbedder(const edm::ParameterSet&);
	~PATTauIDEmbedder(){};

	void produce(edm::Event&, const edm::EventSetup&);

private:

//--- configuration parameters
	edm::EDGetTokenT<pat::TauCollection> src_;
	typedef std::pair<std::string, edm::InputTag> NameTag;
	std::vector<NameTag> tauIDSrcs_;
	std::vector<edm::EDGetTokenT<pat::PATTauDiscriminator> > patTauIDTokens_;
};

PATTauIDEmbedder::PATTauIDEmbedder(const edm::ParameterSet& cfg)
{
	src_ = consumes<pat::TauCollection>(cfg.getParameter<edm::InputTag>("src"));
	// it might be a single tau ID
	if (cfg.existsAs<edm::InputTag>("tauIDSource")) {
		tauIDSrcs_.push_back(NameTag("", cfg.getParameter<edm::InputTag>("tauIDSource")));
	}
	// or there might be many of them
	if (cfg.existsAs<edm::ParameterSet>("tauIDSources")) {
		// please don't configure me twice
		if (!tauIDSrcs_.empty()){
			throw cms::Exception("Configuration") << "PATTauProducer: you can't specify both 'tauIDSource' and 'tauIDSources'\n";
		}
		// read the different tau ID names
		edm::ParameterSet idps = cfg.getParameter<edm::ParameterSet>("tauIDSources");
		std::vector<std::string> names = idps.getParameterNamesForType<edm::InputTag>();
		for (std::vector<std::string>::const_iterator it = names.begin(), ed = names.end(); it != ed; ++it) {
			tauIDSrcs_.push_back(NameTag(*it, idps.getParameter<edm::InputTag>(*it)));
		}
	}
	// but in any case at least once
	if (tauIDSrcs_.empty()) throw cms::Exception("Configuration") <<
		"PATTauProducer: id addTauID is true, you must specify either:\n" <<
		"\tInputTag tauIDSource = <someTag>\n" << "or\n" <<
		"\tPSet tauIDSources = { \n" <<
		"\t\tInputTag <someName> = <someTag>   // as many as you want \n " <<
		"\t}\n";
	patTauIDTokens_ = edm::vector_transform(tauIDSrcs_, [this](NameTag const & tag){return mayConsume<pat::PATTauDiscriminator>(tag.second);});

	produces<std::vector<pat::Tau> >();
}

void PATTauIDEmbedder::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<pat::TauCollection> inputTaus;
  evt.getByToken(src_, inputTaus);
  
  auto outputTaus = std::make_unique<std::vector<pat::Tau> >();
  outputTaus->reserve(inputTaus->size());
  
  int tau_idx = 0;
  for(pat::TauCollection::const_iterator inputTau = inputTaus->begin(); inputTau != inputTaus->end(); ++inputTau, ++tau_idx){
    pat::Tau outputTau(*inputTau);
    pat::TauRef inputTauRef(inputTaus, tau_idx);
    size_t nTauIds = inputTau->tauIDs().size();
    std::vector<pat::Tau::IdPair> tauIds(nTauIds+tauIDSrcs_.size());
    
    for(size_t i = 0; i < nTauIds; ++i){
      tauIds[i] = inputTau->tauIDs().at(i);
    }
    
    for(size_t i = 0; i < tauIDSrcs_.size(); ++i){
      edm::Handle<pat::PATTauDiscriminator> tauDiscr;
      evt.getByToken(patTauIDTokens_[i], tauDiscr);
      
      tauIds[nTauIds+i].first = tauIDSrcs_[i].first;
      tauIds[nTauIds+i].second = (*tauDiscr)[inputTauRef];
    }
    
    outputTau.setTauIDs(tauIds);
    outputTaus->push_back(outputTau);
  }
  
  evt.put(std::move(outputTaus));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATTauIDEmbedder);
