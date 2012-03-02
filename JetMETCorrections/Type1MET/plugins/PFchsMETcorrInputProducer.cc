#include "JetMETCorrections/Type1MET/plugins/PFchsMETcorrInputProducer.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

PFchsMETcorrInputProducer::PFchsMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    src_(cfg.getParameter<edm::InputTag>("src")),
    goodVtxNdof_(cfg.getParameter<unsigned int>("goodVtxNdof")),
    goodVtxZ_(cfg.getParameter<double>("goodVtxZ"))
{
  produces<CorrMETData>("type0");
}

PFchsMETcorrInputProducer::~PFchsMETcorrInputProducer()
{

}

void PFchsMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<reco::VertexCollection> recVtxs;
  evt.getByLabel(src_, recVtxs);

  std::auto_ptr<CorrMETData> chsSum(new CorrMETData());

  for (unsigned i = 1; i < recVtxs->size(); ++i)
    {
      const reco::Vertex& v = recVtxs->at(i);
      if (v.isFake()) continue;
      if (v.ndof() < goodVtxNdof_) continue;
      if (fabs(v.z()) > goodVtxZ_) continue;

      for (reco::Vertex::trackRef_iterator track = v.tracks_begin(); track != v.tracks_end(); ++track)
	{
	  if ((*track)->charge() != 0) 
	    {
	      chsSum->mex += (*track)->px();
	      chsSum->mey += (*track)->py();
	      chsSum->sumet += (*track)->pt();
	    }
	}
    }

  evt.put(chsSum, "type0");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFchsMETcorrInputProducer);
