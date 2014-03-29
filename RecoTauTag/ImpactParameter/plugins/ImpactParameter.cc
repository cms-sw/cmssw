/* class ImpactParameter
 * EDProducer for TauImpactParameterInfoCollection
 * author: Andrea Rizzi
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoTauTag/ImpactParameter/interface/ImpactParameterAlgorithm.h"


class ImpactParameter : public edm::EDProducer {
public:
  explicit ImpactParameter(const edm::ParameterSet&);
  ~ImpactParameter();


  virtual void produce(edm::Event&, const edm::EventSetup&);
private:
  ImpactParameterAlgorithm* algo;
  edm::InputTag jetTrackSrc;
  edm::InputTag vertexSrc;
  bool usingVertex;
};


//
// constructors and destructor
//
ImpactParameter::ImpactParameter(const edm::ParameterSet& iConfig) {

        jetTrackSrc = iConfig.getParameter<edm::InputTag>("JetTagProd");
        vertexSrc   = iConfig.getParameter<edm::InputTag>("vertexSrc");
        usingVertex = iConfig.getParameter<bool>("useVertex");

        algo = new ImpactParameterAlgorithm(iConfig);

        std::string modulname = iConfig.getParameter<std::string>( "@module_label" );
        produces<reco::JetTagCollection>().setBranchAlias(modulname);
        std::string infoBranchName = modulname + "Info";
        produces<reco::TauImpactParameterInfoCollection>().setBranchAlias(infoBranchName);
}


ImpactParameter::~ImpactParameter(){
        delete algo;
}



//
// member functions
//
// ------------ method called to produce the data  ------------
void ImpactParameter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

        using namespace reco;

        edm::Handle<IsolatedTauTagInfoCollection> isolatedTaus;
        iEvent.getByLabel(jetTrackSrc,isolatedTaus);

        std::auto_ptr<JetTagCollection>                 tagCollection;
        std::auto_ptr<TauImpactParameterInfoCollection> extCollection( new TauImpactParameterInfoCollection() );
        if (not isolatedTaus->empty()) {
          edm::RefToBaseProd<reco::Jet> prod( isolatedTaus->begin()->jet() );
          tagCollection.reset( new JetTagCollection(prod) );
        } else {
          tagCollection.reset( new JetTagCollection() );
        }

        edm::ESHandle<TransientTrackBuilder> builder;
        iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",builder);
        algo->setTransientTrackBuilder(builder.product());

        Vertex PV;
        if (usingVertex) {
          edm::Handle<reco::VertexCollection> vertices;
          iEvent.getByLabel(vertexSrc,vertices);

          const reco::VertexCollection vertCollection = *(vertices.product());
          reco::VertexCollection::const_iterator iVertex;

          for(iVertex = vertCollection.begin();iVertex!=vertCollection.end();iVertex++){
            PV = *iVertex;
          }

        } else {
          Vertex::Error e;
          e(0,0)=0;
          e(1,1)=0;
          e(2,2)=0;
          Vertex::Point p(0,0,0);

          Vertex dummyPV(p,e,1,1,1);
          PV = dummyPV;
        }

        for (unsigned int i = 0; i < isolatedTaus->size(); ++i) {
            IsolatedTauTagInfoRef tauRef(isolatedTaus, i);
            std::pair<float, TauImpactParameterInfo> ipInfo = algo->tag(tauRef,PV);
            tagCollection->setValue(i, ipInfo.first);    
            extCollection->push_back(ipInfo.second);
        }

        iEvent.put(extCollection);
        iEvent.put(tagCollection);
}


DEFINE_FWK_MODULE(ImpactParameter);
