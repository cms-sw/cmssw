// -*- C++ -*-
//
// Package:    ImpactParameter
// Class:      ImpactParameter
// 
/**\class ImpactParameter ImpactParameter.cc RecoBTag/ImpactParameter/src/ImpactParameter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Thu Apr  6 09:56:23 CEST 2006
// $Id: ImpactParameter.cc,v 1.3 2006/06/14 17:38:04 gennai Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "RecoTauTag/ImpactParameter/interface/ImpactParameter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"

//
// constructors and destructor
//
ImpactParameter::ImpactParameter(const edm::ParameterSet& iConfig) {

	jetTrackSrc = iConfig.getParameter<string>("JetTagProd");
	vertexSrc   = iConfig.getParameter<string>("vertexSrc");
        usingVertex = iConfig.getParameter<bool>("useVertex");

	algo = new ImpactParameterAlgorithm(iConfig);

//	produces<reco::JetTagCollection>();
//	produces<reco::TauImpactParameterInfoCollection>();
	std::string modulname = iConfig.getParameter<string>( "@module_label" );
	produces<reco::JetTagCollection>().setBranchAlias(modulname);
	string infoBranchName = modulname + "Info";
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

	Handle<IsolatedTauTagInfoCollection> isolatedTaus;
	iEvent.getByLabel(jetTrackSrc,isolatedTaus);
   
	JetTagCollection                 *baseCollection = new JetTagCollection();
	TauImpactParameterInfoCollection *extCollection  = new TauImpactParameterInfoCollection();

        edm::ESHandle<TransientTrackBuilder> builder;
        iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",builder);
        algo->setTransientTrackBuilder(builder.product());

	Vertex PV;
        if(usingVertex){
	  Handle<reco::VertexCollection> vertices;
	  iEvent.getByLabel(vertexSrc,vertices);

          const reco::VertexCollection vertCollection = *(vertices.product());
          reco::VertexCollection::const_iterator iVertex;

	  for(iVertex = vertCollection.begin();iVertex!=vertCollection.end();iVertex++){
	    cout << "Vertex loop " << iVertex->z() << endl;
	    PV = *iVertex;
	  }

	}else{
	  Vertex::Error e;
	  e(0,0)=1;
	  e(1,1)=1;
	  e(2,2)=1;
	  Vertex::Point p(0,0,0);

	  Vertex dummyPV(p,e,1,1,1);
	  PV = dummyPV;
	}

	IsolatedTauTagInfoCollection::const_iterator it;
	int theKey = 0;
	for(it = isolatedTaus->begin(); it != isolatedTaus->end(); it++) {
		IsolatedTauTagInfoRef tauRef(isolatedTaus,theKey);
		pair<JetTag,TauImpactParameterInfo> ipInfo = algo->tag(tauRef,PV);

	        baseCollection->push_back(ipInfo.first);    
	        extCollection->push_back(ipInfo.second);
		theKey++;
	}


	std::auto_ptr<reco::JetTagCollection> resultBase(baseCollection);
        iEvent.put(resultBase);

	std::auto_ptr<reco::TauImpactParameterInfoCollection> resultExt(extCollection);  
	iEvent.put(resultExt);
}


