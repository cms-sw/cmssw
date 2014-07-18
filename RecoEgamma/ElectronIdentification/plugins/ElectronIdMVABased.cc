// -*- C++ -*-
//
// Package:    ElectronIdMVABased
// Class:      ElectronIdMVABased
//
/**\class ElectronIdMVABased ElectronIdMVABased.cc MyAnalyzer/ElectronIdMVABased/src/ElectronIdMVABased.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Zablocki Jakub
//         Created:  Thu Feb  9 10:47:50 CST 2012
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimator.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//
// class declaration
//

using namespace std;
using namespace reco;
class ElectronIdMVABased : public edm::stream::EDFilter<> {
	public:
		explicit ElectronIdMVABased(const edm::ParameterSet&);
		~ElectronIdMVABased();

	private:
		virtual bool filter(edm::Event&, const edm::EventSetup&) override;


		// ----------member data ---------------------------
		edm::EDGetTokenT<reco::VertexCollection> vertexToken;
		edm::EDGetTokenT<reco::GsfElectronCollection> electronToken;
                string mvaWeightFileEleID;
                string path_mvaWeightFileEleID;
		double thresholdBarrel;
		double thresholdEndcap;
                double thresholdIsoBarrel;
                double thresholdIsoEndcap;

		ElectronMVAEstimator *mvaID_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ElectronIdMVABased::ElectronIdMVABased(const edm::ParameterSet& iConfig) {
	vertexToken = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexTag"));
	electronToken = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electronTag"));
	mvaWeightFileEleID = iConfig.getParameter<string>("HZZmvaWeightFile");
	thresholdBarrel = iConfig.getParameter<double>("thresholdBarrel");
	thresholdEndcap = iConfig.getParameter<double>("thresholdEndcap");
	thresholdIsoBarrel = iConfig.getParameter<double>("thresholdIsoDR03Barrel");
	thresholdIsoEndcap = iConfig.getParameter<double>("thresholdIsoDR03Endcap");

	produces<reco::GsfElectronCollection>();
	path_mvaWeightFileEleID = edm::FileInPath ( mvaWeightFileEleID.c_str() ).fullPath();
	FILE * fileEleID = fopen(path_mvaWeightFileEleID.c_str(), "r");
	if (fileEleID) {
	  fclose(fileEleID);
	}
	else {
	  string err = "ElectronIdMVABased: cannot open weight file '";
	  err += path_mvaWeightFileEleID;
	  err += "'";
	  throw invalid_argument( err );
	}

	mvaID_ = new ElectronMVAEstimator(path_mvaWeightFileEleID);
}


ElectronIdMVABased::~ElectronIdMVABased()
{

  delete mvaID_;
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool ElectronIdMVABased::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
	using namespace edm;

	std::auto_ptr<reco::GsfElectronCollection> mvaElectrons(new reco::GsfElectronCollection);

	Handle<reco::VertexCollection>  vertexCollection;
	iEvent.getByToken(vertexToken, vertexCollection);
	int nVtx = vertexCollection->size();

	Handle<reco::GsfElectronCollection> egCollection;
	iEvent.getByToken(electronToken,egCollection);
	const reco::GsfElectronCollection egCandidates = (*egCollection.product());
	for ( reco::GsfElectronCollection::const_iterator egIter = egCandidates.begin(); egIter != egCandidates.end(); ++egIter) {
	  double mvaVal = mvaID_->mva( *egIter, nVtx );
	  double isoDr03 = egIter->dr03TkSumPt() + egIter->dr03EcalRecHitSumEt() + egIter->dr03HcalTowerSumEt();
	  double eleEta = fabs(egIter->eta());
	  if (eleEta <= 1.485 && mvaVal > thresholdBarrel && isoDr03 < thresholdIsoBarrel) {
	    mvaElectrons->push_back( *egIter );
	    reco::GsfElectron::MvaOutput myMvaOutput;
	    myMvaOutput.mva = mvaVal;
	    mvaElectrons->back().setMvaOutput(myMvaOutput);
	  }
	  else if (eleEta > 1.485 && mvaVal > thresholdEndcap  && isoDr03 < thresholdIsoEndcap) {
	    mvaElectrons->push_back( *egIter );
	    reco::GsfElectron::MvaOutput myMvaOutput;
	    myMvaOutput.mva = mvaVal;
	    mvaElectrons->back().setMvaOutput(myMvaOutput);
	  }


	}


	iEvent.put(mvaElectrons);

	return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronIdMVABased);
