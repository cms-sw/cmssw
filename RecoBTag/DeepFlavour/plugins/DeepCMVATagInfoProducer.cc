// -*- C++ -*-
//
// Package:    ​RecoBTag/​SecondaryVertex
// Class:      DeepNNTagInfoProducer
//
/**\class DeepNNTagInfoProducer DeepNNTagInfoProducer.cc ​RecoBTag/DeepFlavour/plugins/DeepNNTagInfoProducer.cc
 *
 * Description: EDProducer that produces collection of DeepNNTagInfos
 *
 * Implementation:
 *    A collection of CandIPTagInfo and CandSecondaryVertexTagInfo and a CombinedSVComputer ESHandle is taken as input and a collection of DeepNNTagInfos
 *    is produced as output.
 */
//
// Original Author:  Mauro Verzetti (U. Rochester)
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"

#include <map>

//
// class declaration
//

inline bool equals(const RefToBase<Jet> &j1, const RefToBase<Jet> &j2) {
	return j1.id() == j2.id() && j1.key() == j2.key();
}

class DeepNNTagInfoProducer : public edm::stream::EDProducer<> {
public:
	explicit DeepNNTagInfoProducer(const edm::ParameterSet&);
	~DeepNNTagInfoProducer();

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
	virtual void beginStream(edm::StreamID) override {}
	virtual void produce(edm::Event&, const edm::EventSetup&) override;
	virtual void endStream() override {}

	// ----------member data ---------------------------
	const edm::EDGetTokenT< std::vector<reco::ShallowTagInfo> > deepNNSrc_;
	const edm::EDGetTokenT< std::vector<reco::CandIPTagInfo>  > ipInfoSrc_;
	const edm::EDGetTokenT< std::vector<reco::CandSoftLeptonTagInfo> > muInfoSrc_;
	const edm::EDGetTokenT< std::vector<reco::CandSoftLeptonTagInfo> > elInfoSrc_;	
	edm::InputTag jpComputer_, jpbComputer_, softmuComputer_, softelComputer_;	
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
DeepNNTagInfoProducer::DeepNNTagInfoProducer(const edm::ParameterSet& iConfig) :
	deepNNSrc_(consumes< std::vector<reco::ShallowTagInfo> >(iConfig.getParameter<edm::InputTag>("deepNNSrc"))),
	ipInfoSrc_(consumes< std::vector<reco::CandIPTagInfo>  >(iConfig.getParameter<edm::InputTag>("ipInfoSrc"))),
	muInfoSrc_(consumes< std::vector<reco::CandSoftLeptonTagInfo> >(iConfig.getParameter<edm::InputTag>("muInfoSrc"))),
	elInfoSrc_(consumes< std::vector<reco::CandSoftLeptonTagInfo> >(iConfig.getParameter<edm::InputTag>("elInfoSrc"))),
	jpComputer_(iConfig.getParameter<edm::InputTag>("jpComputer")), 
	jpbComputer_(iConfig.getParameter<edm::InputTag>("jpbComputer")), 
	softmuComputer_(iConfig.getParameter<edm::InputTag>("softmuComputer")), 
	softelComputer_(iConfig.getParameter<edm::InputTag>("softelComputer"))
{
	produces<std::vector<reco::ShallowTagInfo> >();
}


DeepNNTagInfoProducer::~DeepNNTagInfoProducer()
{

	// do anything here that needs to be done at destruction time
	// (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DeepNNTagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	// get input TagInfos
	edm::Handle< std::vector<reco::ShallowTagInfo> > nnInfos;
	iEvent.getByToken(deepNNSrc_, nnInfos);
	edm::Handle< std::vector<reco::CandIPTagInfo>  > ipInfos;
	iEvent.getByToken(ipInfoSrc_, ipInfos);
	edm::Handle< std::vector<reco::CandSoftLeptonTagInfo> > muInfos;
	iEvent.getByToken(muInfoSrc_, muInfos);
	edm::Handle< std::vector<reco::CandSoftLeptonTagInfo> > elInfos;
	iEvent.getByToken(elInfoSrc_, elInfos);	

	//get computers
	edm::ESHandle<JetTagComputer> jp;
	iSetup.get<JetTagComputer>(jpComputer_).get(jp);
	edm::ESHandle<JetTagComputer> jpb;
	iSetup.get<JetTagComputer>(jpbComputer_).get(jpb);
	edm::ESHandle<JetTagComputer> softmu;
	iSetup.get<JetTagComputer>(softmuComputer_).get(softmu);
	edm::ESHandle<JetTagComputer> softel;
	iSetup.get<JetTagComputer>(softelComputer_).get(softel);

	// create the output collection
	auto tagInfos = std::make_unique<std::vector<reco::ShallowTagInfo> >();

	// loop over TagInfos, assume they are ordered in the same way, check later and throw exception if not
	for(size_t idx = 0; idx<nnInfos->size(); ++idx) {
		auto& nnInfo = nnInfos->at(idx);
		auto& ipInfo = ipInfos->at(idx);
		auto& muInfo = muInfos->at(idx);
		auto& elInfo = elInfos->at(idx);
		
		if(
			!equals(nnInfo.jet(), ipInfo.jet()) ||
			!equals(nnInfo.jet(), muInfo.jet()) ||
			!equals(nnInfo.jet(), elInfo.jet())
			) {
			throw cms::Exception("ValueError") << "DeepNNTagInfoProducer::produce: The tagInfos taken belong to different jets!" << std::endl 
																				 << "This could be due to: " << std::endl
																				 << "  - You passed tagInfos computed on different jet collection" << std::endl
																				 << "  - The assumption that the tagInfos are filled in the same order is actually wrong" << std::endl;
		}

		
		tagInfos->emplace_back(
			vars, 
			svTagInfo.jet()
			);
	}

	// put the output in the event
	iEvent.put( std::move(tagInfos) );
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DeepNNTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepNNTagInfoProducer);
