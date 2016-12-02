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
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "DataFormats/Common/interface/RefToBase.h"

 #include "DataFormats/Common/interface/View.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "DataFormats/BTauReco/interface/JetTag.h"


#include <map>

using namespace reco;

//
// class declaration
//

inline bool equals(const edm::RefToBase<Jet> &j1, const edm::RefToBase<Jet> &j2) {
	return j1.id() == j2.id() && j1.key() == j2.key();
}

class DeepCMVATagInfoProducer : public edm::stream::EDProducer<> {
public:
	explicit DeepCMVATagInfoProducer(const edm::ParameterSet&);
	~DeepCMVATagInfoProducer();

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
	virtual void beginStream(edm::StreamID) override {}
	virtual void produce(edm::Event&, const edm::EventSetup&) override;
	virtual void endStream() override {}

	// ----------member data ---------------------------
	const edm::EDGetTokenT< std::vector<reco::ShallowTagInfo> > deepNNSrc_;
	const edm::EDGetTokenT< edm::View<reco::BaseTagInfo> > ipInfoSrc_;
	const edm::EDGetTokenT< edm::View<reco::BaseTagInfo> > muInfoSrc_;
	const edm::EDGetTokenT< edm::View<reco::BaseTagInfo> > elInfoSrc_;
	/*const edm::EDGetTokenT< std::vector<reco::CandIPTagInfo>  > ipInfoSrc_;
	const edm::EDGetTokenT< std::vector<reco::CandSoftLeptonTagInfo> > muInfoSrc_;
	const edm::EDGetTokenT< std::vector<reco::CandSoftLeptonTagInfo> > elInfoSrc_;*/
	std::string jpComputer_, jpbComputer_, softmuComputer_, softelComputer_;
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
DeepCMVATagInfoProducer::DeepCMVATagInfoProducer(const edm::ParameterSet& iConfig) :
	deepNNSrc_(consumes< std::vector<reco::ShallowTagInfo> >(iConfig.getParameter<edm::InputTag>("deepNNTagInfos"))),
	ipInfoSrc_(consumes< edm::View<reco::BaseTagInfo> >(iConfig.getParameter<edm::InputTag>("ipInfoSrc"))),
	muInfoSrc_(consumes< edm::View<reco::BaseTagInfo> >(iConfig.getParameter<edm::InputTag>("muInfoSrc"))),
	elInfoSrc_(consumes< edm::View<reco::BaseTagInfo> >(iConfig.getParameter<edm::InputTag>("elInfoSrc"))),
	/*ipInfoSrc_(consumes< std::vector<reco::CandIPTagInfo>  >(iConfig.getParameter<edm::InputTag>("ipInfoSrc"))),
	elInfoSrc_(consumes< std::vector<reco::CandSoftLeptonTagInfo> >(iConfig.getParameter<edm::InputTag>("elInfoSrc"))),
	elInfoSrc_(consumes< std::vector<reco::CandSoftLeptonTagInfo> >(iConfig.getParameter<edm::InputTag>("elInfoSrc"))),*/
	
	jpComputer_(iConfig.getParameter<std::string>("jpComputerSrc")), 
	jpbComputer_(iConfig.getParameter<std::string>("jpbComputerSrc")), 
	softmuComputer_(iConfig.getParameter<std::string>("softmuComputerSrc")), 
	softelComputer_(iConfig.getParameter<std::string>("softelComputerSrc"))
{

	produces<std::vector<reco::ShallowTagInfo> >();
	
	/*uses(0, "ipTagInfos");
  	uses(1, "smTagInfos");
	uses(2, "seTagInfos");*/
	
}


DeepCMVATagInfoProducer::~DeepCMVATagInfoProducer()
{

	// do anything here that needs to be done at destruction time
	// (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DeepCMVATagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{	
	// get input TagInfos from DeepCSV
	edm::Handle< std::vector<reco::ShallowTagInfo> > nnInfos;
	iEvent.getByToken(deepNNSrc_, nnInfos);
	
	/*
	// TagInfos for JP taggers
  	std::vector<const BaseTagInfo*> ipInfos({ &info.getBase(0) });
  	// TagInfos for the Softelon tagger
  	std::vector<const BaseTagInfo*> elInfos({ &info.getBase(1) });
  	// TagInfos for the SoftElectron tagger
	std::vector<const BaseTagInfo*> elInfos({ &info.getBase(2) });
	*/
	//edm::Handle< std::vector<reco::CandIPTagInfo>  > ipInfos;
	edm::Handle< edm::View<BaseTagInfo> > ipInfos;
	iEvent.getByToken(ipInfoSrc_, ipInfos);
	std::vector<const BaseTagInfo*> ipBaseInfo;
	for(edm::View<BaseTagInfo>::const_iterator iter = ipInfos->begin(); iter != ipInfos->end(); iter++) {
		ipBaseInfo.push_back(&*iter);
	}
	//std::vector< const reco::BaseTagInfo *> ipInfo = static_cast<std::vector< const reco::BaseTagInfo *> >(*ipInfos)
	//edm::Handle< std::vector<reco::CandSoftLeptonTagInfo> > elInfos;
	//edm::Handle< edm::View<BaseTagInfo> > elInfos;
	//iEvent.getByToken(elInfoSrc_, elInfos);
	edm::Handle< edm::View<BaseTagInfo> > muInfos;
	iEvent.getByToken(muInfoSrc_, muInfos);
	std::vector<const BaseTagInfo*> muBaseInfo;
	for(edm::View<BaseTagInfo>::const_iterator iter = muInfos->begin(); iter != muInfos->end(); iter++) {
		muBaseInfo.push_back(&*iter);
	}
	//edm::Handle< std::vector<reco::CandSoftLeptonTagInfo> > elInfos;
	//edm::Handle< edm::View<BaseTagInfo> > elInfos;
	//iEvent.getByToken(elInfoSrc_, elInfos);
	edm::Handle< edm::View<BaseTagInfo> > elInfos;
	iEvent.getByToken(elInfoSrc_, elInfos);
	std::vector<const BaseTagInfo*> elBaseInfo;
	for(edm::View<BaseTagInfo>::const_iterator iter = elInfos->begin(); iter != elInfos->end(); iter++) {
		elBaseInfo.push_back(&*iter);
	}
	

	//get computers
	edm::ESHandle<JetTagComputer> jp;
	iSetup.get<JetTagComputerRecord>().get(jpComputer_,jp);
	const JetTagComputer* compjp = jp.product();
	edm::ESHandle<JetTagComputer> jpb;
	iSetup.get<JetTagComputerRecord>().get(jpbComputer_,jpb);
	const JetTagComputer* compjpb = jpb.product();
	edm::ESHandle<JetTagComputer> softmu;
	iSetup.get<JetTagComputerRecord>().get(softmuComputer_,softmu);
	const JetTagComputer* compsoftmu = softmu.product();
	edm::ESHandle<JetTagComputer> softel;
	iSetup.get<JetTagComputerRecord>().get(softelComputer_,softel);
	const JetTagComputer* compsoftel = softel.product();

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
		
		// Copy the DeepNN TaggingVariables + add the other discriminators
		TaggingVariableList vars = nnInfo.taggingVariables();
		vars.insert(reco::btau::Jet_SoftMu, (*compsoftmu)( JetTagComputer::TagInfoHelper(muBaseInfo) )  );
		vars.insert(reco::btau::Jet_SoftEl, (*compsoftel)( JetTagComputer::TagInfoHelper(elBaseInfo) )  );
		vars.insert(reco::btau::Jet_JBP, (*compjpb)( JetTagComputer::TagInfoHelper(ipBaseInfo) )  );
		vars.insert(reco::btau::Jet_JP, (*compjp)( JetTagComputer::TagInfoHelper(ipBaseInfo) )  ); 
		vars.finalize();
		tagInfos->emplace_back(vars, nnInfo.jet());
		

	}

	// put the output in the event
	iEvent.put( std::move(tagInfos) );
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DeepCMVATagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepCMVATagInfoProducer);
