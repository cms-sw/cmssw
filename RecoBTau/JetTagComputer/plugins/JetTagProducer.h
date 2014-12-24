#ifndef RecoBTag_JetTagComputer_JetTagProducer_h
#define RecoBTag_JetTagComputer_JetTagProducer_h

// system include files
#include <string>
#include <vector>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"

class JetTagProducer : public edm::stream::EDProducer<> {
    public:
	explicit JetTagProducer(const edm::ParameterSet&);
	~JetTagProducer();
	static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:
	virtual void produce(edm::Event&, const edm::EventSetup&);

	std::string				m_jetTagComputer;
	std::vector<edm::EDGetTokenT<edm::View<reco::BaseTagInfo> > > token_tagInfos;
	unsigned int nTagInfos;
        edm::ESWatcher<JetTagComputerRecord> recordWatcher_;
};

#endif // RecoBTag_JetTagComputer_JetTagProducer_h
