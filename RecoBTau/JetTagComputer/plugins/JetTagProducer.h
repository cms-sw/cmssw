#ifndef RecoBTag_JetTagComputer_JetTagProducer_h
#define RecoBTag_JetTagComputer_JetTagProducer_h

// system include files
#include <string>
#include <vector>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

class JetTagProducer : public edm::EDProducer {
    public:
	explicit JetTagProducer(const edm::ParameterSet&);
	~JetTagProducer();

    private:
	virtual void produce(edm::Event&, const edm::EventSetup&);

	void setup(const edm::EventSetup&);

	const JetTagComputer			*m_computer;
	std::string				m_jetTagComputer;
	std::vector<edm::InputTag>		m_tagInfos;
};

#endif // RecoBTag_JetTagComputer_JetTagProducer_h
