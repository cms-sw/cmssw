#include <vector>

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Provenance/interface/Provenance.h"
#include "ArbitraryType.h"

class RequireModule : public edm::EDFilter {
	public:
		explicit RequireModule(const edm::ParameterSet& config);
		virtual ~RequireModule();

		bool filter(edm::Event & event, const edm::EventSetup & setup);

	private:
		// input collections
		edm::InputTag m_requirement;
};


RequireModule::RequireModule(const edm::ParameterSet & config) :
	m_requirement( config.getParameter<edm::InputTag>("requirement") )
{
}

RequireModule::~RequireModule() 
{
}

bool RequireModule::filter(edm::Event & event, const edm::EventSetup & setup) 
{
	bool found = false;

	std::vector<const edm::Provenance *> provenances;
	event.getAllProvenance(provenances);

	/////// edm::ArbitraryHandle handle;
	//   std::cout<<"my moduleLabel "<< m_requirement.label() <<std::endl;
	//  std::cout<<"my instancec "<< m_requirement.instance()<<std::endl;
	//  std::cout<<"my process "<< m_requirement.process() <<std::endl;

	for (unsigned int i = 0; i < provenances.size(); ++i) {
		//  std::cout<<"i="<<i<<std::endl;
		// std::cout<<"moduleLabel "<< provenances[i]->moduleLabel() <<std::endl;
		//  std::cout<<"instancec "<< provenances[i]->productInstanceName()<<std::endl;
		//  std::cout<<"process "<< provenances[i]->processName() <<std::endl;
		/////// event.get(provenances[i]->productID(), handle);
		/////// std::cout<<"valid "<< handle.isValid()<<std::endl;

		if ((m_requirement.label()    == provenances[i]->moduleLabel()) && 
				(m_requirement.instance() == provenances[i]->productInstanceName()) && 
				(m_requirement.process()  == provenances[i]->processName() || m_requirement.process()  == "") /////// &&
				/////// event.get(provenances[i]->productID(), handle) &&         handle.isValid()
		   ) {

			//   std::cout<<"Found "<<std::endl;
			found = true;
			break;
		}  
	}

	if (! found) std::cout<<"not Found "<<std::endl;
	return found;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RequireModule);



