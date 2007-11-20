#include "CondCore/DBOutputService/interface/PopConDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

popcon::service::PopConDBOutputService::PopConDBOutputService(const edm::ParameterSet & iConfig,edm::ActivityRegistry & iAR ) : cond::service::PoolDBOutputService::PoolDBOutputService(iConfig,iAR)
{

	typedef std::vector< edm::ParameterSet > Parameters;
	Parameters toPut=iConfig.getParameter<Parameters>("toPut");
	for(Parameters::iterator itToPut = toPut.begin(); itToPut != toPut.end(); ++itToPut) {
		m_record = itToPut->getParameter<std::string>("record");
		m_tag = itToPut->getParameter<std::string>("tag");

	}


}

std::string popcon::service::PopConDBOutputService::getTag()
{
	return m_tag;
}
std::string popcon::service::PopConDBOutputService::getRecord()
{
	return m_record;
}


