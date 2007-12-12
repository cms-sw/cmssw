#include "CondTools/Hcal/interface/HcalQIEDataSourceHandler.h"

popcon::HcalQIEDataSourceHandler::HcalQIEDataSourceHandler(const std::string& name, const std::string& cstring, const edm::Event& evt, const edm::EventSetup& est, unsigned int sinceTime, unsigned int tillTime) 
  : popcon::PopConSourceHandler<HcalQIEData>(name,cstring,evt,est), snc(sinceTime), tll(tillTime)
{
	m_name = name;
	m_cs = cstring;
	lgrdr = new LogReader(m_pop_connect);
	
}

popcon::HcalQIEDataSourceHandler::~HcalQIEDataSourceHandler()
{
}

void popcon::HcalQIEDataSourceHandler::getNewObjects()
{

	std::cout << "------- HCAL src - > getNewObjects\n";
	
	//check whats already inside of database
	std::map<std::string, popcon::PayloadIOV> mp = getOfflineInfo();

	for(std::map<std::string, popcon::PayloadIOV>::iterator it = mp.begin(); it != mp.end();it++)
	{
		std::cout << it->first << " , last object valid since " << it->second.last_since << std::endl;

	}

	//	coral::TimeStamp ts = lgrdr->lastRun(m_name, m_cs);
	
	//	unsigned int snc = edm::IOVSyncValue::beginOfTime().eventID().run();
	//	unsigned int tll = edm::IOVSyncValue::endOfTime().eventID().run();

//	
//	std::cerr << "Source implementation test ::getNewObjects : enter since ? \n";
//	std::cin >> snc;
//	std::cerr << "getNewObjects : enter till ? \n";
//	std::cin >> tll;


	//Using ES to get the data:

	edm::ESHandle<HcalQIEData> pedestals;
	esetup.get<HcalQIEDataRcd>().get(pedestals);
	
	HcalQIEData* mypedestals = new HcalQIEData(*(pedestals.product()));

	//	std::cout << "size " << mypedestals->getAllChannels.size() << std::endl;


	popcon::IOVPair iop = {snc,tll};
	std::cout << "IOV used: " << snc << ", " << tll << std::endl;

	m_to_transfer->push_back(std::make_pair((HcalQIEData*)mypedestals,iop));

	std::cout << "HCAL src - > getNewObjects -----------\n";
}
