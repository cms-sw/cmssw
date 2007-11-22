#include "CondTools/Hcal/interface/HcalPedestalsSourceHandler.h"

popcon::HcalPedestalsSourceHandler::HcalPedestalsSourceHandler(std::string name, std::string cstring, std::string cat,const edm::Event& evt, const edm::EventSetup& est, std::string pconnect) : popcon::PopConSourceHandler<HcalPedestals>(name,cstring,cat,evt,est), m_pop_connect(pconnect)
{
	m_name = name;
	m_cs = cstring;
	lgrdr = new LogReader(m_pop_connect);
	
}

popcon::HcalPedestalsSourceHandler::~HcalPedestalsSourceHandler()
{
}

void popcon::HcalPedestalsSourceHandler::getNewObjects()
{

	std::cout << "------- HCAL src - > getNewObjects\n";
	
	//check whats already inside of database
	std::map<std::string, popcon::PayloadIOV> mp = getOfflineInfo();

	for(std::map<std::string, popcon::PayloadIOV>::iterator it = mp.begin(); it != mp.end();it++)
	{
		std::cout << it->first << " , last object valid since " << it->second.last_since << std::endl;

	}

	//	coral::TimeStamp ts = lgrdr->lastRun(m_name, m_cs);
	
	unsigned int snc = edm::IOVSyncValue::beginOfTime().eventID().run();
	unsigned int tll = edm::IOVSyncValue::endOfTime().eventID().run();

//	
//	std::cerr << "Source implementation test ::getNewObjects : enter since ? \n";
//	std::cin >> snc;
//	std::cerr << "getNewObjects : enter till ? \n";
//	std::cin >> tll;


	//Using ES to get the data:

	edm::ESHandle<HcalPedestals> pedestals;
	esetup.get<HcalPedestalsRcd>().get(pedestals);
	
	HcalPedestals* mypedestals = new HcalPedestals(*(pedestals.product()));

	//	std::cout << "size " << mypedestals->getAllChannels.size() << std::endl;


	popcon::IOVPair iop = {snc,tll};

	m_to_transfer->push_back(std::make_pair((HcalPedestals*)mypedestals,iop));

	std::cout << "HCAL src - > getNewObjects -----------\n";
}
