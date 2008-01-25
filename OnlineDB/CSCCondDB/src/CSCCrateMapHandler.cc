#include "OnlineDB/CSCCondDB/interface/CSCCrateMapHandler.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

popcon::CSCCrateMapImpl::CSCCrateMapImpl(const std::string& name, 
						 const std::string& cstring,
						 const edm::Event& evt, 
						 const edm::EventSetup& est, 
						 std::string& pconnect) : popcon::PopConSourceHandler<CSCCrateMap>(name,cstring,evt,est), m_pop_connect(pconnect)
{
	m_name = name;
	m_cs = cstring;
	lgrdr = new LogReader(m_pop_connect);
	
}

popcon::CSCCrateMapImpl::~CSCCrateMapImpl()
{
}

void popcon::CSCCrateMapImpl::getNewObjects()
{

	std::cout << "------- CSC src - > getNewObjects\n";
	
	//check whats already inside of database
	
	std::map<std::string, popcon::PayloadIOV> mp = getOfflineInfo();

	for(std::map<std::string, popcon::PayloadIOV>::iterator it = mp.begin(); it != mp.end();it++)
	{
		std::cout << it->first << " , last object valid since " << it->second.last_since << std::endl;

	}

	coral::TimeStamp ts = lgrdr->lastRun(m_name, m_cs);
	
	unsigned int snc,tll;
	/*
	std::cerr << "Source implementation test ::getNewObjects : enter since ? \n";
	std::cin >> snc;
	std::cerr << "getNewObjects : enter till ? \n";
	std::cin >> tll;
	*/

	//the following code works, however since 1.6.0_pre7 it causes glibc 
	//double free error (inside CSC specific code) - commented 
	//
	//Using ES to get the data:

	edm::ESHandle<CSCCrateMap> cr_map;
	esetup.get<CSCCrateMapRcd>().get(cr_map);
	mycrate_map = cr_map.product();
	std::cout << "size " << cr_map->crate_map.size() << std::endl;

	snc = edm::IOVSyncValue::beginOfTime().eventID().run();
	tll = edm::IOVSyncValue::endOfTime().eventID().run();//infinite
	//snc =1;
	//tll =10;

	popcon::IOVPair iop = {snc,tll};
	//popcon::IOVPair iop2 = {snc+20,tll};
	//popcon::IOVPair iop3 = {snc+10,tll};

	//CSCCrateMap * p1 = new CSCCrateMap(*mycrate_map);
	//CSCCrateMap * p2 = new CSCCrateMap(*mycrate_map);


	m_to_transfer->push_back(std::make_pair((CSCCrateMap*)mycrate_map,iop));
	//m_to_transfer->push_back(std::make_pair((CSCCrateMap*)p1,iop2));
	//m_to_transfer->push_back(std::make_pair((CSCCrateMap*)p2,iop3));

	std::cout << "CSC src - > getNewObjects -----------\n";
}
