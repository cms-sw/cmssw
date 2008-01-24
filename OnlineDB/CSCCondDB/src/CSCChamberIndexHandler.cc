#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexHandler.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

popcon::CSCChamberIndexImpl::CSCChamberIndexImpl(const std::string& name, 
						 const std::string& cstring,
						 const edm::Event& evt, 
						 const edm::EventSetup& est, 
						 std::string& pconnect) : popcon::PopConSourceHandler<CSCChamberIndex>(name,cstring,evt,est), m_pop_connect(pconnect)
{
	m_name = name;
	m_cs = cstring;
	lgrdr = new LogReader(m_pop_connect);
	
}

popcon::CSCChamberIndexImpl::~CSCChamberIndexImpl()
{
}

void popcon::CSCChamberIndexImpl::getNewObjects()
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

	edm::ESHandle<CSCChamberIndex> cham_index;
	esetup.get<CSCChamberIndexRcd>().get(cham_index);
	mycham_index = cham_index.product();
	std::cout << "size " << cham_index->ch_index.size() << std::endl;

	snc = edm::IOVSyncValue::beginOfTime().eventID().run();
	tll = edm::IOVSyncValue::endOfTime().eventID().run();//infinite
	//snc =1;
	//tll =10;

	popcon::IOVPair iop = {snc,tll};
	//popcon::IOVPair iop2 = {snc+20,tll};
	//popcon::IOVPair iop3 = {snc+10,tll};

	//CSCChamberIndex * p1 = new CSCChamberIndex(*mycham_index);
	//CSCChamberIndex * p2 = new CSCChamberIndex(*mycham_index);


	m_to_transfer->push_back(std::make_pair((CSCChamberIndex*)mycham_index,iop));
	//m_to_transfer->push_back(std::make_pair((CSCChamberIndex*)p1,iop2));
	//m_to_transfer->push_back(std::make_pair((CSCChamberIndex*)p2,iop3));

	std::cout << "CSC src - > getNewObjects -----------\n";
}
