#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkHandler.h"

popcon::CSCDBCrosstalkImpl::CSCDBCrosstalkImpl(std::string name, std::string cstring, std::string cat,const edm::Event& evt, const edm::EventSetup& est, std::string pconnect) : popcon::PopConSourceHandler<CSCDBCrosstalk>(name,cstring,cat,evt,est), m_pop_connect(pconnect)
{
	m_name = name;
	m_cs = cstring;
	lgrdr = new LogReader(m_pop_connect);
	
}

popcon::CSCDBCrosstalkImpl::~CSCDBCrosstalkImpl()
{
}

void popcon::CSCDBCrosstalkImpl::getNewObjects()
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
	
	std::cerr << "Source implementation test ::getNewObjects : enter since ? \n";
	std::cin >> snc;
	std::cerr << "getNewObjects : enter till ? \n";
	std::cin >> tll;


	//the following code works, however since 1.6.0_pre7 it causes glibc 
	//double free error (inside CSC specific code) - commented 
	//
	//Using ES to get the data:

	edm::ESHandle<CSCDBCrosstalk> crosstalk;
	esetup.get<CSCDBCrosstalkRcd>().get(crosstalk);
	mycrosstalk = crosstalk.product();
	std::cout << "size " << crosstalk->crosstalk.size() << std::endl;

	popcon::IOVPair iop = {snc,tll};
	popcon::IOVPair iop2 = {snc+20,tll};
	popcon::IOVPair iop3 = {snc+10,tll};

	CSCDBCrosstalk * p1 = new CSCDBCrosstalk(*mycrosstalk);
	CSCDBCrosstalk * p2 = new CSCDBCrosstalk(*mycrosstalk);
	
	m_to_transfer->push_back(std::make_pair((CSCDBCrosstalk*)mycrosstalk,iop));
	m_to_transfer->push_back(std::make_pair((CSCDBCrosstalk*)p1,iop2));
	m_to_transfer->push_back(std::make_pair((CSCDBCrosstalk*)p2,iop3));

	std::cout << "CSC src - > getNewObjects -----------\n";
}
