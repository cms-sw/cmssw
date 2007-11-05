#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixHandler.h"

popcon::CSCDBNoiseMatrixImpl::CSCDBNoiseMatrixImpl(std::string name, std::string cstring, std::string cat,const edm::Event& evt, const edm::EventSetup& est, std::string pconnect) : popcon::PopConSourceHandler<CSCDBNoiseMatrix>(name,cstring,cat,evt,est), m_pop_connect(pconnect)
{
	m_name = name;
	m_cs = cstring;
	lgrdr = new LogReader(m_pop_connect);
	
}

popcon::CSCDBNoiseMatrixImpl::~CSCDBNoiseMatrixImpl()
{
}

void popcon::CSCDBNoiseMatrixImpl::getNewObjects()
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

	edm::ESHandle<CSCDBNoiseMatrix> matrix;
	esetup.get<CSCDBNoiseMatrixRcd>().get(matrix);
	mymatrix = matrix.product();
	std::cout << "size " << matrix->matrix.size() << std::endl;

	//changed to an empty object
	//mymatrix = new CSCDBNoiseMatrix();
	
	
	popcon::IOVPair iop = {snc,tll};
	popcon::IOVPair iop2 = {snc+20,tll};
	popcon::IOVPair iop3 = {snc+10,tll};

	CSCDBNoiseMatrix * p1 = new CSCDBNoiseMatrix(*mymatrix);
	CSCDBNoiseMatrix * p2 = new CSCDBNoiseMatrix(*mymatrix);
	
	m_to_transfer->push_back(std::make_pair((CSCDBNoiseMatrix*)mymatrix,iop));
	m_to_transfer->push_back(std::make_pair((CSCDBNoiseMatrix*)p1,iop2));
	m_to_transfer->push_back(std::make_pair((CSCDBNoiseMatrix*)p2,iop3));

	std::cout << "CSC src - > getNewObjects -----------\n";
}
