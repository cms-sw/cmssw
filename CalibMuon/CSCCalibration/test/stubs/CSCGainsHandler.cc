#include "CalibMuon/CSCCalibration/test/stubs/CSCGainsHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include<iostream>

popcon::CSCDBGainsImpl::CSCDBGainsImpl(const edm::ParameterSet& pset): m_name(pset.getUntrackedParameter<std::string>("name","CSCDBGainsImpl"))
{}

popcon::CSCDBGainsImpl::~CSCDBGainsImpl()
{
}

void popcon::CSCDBGainsImpl::getNewObjects()
{

	std::cout << "------- CSC src - > getNewObjects\n"<<m_name;
	
	//check whats already inside of database
	
	std::cerr<<"got offlineInfo"<<std::endl;
	std::cerr << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl; 
	
	unsigned int snc,tll;
	
	std::cout << "Source implementation test ::getNewObjects : enter since ? \n";
	std::cin >> snc;
	std::cerr << "getNewObjects : enter till ? \n";
	std::cin >> tll;


	//the following code works, however since 1.6.0_pre7 it causes glibc 
	//double free error (inside CSC specific code) - commented 
	//
	//Using ES to get the data:

	edm::ESHandle<CSCDBGains> gains;
	mygains = gains.product();
	std::cout << "size " << gains->gains.size() << std::endl;

	//changed to an empty object
	//mygains = new CSCDBGains();

	CSCDBGains * p1 = new CSCDBGains(*mygains);
	CSCDBGains * p2 = new CSCDBGains(*mygains);
	
	m_to_transfer.push_back(std::make_pair((CSCDBGains*)mygains,snc));
	m_to_transfer.push_back(std::make_pair((CSCDBGains*)p1,snc+20));
	m_to_transfer.push_back(std::make_pair((CSCDBGains*)p2,snc+10));

	std::cout << "------- " << m_name << "CSC src - > getNewObjects -----------\n"<< std::endl;
}
