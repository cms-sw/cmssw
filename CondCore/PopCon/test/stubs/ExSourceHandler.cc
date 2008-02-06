#include "ExSourceHandler.h"

popcon::ExPedestalSource::ExPedestalSource(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","ExPedestalSource")){
}

popcon::ExPedestalSource::~ExPedestalSource()
{
 
}

void popcon::ExPedestalSource::getNewObjects() {
  std::cout << "------- " << name << " - > getNewObjects\n";
  //check whats already inside of database
  std::cout<<"got offlineInfo"<<std::endl;
  std::cout << tagInfo().name << " , last object valid since " 
	    << tagInfo().lastInterval.first << std::endl;  

  
  
  
  unsigned int snc;
  
  std::cerr << "Source implementation test ::getNewObjects : enter first since ? \n";
  std::cin >> snc;


  //the following code cannot work as event setup is not initialized with a real time!
  //
  //Using ES to get the data:
  
  /*	edm::ESHandle<CSCPedestals> pedestals;
  //esetup.get<CSCPedestalsRcd>().get(pedestals);
  //mypedestals = pedestals.product();
  std::cout << "size " << mypedestals->pedestals.size() << std::endl;
  */

  //changed to an empty objec
  Pedestals * p0 = new Pedestals;
  Pedestals * p1 = new Pedestals;
  Pedestals * p2 = new Pedestals;
  m_to_transfer.push_back(std::make_pair((Pedestals*)p0,snc));
  m_to_transfer.push_back(std::make_pair((Pedestals*)p1,snc+20));
  m_to_transfer.push_back(std::make_pair((Pedestals*)p2,snc+10));
  std::cerr << "------- " << name << " - > getNewObjects\" << std::endl;
}
