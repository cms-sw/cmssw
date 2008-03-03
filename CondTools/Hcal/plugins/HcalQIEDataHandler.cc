#include "CondTools/Hcal/interface/HcalQIEDataHandler.h"
#include "DataFormats/DetId/interface/DetId.h"

HcalQIEDataHandler::HcalQIEDataHandler(edm::ParameterSet const & ps)
{
  m_name = ps.getUntrackedParameter<std::string>("name","HcalQIEDataHandler");
  sinceTime = ps.getUntrackedParameter<unsigned>("IOVRun",0);
  fFile = ps.getUntrackedParameter<std::string>("CondFile","");
}

HcalQIEDataHandler::~HcalQIEDataHandler()
{
}

void HcalQIEDataHandler::getNewObjects()
{
  edm::LogInfo   ("HcalQIEDataHandler") << "------- " << m_name 
					  << " - > getNewObjects\n" << 
    //check whats already inside of database
    "got offlineInfo"<<
    tagInfo().name << ", size " << tagInfo().size 
					  << ", last object valid since " 
					  << tagInfo().lastInterval.first << std::endl;  
  

  HcalQIEData* myobject = new HcalQIEData();
  std::cout << "Using file: " << fFile << std::endl;

  std::ifstream inStream(fFile.c_str() );
  bool g = HcalDbASCIIIO::getObject(inStream, myobject);

  std::cout << "bool=" << g << std::endl;
  std::cout << myobject << std::endl;

  //  IOV information
  cond::Time_t myTime = sinceTime;

  std::cout << "Using IOV run " << sinceTime << std::endl;

  // prepare for transfer:
  m_to_transfer.push_back(std::make_pair(myobject,myTime));

  edm::LogInfo("HcalQIEDataHandler") << "------- " << m_name << " - > getNewObjects" << std::endl;

}
