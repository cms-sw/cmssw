#include "CondTools/Hcal/interface/HcalElectronicsMapHandler.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include <memory>

HcalElectronicsMapHandler::HcalElectronicsMapHandler(edm::ParameterSet const & ps)
{
  m_name = ps.getUntrackedParameter<std::string>("name","HcalElectronicsMapHandler");
  sinceTime = ps.getUntrackedParameter<unsigned>("IOVRun",0);
  fFile = ps.getParameter<edm::FileInPath>("CondFile");
}

HcalElectronicsMapHandler::~HcalElectronicsMapHandler()
{
}

void HcalElectronicsMapHandler::getNewObjects()
{
  edm::LogInfo   ("HcalElectronicsMapHandler") 
    << "------- " << m_name 
    << " - > getNewObjects\n" << 
    //check whats already inside of database
    "got offlineInfo"<<
    tagInfo().name << ", size " << tagInfo().size 
					  << ", last object valid since " 
					  << tagInfo().lastInterval.first << std::endl;  
  
  //  HcalElectronicsMap* myobject = new HcalElectronicsMap();
  std::auto_ptr<HcalElectronicsMap> myobject (new HcalElectronicsMap() );
  std::cout << "Using file: " << fFile.fullPath() << std::endl;

  std::ifstream inStream(fFile.fullPath().c_str() );
  if (!inStream.good ()) {
    std::cerr << "HcalTextCalibrations-> Unable to open file '" << fFile << "'" << std::endl;
    throw cms::Exception("FileNotFound") << "Unable to open '" << fFile << "'" << std::endl;
  }

  HcalDbASCIIIO::getObject(inStream, &*myobject);

//  std::cout << "bool=" << g << std::endl;
//  std::cout << myobject << std::endl;
//  std::cout << myobject->getAllChannels().size() << std::endl;
//  std::vector<DetId> channels = myobject->getAllChannels();
//  for (unsigned int i = 0; i < channels.size(); i++)
//    {
//      HcalGenericDetId genid(channels.at(i).rawId() );
//      std::cout << "channel=" << channels.at(i).rawId() 
//		<< ", index=" << genid.hashedId()
//		<< ", 1st value=" 
//		<< myobject->getValues(channels.at(i))->getValue(0) << std::endl;
//    }

  //  IOV information
  cond::Time_t myTime = sinceTime;

  std::cout << "Using IOV run " << sinceTime << std::endl;

  // prepare for transfer:
  //  m_to_transfer.push_back(std::make_pair(myobject,myTime));

  edm::LogInfo("HcalElectronicsMapHandler") << "------- " << m_name << " - > getNewObjects" << std::endl;

}
