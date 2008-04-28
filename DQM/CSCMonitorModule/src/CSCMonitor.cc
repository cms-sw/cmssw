#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

CSCMonitor::CSCMonitor(const edm::ParameterSet& iConfig )
{

  setParameters();

  printout = iConfig.getUntrackedParameter<bool>("monitorVerbosity", false);
  xmlHistosBookingCfgFile = iConfig.getUntrackedParameter<std::string>("BookingFile", "emuDQMBooking.xml"); 
  fSaveHistos  = iConfig.getUntrackedParameter<bool>("CSCDQMSaveRootFile", false);
  saveRootFileEventsInterval  = iConfig.getUntrackedParameter<int>("EventsInterval", 20000);
  RootHistoFile  = iConfig.getUntrackedParameter<std::string>("RootFileName", "DQM_CSC_Monitor.root");

  // CSC Mapping
  cscMapping  = CSCReadoutMappingFromFile(iConfig);
      
  this->loadBooking();

  dbe = edm::Service<DQMStore>().operator->();
  dbe->showDirStructure();
  gStyle->SetPalette(1,0);
}


void CSCMonitor::setParameters() 
{
  nEvents = 0;
  L1ANumber = 0;
  BXN = 0;
  unpackMask = UNPACK_ALL;
  nDMBEvents.clear();
  unpackedDMBcount = 0;
  logger_ = "CSC_DQM:";
  dduCheckMask = 0xFFFFFFFF;
  binCheckMask = 0xFFFFFFFF;
  xmlHistosBookingCfgFile = "";
  tmap = getCSCTypeToBinMap();
}

void CSCMonitor::clearMECollection(ME_List & collection) 
{

  if (collection.size() > 0) {
    for (ME_List_iterator itr = collection.begin();itr != collection.end(); ++itr) {
      delete itr->second;
    }
    collection.clear();
  }

}
void CSCMonitor::printMECollection(ME_List & collection)
{
  int i = 0;
  for (ME_List_iterator itr = collection.begin();itr != collection.end(); ++itr) {
    LOG4CPLUS_DEBUG(logger_, ++i << ":" << itr->first << ":" << itr->second->getFullName());
  }

}


CSCMonitor::~CSCMonitor() {
  std::map<std::string, ME_List >::iterator itr;	
  for (itr = MEs.begin(); itr != MEs.end(); ++itr) {
    clearMECollection(itr->second);
  }

  MEs.clear();
  clearMECollection(commonMEfactory);
  clearMECollection(chamberMEfactory);
  clearMECollection(dduMEfactory);
}

void CSCMonitor::loadBooking() {
  if (MEs.size() > 0) {
    std::map<std::string, ME_List >::iterator itr;
    for (itr = MEs.begin(); itr != MEs.end(); ++itr) {
      clearMECollection(itr->second);
    }
    MEs.clear();
  }

  if (loadXMLBookingInfo(xmlHistosBookingCfgFile) == 0) 
    {
      setParameters();
    }

}

std::map<std::string, int> CSCMonitor::getCSCTypeToBinMap()
{
	std::map<std::string, int> tmap;
	tmap["ME-4/2"] = 0;
	tmap["ME-4/1"] = 1;	
        tmap["ME-3/2"] = 2;
	tmap["ME-3/1"] = 3;
	tmap["ME-2/2"] = 4;
	tmap["ME-2/1"] = 5;
	tmap["ME-1/3"] = 6;
	tmap["ME-1/2"] = 7;
	tmap["ME-1/1"] = 8;
	tmap["ME+1/1"] = 9;
	tmap["ME+1/2"] = 10;
	tmap["ME+1/3"] = 11;
	tmap["ME+2/1"] = 12;
	tmap["ME+2/2"] = 13;
	tmap["ME+3/1"] = 14;
	tmap["ME+3/2"] = 15;
	tmap["ME+4/1"] = 16;
	tmap["ME+4/2"] = 17;
	return tmap;
	
}

std::string CSCMonitor::getCSCTypeLabel(int endcap, int station, int ring )
{
	std::string label = "Unknown";
	std::ostringstream st;
	if ((endcap > 0) && (station>0) && (ring>0)) {
		if (endcap==1) {
			st << "ME+" << station << "/" << ring;
			label = st.str();
		} else if (endcap==2) {
			st << "ME-" << station << "/" << ring;
                        label = st.str();
		} else {
			label = "Unknown";
		}
	}
	return label;
}

void CSCMonitor::getCSCFromMap(int crate, int slot, int& csctype, int& cscposition)
{
//  LOG4CPLUS_INFO(logger_, "========== get CSC from Map crate" << crate << " slot" << slot);
  int iendcap = -1;
  int istation = -1;
  int iring = -1;
  // TODO: Add actual Map conversion
  int id = cscMapping.chamber(iendcap, istation, crate, slot, -1);
  if (id==0) { 
	return;
  }
  CSCDetId cid( id );
  iendcap = cid.endcap();
  istation = cid.station();
  iring = cid.ring();
  cscposition = cid.chamber();

//  std::map<std::string, int> tmap = getCSCTypeToBinMap();
  std::string tlabel = getCSCTypeLabel(iendcap, istation, iring );
  std::map<std::string,int>::const_iterator it = tmap.find( tlabel );
  if (it != tmap.end()) {
	csctype = it->second;
//	LOG4CPLUS_INFO(logger_, "========== get CSC from Map label:" << tlabel << "/" << cscposition);
  } else {
//	LOG4CPLUS_INFO(logger_, "========== can not find map");
	csctype = 0;
  }
 
  // return bin number which corresponds for CSC Type (ex. ME+4/2 -> bin 18)  
  

}

bool CSCMonitor::isMEvalid(ME_List& MEs, std::string name, CSCMonitorObject*& me, uint32_t mask)
{
  if ((unpackMask & mask)==0) return false;
  ME_List_iterator res = MEs.find(name);
  if (res != MEs.end() && (res->second != 0)) {
    me = res->second;
    return true;
  } else {
		
    // edm::LogWarning ("CSC DQM: ") << "Can not find ME " << name;
    LOG4CPLUS_WARN(logger_, "Can not find ME " << name);
    me = 0;
    return false;
  }
	
}

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using namespace edm::serviceregistry;

typedef ParameterSetMaker<CSCMonitorInterface,CSCMonitor> maker;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE_MAKER(CSCMonitor,maker);



