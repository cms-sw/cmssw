#ifndef DQM_CASTORMONITOR_CASTORDATAINTEGRITYMONITOR_H
#define DQM_CASTORMONITOR_CASTORDATAINTEGRITYMONITOR_H

#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "EventFilter/CastorRawToDigi/interface/CastorUnpacker.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <map>

class CastorDataIntegrityMonitor: public CastorBaseMonitor 
{

 public:
  CastorDataIntegrityMonitor();
  ~CastorDataIntegrityMonitor();
  
  void setup(const edm::ParameterSet& ps, DQMStore* dbe);

  void processEvent(const FEDRawDataCollection& RawData, const HcalUnpackerReport& report, const CastorElectronicsMap& emap);

  void unpack(const FEDRawData& raw, const CastorElectronicsMap& emap);

  void cleanup();

  void reset();

 public: 
  std::map<uint32_t, std::vector<HcalCastorDetId> > DCCtoCell;
  std::map<uint32_t, std::vector<HcalCastorDetId> > ::iterator thisDCC;

 private: 
 
  std::vector<std::vector<bool> > problemCASTOR;    

  void mapHTRproblem (int dcc, int spigot) ;
  void mapDCCproblem(int dcc) ;
  void fillzoos(int bin, int dccid);
  void UpdateMap();

  
  std::vector<int> fedUnpackList_; //-- vector of CASTOR FEDs
  int  dccid;
  int ievt_;
  bool CDFProbThisDCC;
  int spigotStatus;
  double statusSpigotDCC;

 private:  

  int problemsSpigot[15][3];

  ////---- define histograms
  
  MonitorElement* meEVT_;
  MonitorElement* fedEntries;
  MonitorElement* fedFatal;
  MonitorElement* fedNonFatal;

  MonitorElement* meDCCVersion;
  MonitorElement* spigotStatusMap;

  ////---- Member variables for reference values to be used in consistency checks.
  std::map<int, short> CDFversionNumber_list;
  std::map<int, short>::iterator CDFvers_it;
  std::map<int, short> CDFEventType_list;
  std::map<int, short>::iterator CDFEvT_it;
  std::map<int, short> CDFReservedBits_list;
  std::map<int, short>::iterator CDFReservedBits_it;
  std::map<int, short> DCCEvtFormat_list;
  std::map<int, short>::iterator DCCEvtFormat_it;
  std::map<int, short> DCCRsvdBits_list;
  std::map<int, short>::iterator DCCRsvdBits_it;
    
};

#endif
