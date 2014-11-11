#ifndef _SiStripSummaryCreator_h_
#define _SiStripSummaryCreator_h_

#include "DQMServices/Core/interface/MonitorElement.h"
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include "DQMServices/Core/interface/DQMStore.h"

class SiStripConfigWriter;

class SiStripSummaryCreator {

 public:

  SiStripSummaryCreator();
  virtual ~SiStripSummaryCreator();
  bool readConfiguration(std::string & file_path);

  void createSummary(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);

  void fillLayout(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);
  void setSummaryMENames( std::map<std::string, std::string>& me_names);
  int getFrequency() { return summaryFrequency_;}

 private:
  MonitorElement* getSummaryME(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, 
			       std::string& name, std::string htype);


  void fillGrandSummaryHistos(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);
  void fillSummaryHistos(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter);
  void fillHistos(int ival, int istep, std::string htype, 
		  MonitorElement* me_src, MonitorElement* me);


  std::map<std::string, std::string> summaryMEMap;
 
  int summaryFrequency_;


};
#endif
