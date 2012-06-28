#ifndef TriggerReportHelpers_H
#define TriggerReportHelpers_H

#include "xdata/String.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Table.h"

#include "EventFilter/Utilities/interface/MsgBuf.h"
#include "EventFilter/Utilities/interface/TriggerReportDef.h"

#include <vector>
#include <string>

namespace edm{
  class ModuleDescription;
}

namespace evf{


  namespace fuep{
    class TriggerReportHelpers{
    public:
      TriggerReportHelpers() 
	: eventsProcessed_(0)
	, eventsAccepted_(0)
	, tableFormatted_(false)
	, lumiSectionIndex_(0)
	, prescaleIndex_(0)
	, cache_(sizeof(TriggerReportStatic),MSQS_MESSAGE_TYPE_TRR)
	, adjustLsIndex_(false)
	{}
      void resetFormat(){tableFormatted_ = false;}
      void printReportTable();
      void printTriggerReport(edm::TriggerReport &);
      void triggerReportUpdate(edm::TriggerReport &, unsigned int, unsigned int, bool = true);
      void packedTriggerReportToTable();
      void formatReportTable(edm::TriggerReport &
			     , std::vector<edm::ModuleDescription const*>&
			     , std::string &pathIndexTable
			     , bool noNukeLegenda);
      void fillPathIndexTable(std::string &);
      xdata::Table &getTable(){return triggerReportAsTable_;} 
      xdata::Table &getTableWithNames(){return triggerReportAsTableWithNames_;} 
      xdata::UnsignedInteger32 &getProcThisLumi(){return eventsProcessed_;}
      xdata::UnsignedInteger32 &getAccThisLumi(){return eventsAccepted_;}
      bool checkLumiSection(unsigned int ls) {return (ls == lumiSectionIndex_);}
      void packTriggerReport(edm::TriggerReport &, ShmOutputModuleRegistry *);
      void sumAndPackTriggerReport(MsgBuf &);
      void resetPackedTriggerReport();
      void adjustLsIndexForRestart(){adjustLsIndex_ = true; if(lumiSectionIndex_>1) lumiSectionIndex_--;}
      void resetTriggerReport();
      evf::MsgBuf & getPackedTriggerReport(){return cache_;}
      TriggerReportStatic *getPackedTriggerReportAsStruct(){return (TriggerReportStatic *)cache_->mtext;}
      xdata::String *getPathLegenda(){return &pathLegenda_;}
      void resetLumiSectionReferenceIndex(){lumiSectionIndex_=0;}
      void withdrawLumiSectionIncrement(){if(lumiSectionIndex_>0) lumiSectionIndex_--;}
      unsigned int getLumiSectionReferenceIndex(){return lumiSectionIndex_;}
      std::string findLabelOfModuleTypeInEndPath(edm::TriggerReport &, 
						 std::vector<edm::ModuleDescription const*>&,
						 unsigned int, 
						 std::string);
      unsigned int getl1pre(unsigned int ind){return pl1pre_[ind];}
      unsigned int getps(unsigned int ind){return pps_[ind];}
      unsigned int getaccept(unsigned int ind){return paccept_[ind];}
      unsigned int getfailed(unsigned int ind){return pfailed_[ind];}
      unsigned int getexcept(unsigned int ind){return pexcept_[ind];}
    private:
      // scalers table
      xdata::UnsignedInteger32 eventsProcessed_;
      xdata::UnsignedInteger32 eventsAccepted_;
      xdata::Table triggerReportAsTable_;
      xdata::Table triggerReportAsTableWithNames_;
      xdata::String pathLegenda_;
      bool         tableFormatted_;
      std::vector<int> l1pos_;
      std::vector<int> pspos_;
      std::vector<std::string> outname_;
      static const std::string columns[5];
      std::vector<std::string>            paths_;
      std::vector<xdata::UnsignedInteger32> l1pre_;
      std::vector<xdata::UnsignedInteger32> ps_;
      std::vector<xdata::UnsignedInteger32> accept_;
      std::vector<xdata::UnsignedInteger32> except_;
      std::vector<xdata::UnsignedInteger32> failed_;
      std::vector<unsigned int> pl1pre_;
      std::vector<unsigned int> pps_;
      std::vector<unsigned int> paccept_;
      std::vector<unsigned int> pexcept_;
      std::vector<unsigned int> pfailed_;
      std::map<std::string, xdata::UnsignedInteger32> pathIndexMap_;
      unsigned int lumiSectionIndex_;
      unsigned int prescaleIndex_;
      edm::TriggerReport trp_;
      MsgBuf  cache_;
      bool adjustLsIndex_;
    };
  }
}



#endif
