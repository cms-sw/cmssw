#ifndef TriggerReportHelpers_H
#define TriggerReportHelpers_H

#include "xdata/String.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Table.h"

#include "FWCore/Framework/interface/TriggerReport.h"

#include "MsgBuf.h"

#include <vector>
#include <string>

namespace edm{
  class ModuleDescription;
}

namespace evf{
  
  static const size_t max_paths = 300;
  static const size_t max_endpaths = 10;
  static const size_t max_label = 30;
  static const size_t max_modules = 50;


  struct ModuleInPathsSummaryStatic{
    //max length of a module label is 80 characters - name is truncated otherwise
    int timesVisited;
    int timesPassed;
    int timesFailed;
    int timesExcept;
    char moduleLabel[max_label];
  };
  struct PathSummaryStatic
  {
    //max length of a path name is 80 characters - name is truncated otherwise
    //max modules in a path are 100
    //    int bitPosition;
    int timesRun;
    int timesPassedPs;
    int timesPassedL1;
    int timesPassed;
    int timesFailed;
    int timesExcept;
    //    int modulesInPath;
    //    char name[max_label];
    //    ModuleInPathsSummaryStatic moduleInPathSummaries[max_modules];
  };
  struct TriggerReportStatic{
    //max number of paths in a menu is 500
    //max number of endpaths in a menu is 20
    unsigned int           lumiSection;
    edm::EventSummary      eventSummary;
    int                    trigPathsInMenu;
    int                    endPathsInMenu;
    PathSummaryStatic      trigPathSummaries[max_paths];
    PathSummaryStatic      endPathSummaries[max_endpaths];
  };


  namespace fuep{
    class TriggerReportHelpers{
    public:
      TriggerReportHelpers() 
	: tableFormatted_(false)
	, lumiSectionIndex_(0)
	, cache_(sizeof(TriggerReportStatic),MSQS_MESSAGE_TYPE_TRR)
	{}
      void resetFormat(){tableFormatted_ = false;}
      void printReportTable();
      void printTriggerReport(edm::TriggerReport &);
      void triggerReportToTable(edm::TriggerReport &, unsigned int, bool = true);
      void packedTriggerReportToTable();
      void formatReportTable(edm::TriggerReport &
			     , std::vector<edm::ModuleDescription const*>&
			     , bool noNukeLegenda);
      xdata::Table &getTable(){return triggerReportAsTable_;} 
      bool checkLumiSection(unsigned int ls) {return (ls == lumiSectionIndex_);}
      void packTriggerReport(edm::TriggerReport &);
      void sumAndPackTriggerReport(MsgBuf &);
      void resetPackedTriggerReport();
      void resetTriggerReport();
      evf::MsgBuf & getPackedTriggerReport(){return cache_;}
      TriggerReportStatic *getPackedTriggerReportAsStruct(){return (TriggerReportStatic *)cache_->mtext;}
      xdata::String *getPathLegenda(){return &pathLegenda_;}
    private:
      // scalers table
      xdata::Table triggerReportAsTable_;
      xdata::String pathLegenda_;
      bool         tableFormatted_;
      std::vector<int> l1pos_;
      std::vector<int> pspos_;
      static const std::string columns[5];
      std::vector<std::string>              paths_;
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
      unsigned int lumiSectionIndex_;
      edm::TriggerReport trp_;
      MsgBuf  cache_;
    };
  }
}



#endif
