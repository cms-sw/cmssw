#ifndef TriggerReportHelpers_H
#define TriggerReportHelpers_H

#include "xdata/String.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Table.h"

#include <vector>
#include <string>

namespace edm{
  class TriggerReport;   
  class ModuleDescription;
}

namespace evf{
  namespace fuep{
    class TriggerReportHelpers{
    public:
      TriggerReportHelpers() : tableFormatted_(false), lumiSectionIndex_(0){}
      void resetFormat(){tableFormatted_ = false;}
      void printReportTable();
      void printTriggerReport(edm::TriggerReport &);
      void triggerReportToTable(edm::TriggerReport &, unsigned int, bool = true);
      void formatReportTable(edm::TriggerReport &, std::vector<edm::ModuleDescription const*>&);
      xdata::Table &getTable(){return triggerReportAsTable_;} 
      bool checkLumiSection(unsigned int ls) {return (ls == lumiSectionIndex_);}
    private:
      // scalers table
      xdata::Table triggerReportAsTable_;
      bool         tableFormatted_;
      std::vector<int> l1pos_;
      std::vector<int> pspos_;
      static const std::string columns[6];
      std::vector<xdata::String> paths_;
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
    };
  }
}



#endif
