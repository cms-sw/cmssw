#ifndef HLTrigReportService_H
#define HLTrigReportService_H

#include <string>
#include <vector>

// Abstract base class for service used by HLTrigReport

class HLTrigReport;

class HLTrigReportService {

 public:

  virtual void registerModule(const HLTrigReport *)=0;

  virtual void setDatasetNames(const std::vector<std::string>&)=0 ;
  virtual void setDatasetCounts(const std::vector<unsigned int>&)=0;

  virtual void setStreamNames(const std::vector<std::string>&)=0 ;
  virtual void setStreamCounts(const std::vector<unsigned int>&)=0;

};

#endif
