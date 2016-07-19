#ifndef DQM_SiStripHistoricInfoClient_SiStripPopConHistoryDQMBase_H
#define DQM_SiStripHistoricInfoClient_SiStripPopConHistoryDQMBase_H

#include "DQMOffline/CalibTracker/plugins/SiStripPopConSourceHandler.h"
#include "CondFormats/DQMObjects/interface/HDQMSummary.h"
#include "DQMOffline/CalibTracker/plugins/SiStripDQMStoreReader.h"
#include "DQMOffline/CalibTracker/plugins/SiStripDQMHistoryHelper.h"

class HDQMfitUtilities;

class SiStripPopConHistoryDQMBase : public SiStripPopConSourceHandler<HDQMSummary>, protected SiStripDQMStoreReader, protected SiStripDQMHistoryHelper
{
public:
  explicit SiStripPopConHistoryDQMBase(const edm::ParameterSet& pset);
  virtual ~SiStripPopConHistoryDQMBase();
  HDQMSummary* getObj() const;
  bool checkForCompatibility( const std::string& otherMetaData );
private:
  std::unique_ptr<HDQMfitUtilities> fitME_;
  std::string MEDir_;
  typedef std::vector<edm::ParameterSet> VParameters;
  VParameters histoList_;
};

#endif // DQM_SiStripHistoricInfoClient_SiStripPopConHistoryDQMBase_H
