#ifndef DQM_SiStripHistoricInfoClient_SiStripPopConHistoryDQMBase_H
#define DQM_SiStripHistoricInfoClient_SiStripPopConHistoryDQMBase_H

#include "DQMOffline/CalibTracker/plugins/SiStripDQMPopConSourceHandler.h"
#include "CondFormats/DQMObjects/interface/HDQMSummary.h"
#include "DQMOffline/CalibTracker/plugins/SiStripDQMHistoryHelper.h"

class HDQMfitUtilities;

class SiStripPopConHistoryDQMBase : public SiStripDQMPopConSourceHandler<HDQMSummary>, protected SiStripDQMHistoryHelper
{
public:
  explicit SiStripPopConHistoryDQMBase(const edm::ParameterSet& pset);
  ~SiStripPopConHistoryDQMBase() override;
  void dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) override;
  HDQMSummary* getObj() const override;
  bool checkForCompatibility( const std::string& otherMetaData );
private:
  std::unique_ptr<HDQMfitUtilities> fitME_;
  std::string MEDir_;
  typedef std::vector<edm::ParameterSet> VParameters;
  VParameters histoList_;
  HDQMSummary m_obj;
};

#endif // DQM_SiStripHistoricInfoClient_SiStripPopConHistoryDQMBase_H
