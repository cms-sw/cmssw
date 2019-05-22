#ifndef ODWEIGHTSSAMPLESDAT_H
#define ODWEIGHTSSAMPLESDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/IODConfig.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/ODFEWeightsInfo.h"

class ODWeightsSamplesDat : public IODConfig {
public:
  friend class EcalCondDBInterface;
  ODWeightsSamplesDat();
  ~ODWeightsSamplesDat() override;

  // User data methods
  inline std::string getTable() override { return "DCC_WEIGHTSAMPLE_DAT"; }

  inline void setId(int dac) { m_ID = dac; }
  inline int getId() const { return m_ID; }

  inline void setFedId(int dac) { m_fed = dac; }
  inline int getFedId() const { return m_fed; }

  inline void setSampleId(int dac) { m_ss = dac; }
  inline int getSampleId() const { return m_ss; }

  inline void setWeightNumber(int dac) { m_sn = dac; }
  inline int getWeightNumber() const { return m_sn; }

private:
  void clear();
  void prepareWrite() noexcept(false) override;

  void writeDB(const ODWeightsSamplesDat* item, ODFEWeightsInfo* iov) noexcept(false);

  void writeArrayDB(const std::vector<ODWeightsSamplesDat>& data, ODFEWeightsInfo* iov) noexcept(false);

  void fetchData(std::vector<ODWeightsSamplesDat>* fillMap, ODFEWeightsInfo* iov) noexcept(false);

  void fetchData(ODWeightsSamplesDat* p) noexcept(false);

  // User data

  int m_fed;
  int m_ss;
  int m_sn;
  int m_ID;
};

#endif
