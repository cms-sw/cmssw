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
  ~ODWeightsSamplesDat();

  // User data methods
  inline std::string getTable() { return "DCC_WEIGHTSAMPLE_DAT"; }

  inline void setId(int dac) { m_ID = dac; }
  inline int getId() const { return m_ID; }

  inline void setFedId(int dac) { m_fed = dac; }
  inline int getFedId() const { return m_fed; }

  inline void setSampleId(int dac) { m_ss = dac; }
  inline int getSampleId() const { return m_ss; }

  inline void setWeightNumber(int dac) { m_sn = dac; }
  inline int getWeightNumber() const { return m_sn; }



 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const ODWeightsSamplesDat* item, ODFEWeightsInfo* iov )
    throw(std::runtime_error);

  void writeArrayDB(const std::vector< ODWeightsSamplesDat > data, ODFEWeightsInfo* iov)
    throw(std::runtime_error);


  void fetchData(std::vector< ODWeightsSamplesDat >* fillMap, ODFEWeightsInfo* iov)
     throw(std::runtime_error);

  // User data

  int m_fed;
  int m_ss;
  int m_sn;
  int m_ID;
 
};

#endif
