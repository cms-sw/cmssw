#ifndef FECONFSpikeDAT_H
#define FECONFSpikeDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigSpikeInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigSpikeDat : public IDataItem {
 public:
  friend class EcalCondDBInterface; // XXX temp should not need
  FEConfigSpikeDat();
  ~FEConfigSpikeDat();

  // User data methods
  inline std::string getTable() { return "FE_CONFIG_SPIKE_DAT"; }

  inline void setSpikeThreshold(int x) { m_thr = x; }
  inline int getSpikeThreshold() const { return m_thr; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const FEConfigSpikeDat* item, FEConfigSpikeInfo* iconf)
    throw(std::runtime_error);


  void writeArrayDB(const std::map< EcalLogicID, FEConfigSpikeDat>* data, FEConfigSpikeInfo* iconf)
  throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, FEConfigSpikeDat >* fillMap, FEConfigSpikeInfo* iconf)
     throw(std::runtime_error);

  // User data
  int m_thr;

};

#endif
