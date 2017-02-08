#ifndef MONPULSESHAPEDAT_H
#define MONPULSESHAPEDAT_H

#include <vector>
#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonPulseShapeDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  MonPulseShapeDat();
  ~MonPulseShapeDat();

  // User data methods
  inline std::string getTable() { return "MON_PULSE_SHAPE_DAT"; }

  inline void setSamples( std::vector<float> &samples, int gain )
    throw(std::runtime_error)
    { 
      if (samples.size() != 10) {
	throw(std::runtime_error("MonPulseShapeDat::setSamples:  There should be 10 samples."));
      }

      if (gain == 1) {
	m_samplesG1 = samples; 
      } else if (gain == 6) {
	m_samplesG6 = samples;
      } else if (gain == 12) {
	m_samplesG12 = samples;
      } else {
	throw(std::runtime_error("MonPulseShapeDat::setSamples:  Gain should be 1, 6 or 12"));
      }
    }

  inline std::vector<float> getSamples(int gain) const
    throw(std::runtime_error)
    { 
      if (gain == 1) {
	return m_samplesG1;
      } else if (gain == 6) {
	return m_samplesG6;
      } else if (gain == 12) {
	return m_samplesG12;
      } else {
	throw(std::runtime_error("MonPulseShapeDat::getSamples:  Gain should be 1, 6 or 12"));
      }
    }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const MonPulseShapeDat* item, MonRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, MonPulseShapeDat >* fillVec, MonRunIOV* iov)
     throw(std::runtime_error);

  // User data
  std::vector<float> m_samplesG1;
  std::vector<float> m_samplesG6;
  std::vector<float> m_samplesG12;
};

#endif
