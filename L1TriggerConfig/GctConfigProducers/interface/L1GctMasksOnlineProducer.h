//-------------------------------------------------
//
//   \class L1GctMasksOnlineProducer
//
//   Description:  ESProducer for O2O of GCT channel mass objects
//
//   $Date: 2008/11/24 19:00:37 $
//   $Revision: $
//
//   Author :
//   Jim Brooke
//
//--------------------------------------------------
#ifndef GctConfigProducers_L1GctMasksOnlineProducer_h
#define GctConfigProducers_L1GctMasksOnlineProducer_h

// system include files
#include <memory>
#include <boost/shared_ptr.hpp>

// user include files

#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

class L1GctChannelMask;
class L1GctChannelMaskRcd;

//
// class declaration
//


class L1GctMasksOnlineProducer : public L1ConfigOnlineProdBase<L1GctChannelMaskRcd, L1GctChannelMask> {
public:
  L1GctMasksOnlineProducer(const edm::ParameterSet&);
  ~L1GctMasksOnlineProducer();
  
  /// The method that actually implements the production of the parameter objects
  virtual boost::shared_ptr<L1GctChannelMask> newObject( const std::string& objectKey );
 protected:

  void checkCMSSWVersion(const coral::AttributeList& configRecord);
private:
  std::string lookupSoftwareConfigKey(const std::string& globalKey);
  bool ignoreVersionMismatch_;  
};

#endif
