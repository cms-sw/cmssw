#ifndef DaqSource_DaqSource_H
#define DaqSource_DaqSource_H

/** \class DaqSource
 *  An input service for raw data. 
 *  The actual source can be the real DAQ, a file, a random generator, etc.
 *
 *  $Date: 2007/04/19 13:28:45 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - S. Argiro'
 */

#include <memory>

#include "FWCore/Sources/interface/RawInputSource.h"

namespace edm {
    class ParameterSet;
    class InputSourceDescription;
    class Event;
}

class DaqBaseReader;

class DaqSource : public edm::RawInputSource {

 public:
  explicit DaqSource(const edm::ParameterSet& pset, 
		     const edm::InputSourceDescription& desc);

  virtual ~DaqSource();

 private:

  virtual std::auto_ptr<edm::Event> readOneEvent();
  DaqBaseReader*  reader_;
  unsigned int    lumiSegmentSizeInEvents_; //temporary kludge, LS# will come from L1 Global record
  unsigned int    lsid_;
  bool            fakeLSid_;
};

#endif
