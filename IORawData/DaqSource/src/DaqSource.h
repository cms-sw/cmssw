#ifndef DaqSource_DaqSource_H
#define DaqSource_DaqSource_H

/** \class DaqSource
 *  An input service for raw data. 
 *  The actual source can be the real DAQ, a file, a random generator, etc.
 *
 *  $Date: 2005/10/06 18:23:47 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - S. Argiro'
 */

#include <memory>

#include "FWCore/Framework/interface/RawInputSource.h"

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
  DaqBaseReader * reader_;
};

#endif
