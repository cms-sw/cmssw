#ifndef DQMEventSource_DQMEventSource_H
#define DQMEventSource_DQMEventSource_H

/** \class DQMEventSource
 *  An input service for DQM clients. 
 *  Fake events are filled with only event and run IDs from the DQM.
 *  This allows to access the correct ES table, i.e. those
 *  referring to the events analyzed by the DQM sourcese 
 *  in the FF.
 *  The method mui->doMonitoring() is called to excute the clients
 *  analysis during the "onUpdate" status. 
 *  $Date: 2006/01/20 11:45:10 $
 *  $Revision: 1.4 $
 *  \author S. Bolognesi - M. Zanetti
 */

#include <memory>

#include "FWCore/Framework/interface/RawInputSource.h"

namespace edm {
    class ParameterSet;
    class InputSourceDescription;
    class Event;
    class EventId;
    class Timestamp;
}

class MonitorUserInterface;
class SubscriptionHandle;
class QTestHandle;


class DQMEventSource : public edm::RawInputSource {

 public:
  explicit DQMEventSource(const edm::ParameterSet& pset, 
			  const edm::InputSourceDescription& desc);
  
  virtual ~DQMEventSource() {};


 private:

  virtual std::auto_ptr<edm::Event> readOneEvent();

  MonitorUserInterface * mui;

  SubscriptionHandle *subscriber;
  QTestHandle * qtHandler;

  bool getQualityTestsFromFile;

  unsigned int updatesCounter;

};

#endif
