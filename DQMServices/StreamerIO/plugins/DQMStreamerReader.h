#ifndef IOPool_DQMStreamer_DQMStreamerReader_h
#define IOPool_DQMStreamer_DQMStreamerReader_h

#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMFileIterator.h"
#include "TriggerSelector.h"

#include "boost/filesystem.hpp"

#include <memory>
#include <string>
#include <vector>
#include <iterator>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

class EventMsgView;
class InitMsgView;

namespace edm {
class ConfigurationDescriptions;
class EventPrincipal;
class EventSkipperByID;
struct InputSourceDescription;
class ParameterSet;
class StreamerInputFile;

class DQMStreamerReader : public StreamerInputSource {
 public:
  DQMStreamerReader(ParameterSet const& pset,
                    InputSourceDescription const& desc);
  virtual ~DQMStreamerReader();

  bool newHeader();
  static void fillDescriptions(ConfigurationDescriptions& descriptions);

  typedef std::vector<std::string> Strings;

 protected:
  virtual bool checkNextEvent(); /* from raw input source */
  virtual void skip(int toSkip); /* from raw input source */

 private:
  // our own, but we do inherit reset(),
  // which will break things if called
  void reset_();

  void openFile_(std::string filename);
  bool openNextFile_();
  void closeFile_();

  InitMsgView const* getHeaderMsg();
  EventMsgView const* getEventMsg();

  EventMsgView const* prepareNextEvent();
  bool prepareNextFile();
  bool acceptEvent( const EventMsgView*);
 
  bool triggerSel();
  bool matchTriggerSel(Strings const& tnames);
  bool acceptAllEvt_;
  bool matchTriggerSel_;

  unsigned int runNumber_;
  std::string runInputDir_;
  std::string streamLabel_;
  Strings hltSel_;

  unsigned int processedEventPerLs_;
  unsigned int minEventsPerLs_;

  bool flagSkipFirstLumis_;
  bool flagEndOfRunKills_;
  bool flagDeleteDatFiles_;

  DQMFileIterator fiterator_;

  std::unique_ptr<StreamerInputFile> streamReader_;
  std::shared_ptr<EventSkipperByID> eventSkipperByID_;
  TriggerSelectorPtr eventSelector_;
};

}  //end-of-namespace-def

#endif
