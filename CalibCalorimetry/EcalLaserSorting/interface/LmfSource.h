#ifndef SourceModule_H
#define SourceModule_H

#include <vector>
#include <string>
#include <inttypes.h>
#include <fstream>

#include "FWCore/Sources/interface/ProducerSourceBase.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

class LmfSource: public edm::ProducerSourceBase{
private:
  struct IndexRecord{
    uint32_t orbit;
    uint32_t filePos;
    //    bool operator<(const IndexRecord& i) const { return orbit < i.orbit; }
  };
  
public:
  LmfSource(const edm::ParameterSet& pset,
	       const edm::InputSourceDescription& isd);
  virtual ~LmfSource(){}
  
private:
  /** Called by the framework after setRunAndEventInfo()
   */
  virtual void produce(edm::Event &e);

  /** Callback funtion to set run and event information
   * (lumi block, run number, event number, timestamp)
   * Called by the framework before produce()
   */
  virtual bool setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& time);

  bool openFile(int iFile);
  
  bool readFileHeader();

  /** Read event from current opened file. Called by readEvent, which
   * deals with file chaining
   * Beware: readEventHeader must be called beforehand.
   * @param doSkip if true skip event instead of reading it
   * @return true iff succeeded
   */
  bool readEventWithinFile(bool doSkip);
  
  /** timeval to string conversion
   * @param t timestamp
   * @return human readable character string
   */
  std::string toString(edm::TimeValue_t& t) const;

  
private:
  /** List of names of the input files
   */
  std::vector<std::string> fileNames_;

  /** Index of the current process file
   */
  int iFile_;

  /** Buffer for FED block collection
   */
  FEDRawDataCollection fedColl_;
  
  /** empty fed block
   */
  FEDRawData emptyFedBlock_;

  /** FED ID present in FED data collection
   * (only one FED at a time)
   */
  int fedId_;

  /** Buffer for event header readout
   */
  std::vector<uint32_t> header_;

  static unsigned fileHeaderSize;
  
  /** Buffer for file header readout
   */
  std::vector<uint32_t> fileHeader_;


  /** Minimal LMF data format version supported.
   */
  static unsigned char minDataFormatVersion_;
  
  /** Maximal LMF data format version supported.
   */
  static unsigned char maxDataFormatVersion_;

  /** Filtering events. Used for prescale.
   * @return true of event accepted, false if rejected
   */
  bool filter() const;

  /** Reading next event
   * @param doSkip if true skip event instead of reading it
   * @return true iff read out succeeded
   */
  bool readEvent(bool doSkip = false);

  /** Move to next event within the same file.
   * Called by nextEvent method.
   * @return false in case of failure (end of file)
   */
  bool nextEventWithinFile();
  
  /** Checks paths specified in fileNames_ and remove eventual
   * file: prefix.
   * @throw cms::Exception in case of a non valid path
   */
  void checkFileNames();

  /** Reads event index table from input file. readFileHeader()
   * must be called beforehand.
   */
  void readIndexTable();
  
  uint64_t timeStamp_;
  uint32_t lumiBlock_;
  uint32_t runNum_;
  uint32_t bx_;
  uint32_t eventNum_;
  uint32_t orbitNum_;
  
  unsigned char dataFormatVers_;

  std::ifstream in_;

  /** Flags of last event read success: true->succeeded, false->failed
   */
  bool rcRead_;

  unsigned preScale_;

  /** Sequential number of event.
   */
  uint32_t iEvent_;

  /** Sequential number of event reset at each newly opened file
   */
  uint32_t iEventInFile_;

  uint32_t indexTablePos_;

  int calibTrig_;
  
  int nFeds_;

  /** Table with file position of each event order by event time
   * (orbit id used as time measurement).
   */
  std::vector<IndexRecord> indexTable_;

  /** Limit of number of events to prevent exhausting memory
   * with indexTable_ in case of file corruption.
   */
  static const unsigned maxEvents_ = 1<<20;

  /** Limit on event size to prevent exhausting memory
   * in case of error in the event size read from file.
   */
  static const unsigned maxEventSize_ = 1<<20; //1MB. (full DCC event is 49kB)
  
  /** Switch for enabling reading event in ordered using
   * event index table
   */
  bool orderedRead_;

  /** enable reading input file list from text file
   *  and keep watching the text file for updates
   */
  bool watchFileList_;

  /** name of the textfile with the input file list
   */
  std::string fileListName_;

  /** absolute path from which filename in fileListName_ is valid
   */
  std::string inputDir_;

  /** currently open file
   */
  std::string currentFileName_;

  std::ifstream fileList_;

  /** seconds to sleep before checking fileList_ for updates
   */
  int nSecondsToSleep_;

  /** Debugging level
   */
  int verbosity_;
};
#endif //SourceModule_H not defined

