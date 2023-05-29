//#define USE_STORAGE_MANAGER

#ifdef USE_STORAGE_MANAGER
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#else  //USE_STORAGE_MANAGER not defined
#ifndef _LARGEFILE64_SOURCE
#define _LARGEFILE64_SOURCE
#endif  //_LARGEFILE64_SOURCE not defined
#define _FILE_OFFSET_BITS 64
#include <cstdio>
#endif  //USE_STORAGE_MANAGER defined

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "EventFilter/EcalRawToDigi/interface/MatacqRawEvent.h"
#include "EventFilter/EcalRawToDigi/src/MatacqDataFormatter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <string>
#include <cinttypes>
#include <fstream>
#include <memory>

#include <sys/time.h>

struct NullOut {
  NullOut& operator<<(std::ostream& (*pf)(std::ostream&)) { return *this; }
  template <typename T>
  inline NullOut& operator<<(const T& a) {
    return *this;
  }
};

class MatacqProducer : public edm::one::EDProducer<> {
public:
  enum calibTrigType_t { laserType = 4, ledType = 5, tpType = 6, pedType = 7 };

private:
#ifdef USE_STORAGE_MANAGER
  typedef IOOffset filepos_t;
  typedef std::unique_ptr<Storage> FILE_t;
#else
  typedef off_t filepos_t;
  typedef FILE* FILE_t;
#endif
  struct MatacqEventId {
    MatacqEventId() : run(0), orbit(0) {}
    MatacqEventId(uint32_t r, uint32_t o) : run(r), orbit(o) {}

    /** Run number
     */
    uint32_t run;

    /** Orbit id
     */
    uint32_t orbit;

    bool operator<(const MatacqEventId& a) {
      return (this->run < a.run) || ((this->run == a.run) && (this->orbit < a.orbit));
    }

    bool operator>(const MatacqEventId& a) {
      return (this->run > a.run) || ((this->run == a.run) && (this->orbit > a.orbit));
    }

    bool operator==(const MatacqEventId& a) { return !((*this) < a || (*this) > a); }
  };

  /** Estimates matacq event position in a file from its orbit id. This
   * estimator requires that every event in the file has the same length. A
   * linear extrapolation of pos=f(orbit) function from first and last event
   * is performed. It gives only a rough estimate, relevant only to initiliaze
   * the event search.
   */
  class PosEstimator {
    //Note: a better estimate could be obtained by using segment of linear
    //functions. In such implementation, the estimator must be updated
    //each time a point with wrong estimate has been found.
  public:
    PosEstimator() : eventLength_(0), orbitStepMean_(0), firstOrbit_(0), invalid_(true), verbosity_(0) {}
    void init(MatacqProducer* mp);
    bool invalid() const { return invalid_; }
    int64_t pos(int orb) const;
    int eventLength() const { return eventLength_; }
    int firstOrbit() const { return firstOrbit_; }
    void verbosity(int verb) { verbosity_ = verb; }

  private:
    int eventLength_;
    int orbitStepMean_;
    int firstOrbit_;
    bool invalid_;
    int verbosity_;
  };

public:
  /** Constructor
   * @param params seletive readout parameters
   */
  explicit MatacqProducer(const edm::ParameterSet& params);

  /** Destructor
   */
  ~MatacqProducer() override;

  /** Produces the EDM products
   * @param CMS event
   * @param eventSetup event conditions
   */
  void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;

private:
  /** Add matacq digi to the event
   * @param event the event
   * @param digiInstanceName_ name to give to the matacq digi instance
   */
  void addMatacqData(edm::Event& event);

  /** Retrieve the file containing a given matacq event
   * @param runNumber Number of the run the matacq event is looking from
   * @param orbitId Id of the orbit of the matacq event
   * @param fileChange if not null pointer, set to true if the file changed.
   * @return true if file retrieval succeeded, false otherwise.
   * found.
   */
  bool getMatacqFile(uint32_t runNumber, uint32_t orbitId, bool* fileChange = nullptr);

  bool getMatacqEvent(uint32_t runNumber, int32_t orbitId, bool fileChange);
  /*,bool doWrap = false, std::streamoff maxPos = -1);*/

  uint32_t getRunNumber(edm::Event& ev) const;
  uint32_t getOrbitId(edm::Event& ev) const;

  bool getOrbitRange(uint32_t& firstOrb, uint32_t& lastOrb);

  int getCalibTriggerType(edm::Event& ev) const;

  /** Loading orbit correction table from file. @see orbitOffsetFile_
   */
  void loadOrbitOffset();

  /** Move input file read pointer. On failure file is rewind.
   * @param buf buffer to store read data
   * @param n   size of data block
   * @param mess text to insert in the eventual error message.
   * @return true on success, false on failure
   */
  bool mseek(filepos_t offset, int whence = SEEK_SET, const char* mess = nullptr);

  bool mtell(filepos_t& pos);

  /** Read a data block from input file. On failure file position is restored
   * and if position restoring fails, file is rewind.
   * @param buf buffer to store read data
   * @param n   size of data block
   * @param mess text to insert in the eventual error message.
   * @param peek if true file position is restored after the data read
   * @return true on success, false on failure
   */
  bool mread(char* buf, size_t n, const char* mess = nullptr, bool peek = false);

  bool mcheck(const std::string& name);

  bool mopen(const std::string& name);

  void mclose();

  bool misOpened();

  bool meof();

  bool mrewind();

  bool msize(filepos_t& s);

  void newRun(int prevRun, int newRun);

  static std::string runSubDir(uint32_t runNumber);

private:
  std::vector<std::string> fileNames_;

  /** Instance name to use for the produced Matacq digi collection
   */
  std::string digiInstanceName_;

  /** Instance name to use for the produced Matacq raw data collection
   */
  std::string rawInstanceName_;

  /** Parameter to switch module timing.
   */
  bool timing_;

  /** Parameter to disable matacq data production. For timing purpose.
   */
  bool disabled_;

  /** Verbosity level
   */
  int verbosity_;

  /** Swictch for Matacq digi producion
   */
  bool produceDigis_;

  /** Switch for Matacq FED raw data production
   */
  bool produceRaw_;

  /** Name of the raw data collection the Matacq data must be merge to
   * if merging is enabled.
   */
  edm::InputTag inputRawCollection_;

  /** EDM token to access the raw data collection the Matacq data must be merge to
   * if merging is enabled.
   */
  edm::EDGetTokenT<FEDRawDataCollection> inputRawCollectionToken_;

  /** Switch for merging Matacq raw data with existing raw data
   * collection.
   */
  bool mergeRaw_;

  /** When true look for matacq data independently of trigger type.
   */
  bool ignoreTriggerType_;

  MatacqRawEvent matacq_;

  /** Stream of currently opened matacq file
   */
  FILE_t inFile_;

  static const int bufferSize = 30000;  //must greater or equal to maximum
  //                                      matacq event size.
  std::vector<unsigned char> data_;
  MatacqDataFormatter formatter_;
  const static int orbitTolerance_;
  uint32_t openedFileRunNumber_;
  int32_t lastOrb_;
  int fastRetrievalThresh_;

  PosEstimator posEstim_;

  timeval startTime_;

  /** File name of table with orbit offset between
   * matacq event and DCC. Used to recover data suffering from orbit
   * miss-synchonization
   */
  std::string orbitOffsetFile_;

  /** Orbit offset table. @see orbitOffsetFile_
   */
  std::map<uint32_t, uint32_t> orbitOffset_;

  /** Switch for orbit ID correction. @see orbitOffsetFile_
   */
  bool doOrbitOffset_;

  /** Name of currently opened matacq file
   */
  std::string inFileName_;

  static const int matacqFedId_ = 655;

  struct stats_t {
    double nEvents;
    double nLaserEventsWithMatacq;
    double nNonLaserEventsWithMatacq;
  } stats_;

  const static stats_t stats_init;
  /** Log file name
   */
  std::string logFileName_;

  /** Log file
   */
  std::ofstream logFile_;

  /** counter for event skipping
   */
  int eventSkipCounter_;

  /** Number of events to skip in case of error
   */
  int onErrorDisablingEvtCnt_;

  /** Name of file to log timing
   */
  std::string timeLogFile_;
  /** Buffer for timing
   */
  timeval timer_;

  /** Output stream to log code timing
   */
  std::ofstream timeLog_;

  /** Switch for code timing.
   */
  bool logTiming_;

  /** Number of the currently processed run
   */
  uint32_t runNumber_;
};

#include <csignal>
#include <cstdio>
#include <iomanip>
#include <iostream>

#include <glob.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fmt/printf.h>
#include <boost/algorithm/string.hpp>

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalMatacqDigi.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "EventFilter/EcalRawToDigi/src/Majority.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

using namespace std;
using namespace boost;
using namespace edm;

// #undef LogInfo
// #define LogInfo(a) cout << "INFO " << a << ": "
// #undef LogWarning
// #define LogWarning(a) cout << "WARN " << a << ": "
// #undef LogDebug
// #define LogDebug(a) cout << "DBG " << a << ": "

//verbose mode for matacq event retrieval debugging:
//static const bool searchDbg = false;

//laser freq is 1 every 112 orbit => >80 orbit
const int MatacqProducer::orbitTolerance_ = 80;

const MatacqProducer::stats_t MatacqProducer::stats_init = {0, 0, 0};

static std::string now() {
  struct timeval t;
  gettimeofday(&t, nullptr);

  char buf[256];
  strftime(buf, sizeof(buf), "%F %R %S s", localtime(&t.tv_sec));
  buf[sizeof(buf) - 1] = 0;

  stringstream buf2;
  buf2 << buf << " " << ((t.tv_usec + 500) / 1000) << " ms";

  return buf2.str();
}

MatacqProducer::MatacqProducer(const edm::ParameterSet& params)
    : fileNames_(params.getParameter<std::vector<std::string> >("fileNames")),
      digiInstanceName_(params.getParameter<string>("digiInstanceName")),
      rawInstanceName_(params.getParameter<string>("rawInstanceName")),
      timing_(params.getUntrackedParameter<bool>("timing", false)),
      disabled_(params.getParameter<bool>("disabled")),
      verbosity_(params.getUntrackedParameter<int>("verbosity", 0)),
      produceDigis_(params.getParameter<bool>("produceDigis")),
      produceRaw_(params.getParameter<bool>("produceRaw")),
      inputRawCollection_(params.getParameter<edm::InputTag>("inputRawCollection")),
      mergeRaw_(params.getParameter<bool>("mergeRaw")),
      ignoreTriggerType_(params.getParameter<bool>("ignoreTriggerType")),
      matacq_(nullptr, 0),
      inFile_(nullptr),
      data_(bufferSize),
      openedFileRunNumber_(0),
      lastOrb_(0),
      fastRetrievalThresh_(0),
      orbitOffsetFile_(params.getUntrackedParameter<std::string>("orbitOffsetFile", "")),
      inFileName_(""),
      stats_(stats_init),
      logFileName_(params.getUntrackedParameter<std::string>("logFileName", "matacqProducer.log")),
      eventSkipCounter_(0),
      onErrorDisablingEvtCnt_(params.getParameter<int>("onErrorDisablingEvtCnt")),
      timeLogFile_(params.getUntrackedParameter<std::string>("timeLogFile", "")),
      runNumber_(0) {
  if (verbosity_ >= 4)
    cout << "[Matacq " << now() << "] in MatacqProducer ctor" << endl;

  gettimeofday(&timer_, nullptr);

  if (!timeLogFile_.empty()) {
    timeLog_.open(timeLogFile_.c_str());
    if (timeLog_.fail()) {
      cout << "[LaserSorter " << now() << "] "
           << "Failed to open file " << timeLogFile_ << " to log timing.\n";
      logTiming_ = false;
    } else {
      logTiming_ = true;
    }
  }

  posEstim_.verbosity(verbosity_);

  logFile_.open(logFileName_.c_str(), ios::app | ios::out);

  if (logFile_.bad()) {
    throw cms::Exception("FileOpen") << "Failed to open file " << logFileName_ << " for logging.\n";
  }

  inputRawCollectionToken_ = consumes<FEDRawDataCollection>(params.getParameter<InputTag>("inputRawCollection"));

  if (produceDigis_) {
    if (verbosity_ > 0)
      cout << "[Matacq " << now()
           << "] registering new "
              "EcalMatacqDigiCollection product with instance name '"
           << digiInstanceName_ << "'\n";
    produces<EcalMatacqDigiCollection>(digiInstanceName_);
  }

  if (produceRaw_) {
    if (verbosity_ > 0)
      cout << "[Matacq " << now()
           << "] registering new FEDRawDataCollection "
              "product with instance name '"
           << rawInstanceName_ << "'\n";
    produces<FEDRawDataCollection>(rawInstanceName_);
  }

  startTime_.tv_sec = startTime_.tv_usec = 0;
  if (!orbitOffsetFile_.empty()) {
    doOrbitOffset_ = true;
    loadOrbitOffset();
  } else {
    doOrbitOffset_ = false;
  }
  if (verbosity_ >= 4)
    cout << "[Matacq " << now() << "] exiting MatacqProducer ctor" << endl;
}

void MatacqProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  if (verbosity_ >= 4)
    cout << "[Matacq " << now() << "] in MatacqProducer::produce" << endl;
  if (logTiming_) {
    timeval t;
    gettimeofday(&t, nullptr);

    timeLog_ << t.tv_sec << "." << setfill('0') << setw(3) << (t.tv_usec + 500) / 1000 << setfill(' ') << "\t"
             << (t.tv_usec - timer_.tv_usec) * 1. + (t.tv_sec - timer_.tv_sec) * 1.e6 << "\t";
    timer_ = t;
  }

  if (startTime_.tv_sec == 0)
    gettimeofday(&startTime_, nullptr);
  ++stats_.nEvents;
  if (disabled_)
    return;
  const uint32_t runNumber = getRunNumber(event);
  if (runNumber != runNumber_) {
    newRun(runNumber_, runNumber);
  }
  addMatacqData(event);

  if (logTiming_) {
    timeval t;
    gettimeofday(&t, nullptr);
    timeLog_ << (t.tv_usec - timer_.tv_usec) * 1. + (t.tv_sec - timer_.tv_sec) * 1.e6 << "\n";
    timer_ = t;
  }
}

void MatacqProducer::addMatacqData(edm::Event& event) {
  edm::Handle<FEDRawDataCollection> sourceColl;
  event.getByToken(inputRawCollectionToken_, sourceColl);

  std::unique_ptr<FEDRawDataCollection> rawColl;
  if (produceRaw_) {
    if (mergeRaw_) {
      rawColl = std::make_unique<FEDRawDataCollection>(*sourceColl);
    } else {
      rawColl = std::make_unique<FEDRawDataCollection>();
    }
  }

  auto digiColl = std::make_unique<EcalMatacqDigiCollection>();

  if (eventSkipCounter_ == 0) {
    if (sourceColl->FEDData(matacqFedId_).size() > 4 && !produceRaw_) {
      //input raw data collection already contains matacqData
      formatter_.interpretRawData(sourceColl->FEDData(matacqFedId_), *digiColl);
    } else {
      bool isLaserEvent = (getCalibTriggerType(event) == laserType);

      //      cout << "---> " << (ignoreTriggerType_?"yes":"no") << " " << getCalibTriggerType(event) << endl;

      if (isLaserEvent || ignoreTriggerType_) {
        const uint32_t runNumber = getRunNumber(event);
        const uint32_t orbitId = getOrbitId(event);

        LogInfo("Matacq") << "Run " << runNumber << "\t Orbit " << orbitId << "\n";

        bool fileChange;
        if (doOrbitOffset_) {
          map<uint32_t, uint32_t>::iterator it = orbitOffset_.find(runNumber);
          if (it == orbitOffset_.end()) {
            LogWarning("Matacq") << "Orbit offset not found for run " << runNumber
                                 << ". No orbit correction will be applied.";
          }
        }

        if (getMatacqFile(runNumber, orbitId, &fileChange)) {
          //matacq file retrieval succeeded
          LogInfo("Matacq") << "Matacq data file found for "
                            << "run " << runNumber << " orbit " << orbitId;
          if (getMatacqEvent(runNumber, orbitId, fileChange)) {
            if (produceDigis_) {
              formatter_.interpretRawData(matacq_, *digiColl);
            }
            if (produceRaw_) {
              uint32_t dataLen64 = matacq_.getParsedLen();
              if (dataLen64 > bufferSize * 8 || matacq_.getDccLen() != dataLen64) {
                LogWarning("Matacq") << " Error in Matacq event fragment length! "
                                     << "DCC len: " << matacq_.getDccLen()
                                     << "*8 Bytes, Parsed len: " << matacq_.getParsedLen() << "*8 Bytes.  "
                                     << "Matacq data will not be included for this event.\n";
              } else {
                rawColl->FEDData(matacqFedId_).resize(dataLen64 * 8);
                copy(data_.begin(), data_.begin() + dataLen64 * 8, rawColl->FEDData(matacqFedId_).data());
              }
            }
            LogInfo("Matacq") << "Associating matacq data with orbit id " << matacq_.getOrbitId()
                              << " to dcc event with orbit id " << orbitId << std::endl;
            if (isLaserEvent) {
              ++stats_.nLaserEventsWithMatacq;
            } else {
              ++stats_.nNonLaserEventsWithMatacq;
            }
          } else {
            if (isLaserEvent) {
              LogWarning("Matacq") << "No matacq data found for laser event "
                                   << "of run " << runNumber << " orbit " << orbitId;
            }
          }
        } else {
          LogWarning("Matacq") << "No matacq file found for event " << event.id();
        }
      }
    }
    if (eventSkipCounter_ > 0) {  //error occured for this events
      //                       and some events will be skipped following
      //                       to this error.
      LogInfo("Matacq") << " [" << now() << "] " << eventSkipCounter_
                        << " next events will be skipped, following to an "
                        << "error on the last processed event, "
                        << "which is expected to be persistant.";
    }
  } else {
    --eventSkipCounter_;
  }

  if (produceRaw_) {
    if (verbosity_ > 1)
      cout << "[Matacq " << now() << "] "
           << "Adding FEDRawDataCollection collection "
           << " to event.\n";
    event.put(std::move(rawColl), rawInstanceName_);
  }

  if (produceDigis_) {
    if (verbosity_ > 1)
      cout << "[Matacq " << now() << "] "
           << "Adding EcalMatacqDigiCollection collection "
           << " to event.\n";
    event.put(std::move(digiColl), digiInstanceName_);
  }
}

// #if 0
// bool
// MatacqProducer::getMatacqEvent(std::ifstream& f,
// 			       uint32_t runNumber,
// 			       uint32_t orbitId,
// 			       bool doWrap,
// 			       std::streamoff maxPos){
//   bool found = false;
//   streampos startPos = f.tellg();

//   while(!f.eof()
// 	&& !found
// 	&& (maxPos<0 || f.tellg()<=maxPos)){
//     const streamsize headerSize = 8*8;
//     f.read((char*)&data_[0], headerSize);
//     if(f.eof()) break;
//     int32_t orb = MatacqRawEvent::getOrbitId(&data_[0], headerSize);
//     uint32_t len = MatacqRawEvent::getDccLen(&data_[0], headerSize);
//     uint32_t run = MatacqRawEvent::getRunNum(&data_[0], headerSize);
//     //     cout << "Matacq: orbit = " << orb
//     // 	 << " len = " << len
//     // 	 << " run = " << run << endl;
//     if((abs(orb-(int32_t)orbitId) < orbitTolerance_)
//        && (runNumber==0 || runNumber==run)){
//       found = true;
//       //reads the rest of the event:
//       if(data_.size() < len*8){
// 	throw cms::Exception("Matacq") << "Buffer overflow";
//       }
//       f.read((char*)&data_[0]+headerSize, len*8-headerSize);
//       matacq_ = MatacqRawEvent((unsigned char*)&data_[0], len*8);
//     } else{
//       //moves to next event:
//       f.seekg(len*8 - headerSize, ios::cur);
//     }
//   }

//   f.clear(); //clears eof error to allow seekg
//   if(doWrap && !found){
//     f.seekg(0, ios::beg);
//     found =  getMatacqEvent(f, runNumber, orbitId, false, startPos);
//   }
//   return found;
// }
//#endif

bool MatacqProducer::getMatacqEvent(uint32_t runNumber, int32_t orbitId, bool fileChange) {
  filepos_t startPos;
  if (!mtell(startPos))
    return false;

  int32_t startOrb = -1;
  const size_t headerSize = 8 * 8;
  if (mread((char*)&data_[0], headerSize, "Reading matacq header", true)) {
    startOrb = MatacqRawEvent::getOrbitId(&data_[0], headerSize);
    if (startOrb < 0)
      startOrb = 0;
  } else {
    if (verbosity_ > 2) {
      cout << "[Matacq " << now()
           << "] Failed to read matacq header. Moved to start of "
              " the file.\n";
    }
    mrewind();
    if (mread((char*)&data_[0], headerSize, "Reading matacq header", true)) {
      startPos = 0;
      startOrb = MatacqRawEvent::getOrbitId(&data_[0], headerSize);
    } else {
      if (verbosity_ > 2)
        cout << "[Matacq " << now() << "] Looks like matacq file is empty"
             << "\n";
      return false;
    }
  }

  if (verbosity_ > 2)
    cout << "[Matacq " << now() << "] Last read orbit: " << lastOrb_ << " looking for orbit " << orbitId
         << ". Current file position: " << startPos << " Orbit at current position: " << startOrb << "\n";

  //  f.clear();
  bool didCoarseMove = false;

  //FIXME: case where posEtim_.invalid() is false
  if (!posEstim_.invalid() && (abs(lastOrb_ - orbitId) > fastRetrievalThresh_)) {
    filepos_t pos = posEstim_.pos(orbitId);

    //    struct stat st;
    filepos_t fsize = -1;
    //    if(0==stat(inFileName_.c_str(), &st)){
    if (msize(fsize)) {
      //      const int64_t fsize = st.st_size;
      if (0 != posEstim_.eventLength() && pos > fsize) {
        //estimated position is beyong end of file
        //-> move to beginning of last event:
        int64_t evtSize = posEstim_.eventLength() * sizeof(uint64_t);
        pos = ((int64_t)fsize / evtSize - 1) * evtSize;
        if (verbosity_ > 2) {
          cout << "[Matacq " << now()
               << "] Estimated position was beyond end of file. "
                  "Changed to "
               << pos << "\n";
        }
      }
    } else {
      LogWarning("Matacq") << "Failed to access file " << inFileName_ << ".";
    }
    if (pos >= 0) {
      if (verbosity_ > 2)
        cout << "[Matacq " << now() << "] jumping to estimated position " << pos << "\n";
      mseek(pos, SEEK_SET, "Jumping to estimated event position");
      if (mread((char*)&data_[0], headerSize, "Reading matacq header", true)) {
        didCoarseMove = true;
      } else {
        //estimated position might have been beyond the end of the file,
        //try, with original position:
        didCoarseMove = false;
        if (!mread((char*)&data_[0], headerSize, "Reading event header", true)) {
          return false;
        }
      }
    } else {
      if (verbosity_)
        cout << "[Matacq " << now()
             << "] Event orbit outside of orbit range "
                "of matacq data file events\n";
      return false;
    }
  }

  int32_t orb = MatacqRawEvent::getOrbitId(&data_[0], headerSize);

  if (didCoarseMove) {
    //autoadjustement of threshold for coarse move:
    if (abs(orb - orbitId) > fastRetrievalThresh_) {
      if (verbosity_ > 2)
        cout << "[Matacq " << now() << "] Fast retrieval threshold increased from " << fastRetrievalThresh_;
      fastRetrievalThresh_ = 2 * abs(orb - orbitId);
      if (verbosity_ > 2)
        cout << " to " << fastRetrievalThresh_ << "\n";
    }

    //if coarse move did not improve situation, rolls back:
    if (startOrb > 0 && (abs(orb - orbitId) > abs(startOrb - orbitId))) {
      if (verbosity_ > 2)
        cout << "[Matacq " << now() << "] Estimation (-> orbit " << orb
             << ") "
                "was worst than original position (-> orbit "
             << startOrb << "). Restoring position (" << startPos << ").\n";
      mseek(startPos, SEEK_SET);
      mread((char*)&data_[0], headerSize, "Reading event header", true);
      orb = MatacqRawEvent::getOrbitId(&data_[0], headerSize);
    }
  }

  bool searchBackward = (orb > orbitId) ? true : false;
  //BEWARE: len must be signed, because we are using latter in the code (-len)
  //expression
  int len = (int)MatacqRawEvent::getDccLen(&data_[0], headerSize);

  if (len == 0) {
    cout << "[Matacq " << now() << "] read DCC length is null! Cancels matacq event search "
         << " and move matacq file pointer to beginning of the file. "
         << "(" << __FILE__ << ":" << __LINE__ << ")."
         << "\n";
    //rewind(f);
    mrewind();
    return false;
  }

  enum state_t { searching, found, failed } state = searching;

  while (state == searching) {
    orb = MatacqRawEvent::getOrbitId(&data_[0], headerSize);
    len = (int)MatacqRawEvent::getDccLen(&data_[0], headerSize);
    uint32_t run = MatacqRawEvent::getRunNum(&data_[0], headerSize);
    if (verbosity_ > 3) {
      filepos_t pos = -1;
      mtell(pos);
      cout << "[Matacq " << now() << "] Header read at file position " << pos << ":  orbit = " << orb
           << " len = " << len << "x8 Byte"
           << " run = " << run << "\n";
    }
    if ((abs(orb - orbitId) < orbitTolerance_) && (runNumber == 0 || runNumber == run)) {
      state = found;
      lastOrb_ = orb;
      //reads the rest of the event:
      if ((int)data_.size() < len * 8) {
        throw cms::Exception("Matacq") << "Buffer overflow";
      }
      if (verbosity_ > 2)
        cout << "[Matacq " << now()
             << "] Event found. Reading "
                " matacq event."
             << "\n";
      if (!mread((char*)&data_[0], len * 8, "Reading matacq event")) {
        if (verbosity_ > 2)
          cout << "[Matacq " << now() << "] Failed to read matacq event."
               << "\n";
        state = failed;
      }
      matacq_ = MatacqRawEvent((unsigned char*)&data_[0], len * 8);
    } else {
      if ((searchBackward && (orb < orbitId)) || (!searchBackward && (orb > orbitId))) {  //search ended
        lastOrb_ = orb;
        state = failed;
        if (verbosity_ > 2)
          cout << "[Matacq " << now() << "] No matacq data found for run " << run << ", orbit ID " << orbitId << "."
               << "\n";
      } else {
        off_t offset = (searchBackward ? -len : len) * 8;
        lastOrb_ = orb;
        if (verbosity_ > 3) {
          cout << "[Matacq " << now() << "] In matacq file, moving " << abs(offset) << " byte "
               << (offset > 0 ? "forward" : "backward") << ".\n";
        }

        if (mseek(offset, SEEK_CUR, (searchBackward ? "Moving to previous event" : "Moving to next event")) &&
            mread((char*)&data_[0], headerSize, "Reading event header", true)) {
        } else {
          if (!searchBackward)
            mseek(-len * 8, SEEK_CUR, "Moving to start of last complete event");
          state = failed;
        }
      }
    }
  }

  if (state == found) {
    filepos_t pos = -1;
    filepos_t fsize = -1;
    mtell(pos);
    msize(fsize);
    if (pos == fsize - 1) {  //last byte.
      if (verbosity_ > 2) {
        cout << "[Matacq " << now()
             << "] Event found was at the end of the file. Moving "
                "stream position to beginning of this event."
             << "\n";
      }
      mseek(-(int)len * 8 - 1, SEEK_CUR, "Moving to beginning of last matacq event");
    }
  }
  return (state == found);
}

bool MatacqProducer::getMatacqFile(uint32_t runNumber, uint32_t orbitId, bool* fileChange) {
  if (openedFileRunNumber_ != 0 && openedFileRunNumber_ == runNumber) {
    uint32_t firstOrb, lastOrb;
    bool goodRange = getOrbitRange(firstOrb, lastOrb);
    //    if(orbitId < firstOrb || orbitId > lastOrb) continue;
    if (goodRange && firstOrb <= orbitId && orbitId <= lastOrb) {
      if (fileChange != nullptr)
        *fileChange = false;
      return misOpened();
    }
  }

  if (fileNames_.empty())
    return false;

  const string runNumberFormat = "%08d{,_*}";
  string sRunNumber = fmt::sprintf(runNumberFormat, runNumber);
  //cout << "Run number string: " << sRunNumber << "\n";
  bool found = false;
  string fname;
  uint32_t maxOrb = 0;
  //we make two iterations to handle the case where the event is procesed
  //before the matacq data are available. In such case we would have
  //orbitId > maxOrb (maxOrb: orbit of last written matacq event)
  //  for(int itry = 0; itry < 2 && (orbitId > maxOrb); ++itry){
  for (int itry = 0; itry < 1 && (orbitId > maxOrb); ++itry) {
    if (itry > 0) {
      int n_sec = 1;
      std::cout << "[Matacq " << now() << "] Event orbit id (" << orbitId
                << ") goes "
                   "beyound the range of available one. Waiting for "
                << n_sec
                << " seconds in case "
                   "it was not written yet to disk.";
      sleep(n_sec);
    }

    for (unsigned i = 0; i < fileNames_.size() && !found; ++i) {
      fname = fileNames_[i];
      boost::algorithm::replace_all(fname, "%run_subdir%", runSubDir(runNumber));
      boost::algorithm::replace_all(fname, "%run_number%", sRunNumber);

      glob_t g;
      int rc = glob(fname.c_str(), GLOB_BRACE, nullptr, &g);
      if (rc) {
        if (verbosity_ > 1) {
          switch (rc) {
            case GLOB_NOSPACE:
              std::cout << "[Matacq " << now()
                        << "] Running out of memory while calling glob function to look for matacq file paths\n";
              break;
            case GLOB_ABORTED:
              std::cout << "[Matacq " << now()
                        << "] Read error while calling glob function to look for matacq file paths\n";
              break;
            case GLOB_NOMATCH:
              //ok. No message to report.
              break;
          }
          continue;
        }
      }  //rc
      for (unsigned iglob = 0; iglob < g.gl_pathc; ++iglob) {
        char* thePath = g.gl_pathv[iglob];
        //FIXME: add sanity check on the path
        static std::atomic<int> nOpenErrors{0};
        const int maxOpenErrors = 50;
        if (!mopen(thePath) && nOpenErrors < maxOpenErrors) {
          std::cout << "[Matacq " << now() << "] Failed to open file " << thePath;
          ++nOpenErrors;
          if (nOpenErrors == maxOpenErrors) {
            std::cout << nOpenErrors << "This is the " << maxOpenErrors
                      << "th occurence of this error. Report of this error is now disabled.\n";
          } else {
            std::cout << "\n";
          }
        }
        uint32_t firstOrb;
        uint32_t lastOrb;
        bool goodRange = getOrbitRange(firstOrb, lastOrb);
        std::cout << "Get orbit range " << (goodRange ? "succeeded" : "failed") << ". Range: " << firstOrb << "..."
                  << lastOrb << "\n";
        if (goodRange && lastOrb > maxOrb)
          maxOrb = lastOrb;
        if (goodRange && firstOrb <= orbitId && orbitId <= lastOrb) {
          found = true;
          //continue;
          fname = thePath;
          if (verbosity_ > 1)
            std::cout << "[Matacq " << now() << "] Switching to file " << fname << "\n";
          break;
        }
      }  //next iglob
      globfree(&g);
    }  //next filenames
  }    //next itry

  if (found) {
    LogInfo("Matacq") << "Uses matacq data file: '" << fname << "'\n";
  } else {
    if (verbosity_ >= 0)
      cout << "[Matacq " << now()
           << "] no matacq file found "
              "for run "
           << runNumber << ", orbit " << orbitId << "\n";
    eventSkipCounter_ = onErrorDisablingEvtCnt_;
    openedFileRunNumber_ = 0;
    if (fileChange != nullptr)
      *fileChange = false;
    return false;
  }

  if (found) {
    openedFileRunNumber_ = runNumber;
    lastOrb_ = 0;
    posEstim_.init(this);
    if (fileChange != nullptr)
      *fileChange = true;
    return true;
  } else {
    return false;
  }
}

uint32_t MatacqProducer::getRunNumber(edm::Event& ev) const { return ev.run(); }

uint32_t MatacqProducer::getOrbitId(edm::Event& ev) const {
  //on CVS HEAD (June 4, 08), class Event has a method orbitNumber()
  //we could use here. The code would be shorten to:
  //return ev.orbitNumber();
  //we have to deal with what we have in current CMSSW releases:
  edm::Handle<FEDRawDataCollection> rawdata;
  ev.getByToken(inputRawCollectionToken_, rawdata);
  if (!(rawdata.isValid())) {
    throw cms::Exception("NotFound") << "No FED raw data collection found. ECAL raw data are "
                                        "required to retrieve the orbit ID";
  }

  int orbit = 0;
  for (int id = 601; id <= 654; ++id) {
    if (!FEDNumbering::inRange(id))
      continue;
    const FEDRawData& data = rawdata->FEDData(id);
    const int orbitIdOffset64 = 3;
    if (data.size() >= 8 * (orbitIdOffset64 + 1)) {  //orbit id is in 4th 64-bit word
      const unsigned char* pOrbit = data.data() + orbitIdOffset64 * 8;
      int thisOrbit = pOrbit[0] | (pOrbit[1] << 8) | (pOrbit[2] << 16) | (pOrbit[3] << 24);
      if (orbit != 0 && thisOrbit != 0 && abs(orbit - thisOrbit) > orbitTolerance_) {
        //throw cms::Exception("EventCorruption")
        //  << "Orbit ID inconsitency in DCC headers";
        LogWarning("EventCorruption") << "Orbit ID inconsitency in DCC headers";
        orbit = 0;
        break;
      }
      if (thisOrbit != 0)
        orbit = thisOrbit;
    }
  }

  if (orbit == 0) {
    //    throw cms::Exception("NotFound")
    //  << "Failed to retrieve orbit ID of event "<< ev.id();
    LogWarning("NotFound") << "Failed to retrieve orbit ID of event " << ev.id();
  }
  return orbit;
}

int MatacqProducer::getCalibTriggerType(edm::Event& ev) const {
  edm::Handle<FEDRawDataCollection> rawdata;
  ev.getByToken(inputRawCollectionToken_, rawdata);
  if (!(rawdata.isValid())) {
    throw cms::Exception("NotFound") << "No FED raw data collection found. ECAL raw data are "
                                        "required to retrieve the trigger type";
  }

  Majority<int> stat;
  for (int id = 601; id <= 654; ++id) {
    if (!FEDNumbering::inRange(id))
      continue;
    const FEDRawData& data = rawdata->FEDData(id);
    const int detailedTrigger32 = 5;
    if (data.size() >= 4 * (detailedTrigger32 + 1)) {
      const unsigned char* pTType = data.data() + detailedTrigger32 * 4;
      int tType = pTType[1] & 0x7;
      stat.add(tType);
    }
  }
  double p;
  int tType = stat.result(&p);
  if (p < 0) {
    //throw cms::Exception("NotFound") << "No ECAL DCC data found\n";
    LogWarning("NotFound") << "No ECAL DCC data found\n";
    tType = -1;
  }
  if (p < .8) {
    //throw cms::Exception("EventCorruption") << "Inconsitency in detailed trigger type indicated in ECAL DCC data headers\n";
    LogWarning("EventCorruption") << "Inconsitency in detailed trigger type indicated in ECAL DCC data headers\n";
    tType = -1;
  }
  return tType;
}

void MatacqProducer::PosEstimator::init(MatacqProducer* mp) {
  mp->mrewind();

  const size_t headerSize = 8 * 8;
  unsigned char data[headerSize];
  if (!mp->mread((char*)data, headerSize)) {
    if (verbosity_)
      cout << "[Matacq " << now() << "] reached end of file!\n";
    firstOrbit_ = eventLength_ = orbitStepMean_ = 0;
    return;
  } else {
    firstOrbit_ = MatacqRawEvent::getOrbitId(data, headerSize);
    eventLength_ = MatacqRawEvent::getDccLen(data, headerSize);
    if (verbosity_ > 1)
      cout << "[Matacq " << now() << "] First event orbit: " << firstOrbit_ << " event length: " << eventLength_
           << "*8 byte\n";
  }

  mp->mrewind();

  if (eventLength_ == 0) {
    if (verbosity_)
      cout << "[Matacq " << now() << "] event length is null!" << endl;
    return;
  }

  filepos_t s = -1;
  mp->msize(s);

  if (s == -1) {
    if (verbosity_)
      cout << "[Matacq " << now() << "] File is missing!" << endl;
    orbitStepMean_ = 0;
    return;
  } else if (s == 0) {
    if (verbosity_)
      cout << "[Matacq " << now() << "] File is empty!" << endl;
    orbitStepMean_ = 0;
    return;
  }

  //number of complete events:
  const unsigned nEvents = s / eventLength_ / 8;

  if (verbosity_ > 1)
    cout << "[Matacq " << now() << "] File size: " << s << " Number of events: " << nEvents << endl;

  //position of last complete events:
  off_t last = (nEvents - 1) * (off_t)eventLength_ * 8;
  mp->mseek(last,
            SEEK_SET,
            "Moving to beginning of last complete "
            "matacq event");
  if (!mp->mread((char*)data, headerSize, "Reading matacq header", true)) {
    LogWarning("Matacq") << "Fast matacq event retrieval failure. "
                            "Falling back to safe retrieval mode.";
    orbitStepMean_ = 0;
  }

  int32_t lastOrb = MatacqRawEvent::getOrbitId(data, headerSize);
  int32_t lastLen = MatacqRawEvent::getDccLen(data, headerSize);

  if (verbosity_ > 1)
    cout << "[Matacq " << now() << "] Last event orbit: " << lastOrb << " last event length: " << lastLen << endl;

  //some consistency check
  if (lastLen != eventLength_) {
    LogWarning("Matacq")
        //throw cms::Exception("Matacq")
        << "Fast matacq event retrieval failure: it looks like "
           "the matacq file contains events of different sizes.";
    //      " Falling back to safe retrieval mode.";
    invalid_ = false;      //true;
    orbitStepMean_ = 112;  //0;
    return;
  }

  orbitStepMean_ = (lastOrb - firstOrbit_) / nEvents;

  if (verbosity_ > 1)
    cout << "[Matacq " << now() << "] Orbit step mean: " << orbitStepMean_ << "\n";

  invalid_ = false;
}

int64_t MatacqProducer::PosEstimator::pos(int orb) const {
  if (orb < firstOrbit_)
    return -1;
  uint64_t r = orbitStepMean_ != 0 ? (((uint64_t)(orb - firstOrbit_)) / orbitStepMean_) * eventLength_ * 8 : 0;
  if (verbosity_ > 2)
    cout << "[Matacq " << now() << "] Estimated Position for orbit  " << orb << ": " << r << endl;
  return r;
}

MatacqProducer::~MatacqProducer() {
  mclose();
  timeval t;
  gettimeofday(&t, nullptr);
  if (logTiming_ && startTime_.tv_sec != 0) {
    //not using logger, to allow timing with different logging options
    cout << "[Matacq " << now()
         << "] Time elapsed between first event and "
            "destruction of MatacqProducer: "
         << ((t.tv_sec - startTime_.tv_sec) * 1. + (t.tv_usec - startTime_.tv_usec) * 1.e-6) << "s\n";
  }
}

void MatacqProducer::loadOrbitOffset() {
  std::ifstream f(orbitOffsetFile_.c_str());
  if (f.bad()) {
    throw cms::Exception("Matacq") << "Failed to open orbit ID correction file '" << orbitOffsetFile_ << "'\n";
  }

  cout << "[Matacq " << now() << "] "
       << "Offset to substract to Matacq events Orbit ID: \n"
       << "#Run Number\t Offset\n";

  string s;
  stringstream buf;
  while (f.eof()) {
    getline(f, s);
    if (s[0] == '#') {  //comment
      //skip line:
      f.ignore(numeric_limits<streamsize>::max(), '\n');
      continue;
    }
    buf.str("");
    buf << s;
    int run;
    int orbit;
    buf >> run;
    buf >> orbit;
    if (buf.bad()) {
      throw cms::Exception("Matacq") << "Syntax error in Orbit offset file '" << orbitOffsetFile_ << "'";
    }
    cout << run << "\t" << orbit << "\n";
    orbitOffset_.insert(pair<int, int>(run, orbit));
  }
}

#ifdef USE_STORAGE_MANAGER
bool MatacqProducer::mseek(filepos_t offset, int whence, const char* mess) {
  if (0 == inFile_.get())
    return false;
  try {
    Storage::Relative wh;
    if (whence == SEEK_SET)
      wh = Storage::SET;
    else if (whence == SEEK_CUR)
      wh = Storage::CURRENT;
    else if (whence == SEEK_END)
      wh = Storage::END;
    else
      throw cms::Exception("Bug") << "Bug found in " << __FILE__ << ": " << __LINE__ << "\n";

    inFile_->position(offset, wh);
  } catch (cms::Exception& e) {
    if (verbosity_) {
      cout << "[Matacq " << now() << "] ";
      if (mess)
        cout << mess << ". ";
      cout << "Random access error on input matacq file. ";
      if (whence == SEEK_SET)
        cout << "Failed to seek absolute position " << offset;
      else if (whence == SEEK_CUR)
        cout << "Failed to move " << offset << " bytes forward";
      else if (whence == SEEK_END)
        cout << "Failed to seek position at " << offset << " bytes before end of file";
      cout << ". Reopening file. " << e.what() << "\n";
      mopen(inFileName_);
      return false;
    }
  }
  return true;
}

bool MatacqProducer::mtell(filepos_t& pos) {
  if (0 == inFile_.get())
    return false;
  pos = inFile_->position();
  return true;
}

bool MatacqProducer::mread(char* buf, size_t n, const char* mess, bool peek) {
  if (0 == inFile_.get())
    return false;

  filepos_t pos = -1;
  if (!mtell(pos))
    return false;

  bool rc = false;
  try {
    rc = (n == inFile_->xread(buf, n));
  } catch (cms::Exception& e) {
    if (verbosity_) {
      cout << "[Matacq " << now() << "] ";
      if (mess)
        cout << mess << ". ";
      cout << "Read failure from input matacq file: " << e.what() << "\n";
    }
    //recovering from error:
    mopen(inFileName_);
    mseek(pos);
    return false;
  }
  if (peek) {  //asked to restore original file position
    mseek(pos);
  }
  return rc;
}

bool MatacqProducer::msize(filepos_t& s) {
  if (inFile_.get() == 0)
    return false;
  s = inFile_.get()->size();
  return true;
}

bool MatacqProducer::mrewind() {
  Storage* file = inFile_.get();
  if (file == 0)
    return false;
  try {
    file->rewind();
  } catch (cms::Exception e) {
    if (verbosity_)
      cout << "Exception cautgh while rewinding file " << inFileName_ << ": " << e.what() << ". "
           << "File will be reopened.";
    return mopen(inFileName_);
  }
  return true;
}

bool MatacqProducer::mcheck(const std::string& name) { return StorageFactory::get()->check(name); }

bool MatacqProducer::mopen(const std::string& name) {
  //close already opened file if any:
  mclose();

  try {
    inFile_ = unique_ptr<Storage>(StorageFactory::get()->open(name, IOFlags::OpenRead));
    inFileName_ = name;
  } catch (cms::Exception& e) {
    LogWarning("Matacq") << e.what();
    inFile_.reset();
    inFileName_ = "";
    return false;
  }
  return true;
}

void MatacqProducer::mclose() {
  if (inFile_.get() != 0) {
    inFile_->close();
    inFile_.reset();
  }
}

bool MatacqProducer::misOpened() { return inFile_.get() != 0; }

bool MatacqProducer::meof() {
  if (inFile_.get() == 0)
    return true;
  return inFile_->eof();
}

#else  //USE_STORAGE_MANAGER not defined
bool MatacqProducer::mseek(off_t offset, int whence, const char* mess) {
  if (nullptr == inFile_)
    return false;
  const int rc = fseeko(inFile_, offset, whence);
  if (rc != 0 && verbosity_) {
    cout << "[Matacq " << now() << "] ";
    if (mess)
      cout << mess << ". ";
    cout << "Random access error on input matacq file. "
            "Rewind file.\n";
    mrewind();
  }
  return rc == 0;
}

bool MatacqProducer::mtell(filepos_t& pos) {
  if (nullptr == inFile_)
    return false;
  pos = ftello(inFile_);
  return pos != -1;
}

bool MatacqProducer::mread(char* buf, size_t n, const char* mess, bool peek) {
  if (nullptr == inFile_)
    return false;
  off_t pos = ftello(inFile_);
  bool rc = (pos != -1) && (1 == fread(buf, n, 1, inFile_));
  if (!rc) {
    if (verbosity_) {
      cout << "[Matacq " << now() << "] ";
      if (mess)
        cout << mess << ". ";
      cout << "Read failure from input matacq file.\n";
    }
    clearerr(inFile_);
  }
  if (peek || !rc) {  //need to restore file position
    if (0 != fseeko(inFile_, pos, SEEK_SET)) {
      if (verbosity_) {
        cout << "[Matacq " << now() << "] ";
        if (mess)
          cout << mess << ". ";
        cout << "Failed to restore file position of "
                "before read error. Rewind file.\n";
      }
      //rewind(inFile_.get());
      mrewind();
      lastOrb_ = 0;
    }
  }
  return rc;
}

bool MatacqProducer::msize(filepos_t& s) {
  if (nullptr == inFile_)
    return false;
  struct stat buf;
  if (0 != fstat(fileno(inFile_), &buf)) {
    s = 0;
    return false;
  } else {
    s = buf.st_size;
    return true;
  }
}

bool MatacqProducer::mrewind() {
  if (nullptr == inFile_)
    return false;
  clearerr(inFile_);
  return fseeko(inFile_, 0, SEEK_SET) != 0;
}

bool MatacqProducer::mcheck(const std::string& name) {
  struct stat dummy;
  return 0 == stat(name.c_str(), &dummy);
  //   if(stat(name.c_str(), &dummy)==0){
  //     return true;
  //   } else{
  //     cout << "[Matacq " << now() << "] Failed to stat file '"
  // 	 << name.c_str() << "'. "
  // 	 << "Error " << errno << ": " << strerror(errno) << "\n";
  //     return false;
  //   }
}

bool MatacqProducer::mopen(const std::string& name) {
  if (inFile_ != nullptr)
    mclose();
  inFile_ = fopen(name.c_str(), "r");
  if (inFile_ != nullptr) {
    inFileName_ = name;
    return true;
  } else {
    inFileName_ = "";
    return false;
  }
}

void MatacqProducer::mclose() {
  if (inFile_ != nullptr)
    fclose(inFile_);
  inFile_ = nullptr;
}

bool MatacqProducer::misOpened() { return inFile_ != nullptr; }

bool MatacqProducer::meof() {
  if (nullptr == inFile_)
    return true;
  return feof(inFile_) == 0;
}

#endif  //USE_STORAGE_MANAGER defined

std::string MatacqProducer::runSubDir(uint32_t runNumber) {
  int millions = runNumber / (1000 * 1000);
  int thousands = (runNumber - millions * 1000 * 1000) / 1000;
  int units = runNumber - millions * 1000 * 1000 - thousands * 1000;
  return fmt::sprintf("%03d/%03d/%03d", millions, thousands, units);
}

void MatacqProducer::newRun(int prevRun, int newRun) {
  runNumber_ = newRun;
  eventSkipCounter_ = 0;
  logFile_ << "[" << now() << "] Event count for run " << runNumber_ << ": "
           << "total: " << stats_.nEvents << ", "
           << "Laser event with Matacq data: " << stats_.nLaserEventsWithMatacq << ", "
           << "Non laser event (according to DCC header) with Matacq data: " << stats_.nNonLaserEventsWithMatacq << "\n"
           << flush;

  stats_.nEvents = 0;
  stats_.nLaserEventsWithMatacq = 0;
  stats_.nNonLaserEventsWithMatacq = 0;
}

bool MatacqProducer::getOrbitRange(uint32_t& firstOrb, uint32_t& lastOrb) {
  filepos_t pos = -1;
  filepos_t fsize = -1;
  mtell(pos);
  msize(fsize);
  const unsigned headerSize = 8 * 8;
  unsigned char header[headerSize];
  //FIXME: Don't we need here to rewind?
  mseek(0);
  if (!mread((char*)header, headerSize, nullptr, false))
    return false;
  firstOrb = MatacqRawEvent::getOrbitId(header, headerSize);
  int len = (int)MatacqRawEvent::getDccLen(header, headerSize);
  //number of complete events. If last event is partially written,
  //it won't be included in the count.
  unsigned nEvts = fsize / (len * 8);
  //Position of last complete event:
  filepos_t lastEvtPos = (filepos_t)(nEvts - 1) * len * 8;
  //  std::cout << "Move to position : " << lastEvtPos
  //	    << "(" << (nEvts - 1) << "*" << len << "*" << 64 << ")"
  //<< "\n";
  mseek(lastEvtPos);
  filepos_t tmp;
  mtell(tmp);
  //std::cout << "New position, sizeof(tmp): " << tmp << "," << sizeof(tmp) << "\n";
  mread((char*)header, headerSize, nullptr, false);
  lastOrb = MatacqRawEvent::getOrbitId(header, headerSize);

  //restore file position:
  mseek(pos);

  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MatacqProducer);
