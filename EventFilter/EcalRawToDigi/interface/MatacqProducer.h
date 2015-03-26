#ifndef PRODUCER_H
#define PRODUCER_H

//#define USE_STORAGE_MANAGER

#ifdef USE_STORAGE_MANAGER
#  include "Utilities/StorageFactory/interface/Storage.h"
#  include "Utilities/StorageFactory/interface/StorageFactory.h"
#else //USE_STORAGE_MANAGER not defined
#  ifndef _LARGEFILE64_SOURCE
#    define _LARGEFILE64_SOURCE
#  endif //_LARGEFILE64_SOURCE not defined
#  define  _FILE_OFFSET_BITS 64 
#  include <stdio.h>
#endif //USE_STORAGE_MANAGER defined


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "EventFilter/EcalRawToDigi/interface/MatacqRawEvent.h"
#include "EventFilter/EcalRawToDigi/src/MatacqDataFormatter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <string>
#include <inttypes.h>
#include <fstream>
#include <memory>

#include <sys/time.h>

struct NullOut{
  NullOut&
  operator<<(std::ostream& (*pf)(std::ostream&)){
    return *this;
  }
  template<typename T>
  inline NullOut& operator<<(const T& a){
    return *this;
  }
};

class MatacqProducer : public edm::EDProducer
{
public:
  enum calibTrigType_t{
    laserType = 4,
    ledType   = 5,
    tpType    = 6,
    pedType   = 7
  };
private:
#ifdef USE_STORAGE_MANAGER
  typedef IOOffset filepos_t;
  typedef std::auto_ptr<Storage> FILE_t;
#else
  typedef off_t filepos_t;
  typedef FILE* FILE_t;
#endif
  struct MatacqEventId{
    MatacqEventId(): run(0), orbit(0){}
    MatacqEventId(uint32_t r, uint32_t o): run(r), orbit(o){}
					  
    /** Run number
     */
    uint32_t run;
    
    /** Orbit id
     */
    uint32_t orbit;
    
    bool
    operator<(const MatacqEventId& a){
      return (this->run < a.run)
	|| ((this->run == a.run) && (this->orbit < a.orbit));
    }
    
    bool
    operator>(const MatacqEventId& a){
      return (this->run > a.run)
	|| ((this->run == a.run) && (this->orbit > a.orbit));
    }
    
    bool
    operator==(const MatacqEventId& a){
      return !((*this) < a || (*this) > a);
    }
  };


  /** Estimates matacq event position in a file from its orbit id. This
   * estimator requires that every event in the file has the same length. A
   * linear extrapolation of pos=f(orbit) function from first and last event
   * is performed. It gives only a rough estimate, relevant only to initiliaze
   * the event search.
   */
  class PosEstimator{
    //Note: a better estimate could be obtained by using segment of linear
    //functions. In such implementation, the estimator must be updated
    //each time a point with wrong estimate has been found.
  public:
    PosEstimator():eventLength_(0), orbitStepMean_(0), firstOrbit_(0),
		   invalid_(true), verbosity_(0) { }
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
  explicit
  MatacqProducer(const edm::ParameterSet& params);

  /** Destructor
   */
  ~MatacqProducer();

  /** Produces the EDM products
   * @param CMS event
   * @param eventSetup event conditions
   */
  virtual void
  produce(edm::Event& event, const edm::EventSetup& eventSetup);

private:
  /** Add matacq digi to the event
   * @param event the event
   * @param digiInstanceName_ name to give to the matacq digi instance
   */
  void
  addMatacqData(edm::Event& event);

  /** Retrieve the file containing a given matacq event
   * @param runNumber Number of the run the matacq event is looking from
   * @param orbitId Id of the orbit of the matacq event
   * @param fileChange if not null pointer, set to true if the file changed.
   * @return true if file retrieval succeeded, false otherwise.
   * found.
   */
  bool
  getMatacqFile(uint32_t runNumber, uint32_t orbitId, bool* fileChange =0);

  bool
  getMatacqEvent(uint32_t runNumber, int32_t orbitId,
		 bool fileChange);
  /*,bool doWrap = false, std::streamoff maxPos = -1);*/

  uint32_t getRunNumber(edm::Event& ev) const;
  uint32_t getOrbitId(edm::Event& ev) const;
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
  bool mseek(filepos_t offset, int whence = SEEK_SET, const char* mess = 0);

  bool mtell(filepos_t& pos);
  
  /** Read a data block from input file. On failure file position is restored
   * and if position restoring fails, file is rewind.
   * @param buf buffer to store read data
   * @param n   size of data block
   * @param mess text to insert in the eventual error message.
   * @param peek if true file position is restored after the data read
   * @return true on success, false on failure
   */
  bool mread(char* buf, size_t n, const char* mess = 0, bool peek = false);

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

  static const int bufferSize =  30000; //must greater or equal to maximum
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
  std::map<uint32_t,uint32_t> orbitOffset_;
  
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

#endif 
