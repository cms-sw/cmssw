/*
 * $Id: LaserSorter.h,v 1.8 2012/10/09 19:00:18 wdd Exp $
 */

#ifndef EVENT_SELECT_H
#define EVENT_SELECT_H

#include <vector>
#include <iostream>
#include <fstream>
#include <inttypes.h>
#include "boost/ptr_container/ptr_list.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include <sys/time.h>
#include <time.h>
#include <map>

/**
 * This module is used to classify events of laser sequence acquired in 
 * a global run. 
 * Sorting: Events are grouped by bunch of consecutive events from the same 
 * FED. A file of such a bunch of events is identified by the FED or ECAL
 * sector and the luminosity block id of the first event.
 * 
 * Sorting strategy:
 * It is assumes that:
 * - one sequence FED does not overlap more than
 *   two lumi blocks. 
 * - lumi blocks are read in order
 * A FED file will be closed as soon as one event 2 lumi block ahead
 * from the first event feed in the file is read.
 *
 * File completion: while being feeding, .part is appended at the end of 
 * the each output file.Once a file is completed (see above), it is renamed
 * without the enclosing .part suffix.
 */
class LaserSorter : public edm::EDAnalyzer {
  //inner classes
private:
  struct IndexRecord{
    int orbit;
    std::streampos filePos;
    bool operator<(const IndexRecord& i) const { return orbit < i.orbit; }
  };
  
  class OutStreamRecord{
  public:
    //default ctor
    OutStreamRecord(int fedId__,
		    edm::LuminosityBlockNumber_t startingLumiBlock__,
		    std::ofstream* out__,
		    std::string& tmpFileName__,
		    std::string& finalFileName__):
      fedId_(fedId__),
      startingLumiBlock_(startingLumiBlock__),
      out_(out__),
      tmpFileName_(tmpFileName__), finalFileName_(finalFileName__),
      indexError_(false){
      indices_.reserve(indexReserve_);
    }
    
    int fedId() const { return fedId_; }
    edm::LuminosityBlockNumber_t startingLumiBlock() const{
      return startingLumiBlock_; 
    } 
    std::ofstream* out() const { return out_.get(); }
    std::string finalFileName() const { return finalFileName_; }
    std::string tmpFileName() const { return tmpFileName_; }
    std::vector<IndexRecord> * indices() { return &indices_; }
 
    
    /** Gets the list of orbits to skip. Used to update an existing file:
     * orbits of events already present in the file are excluded.
     * @return the list of orbits to skip.
     */
    std::set<uint32_t>&  excludedOrbit() { return excludedOrbit_; }   

  private:
    int fedId_;
    edm::LuminosityBlockNumber_t startingLumiBlock_;
    std::auto_ptr<std::ofstream> out_;
    std::string tmpFileName_;
    std::string finalFileName_;
    static std::string emptyString_;

    /** Index table. This map is used to index the events in the file
     * according to their orbit id. An index table is stored at the end
     * of the output files.
     */
    std::vector<IndexRecord> indices_;

    /** List of orbits to skip. Used to update an existing file: orbits
     * of events already present in the file are excluded. 
     */
    std::set<uint32_t> excludedOrbit_; 
    
    /** Used to invalidate index table in case a problem preventing
     * indexing is encountered: in principle non unicity of the orbit id.
     */
    bool indexError_;

    /** Initial memory allocation for index table (see vector::reserve()).
     */
    static size_t indexReserve_;

  };

  //typedefs:
private:
  typedef boost::ptr_list<OutStreamRecord> OutStreamList;
  
  //ctors/dtors
public:
  LaserSorter(const edm::ParameterSet&);
  ~LaserSorter();


  //methods
public:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  virtual void beginJob();

private:
  int dcc2Lme(int dccNum, int dccSide);

  /** Retrieve detailed trigger type (trigger type, DCC, side) from raw event
   * @param rawdata FED data collection
   * @param proba if not null used to store the maximum of occurence frequency
   * of the detailed trigger types (DTT) appearing in each DCC block. In normal
   * condition every DCC indicated the same DTT and this value is 1.
   * @return detailed trigger type. In case of descripancy between the DCCs, the
   * most frequent value is returned if it covers more than 80% of the present DCC
   * blocks, -1 is returned otherwise. If event does not contain any ECAL data
   * -2 is returned.
   */
  int getDetailedTriggerType(const edm::Handle<FEDRawDataCollection>& rawdata,
                             double* proba = 0);

  /** Closes output stream 2 lumi block older than the input 'lumiBlock' ID.
   * @param lumiBlock ID of the reference luminosity block.
   */
  void closeOldStreams(edm::LuminosityBlockNumber_t lumiBlock);

  /** Closes all opened output streams.
   */
  void closeAllStreams();

  
  /** Gets and eventually creates the output stream for writing the events 
   * of a given FED and luminosity block.
   * @param fedId ID of the FED the event is issued from
   * @param lumiBlock luminositu block of the event
   * @return pointer of the output stream record or null if not found.
   */
  OutStreamRecord* getStream(int fedId,
			     edm::LuminosityBlockNumber_t lumiBlock);

  /** Writes a monitoring events to an output stream.
   * @param out stream to write the event out
   * @param event EDM event, used to retrieve meta information like timestamp
   * and ID.
   * @param ID of the unique FED block of the event
   * @param data DCC data
   * @return true on success, false on failure
   * @see getStream(int, edm::LuminosityBlockNumber_t)
   */
  bool writeEvent(OutStreamRecord& out, const edm::Event& event,
		  int detailedTriggerType,
		  const FEDRawDataCollection& data);
  
  /** Writes out data of a FED
   * @param out stream to write the event out
   * @param data FED data
   * @return true on success, false on failure
   */
  bool writeFedBlock(std::ofstream& out,
                     const FEDRawData& data);

  /** Closes an output stream and removed it from opened stream records.
   * Beware: this methode modifies outStreamList_.
   * @param streamRecord record of the output stream to close.
   * @return iterator to element next to the deleted one.
   */
  OutStreamList::iterator closeOutStream(OutStreamList::iterator streamRecord);

  /** Creates an output stream. It must be ensured before calling this method
   * that the output stream is not already opened (See outStreamList_).
   * @param fedId FED ID of the event to stream out.
   * @param lumiBlock starting lumi block of the event group to write to the
   * stream.
   * @return iterator to the new stream record. outStreamList_.end() in case
   * of failure.
   */
  OutStreamList::iterator createOutStream(int fedId, 
					  edm::LuminosityBlockNumber_t lumiBlock);

  /** Writing file header for an LMF binary file
   * @param out stream of the output file
   */
  void writeFileHeader(std::ofstream& out);

  /** Write event header with event identification and timestamp.
   * @param out output stream to write to
   * @param evt event
   * @return false in case of write failure
   */
  bool writeEventHeader(std::ofstream& out,
                        const edm::Event& evt,
                        int fedId,
                        unsigned nFeds);

  
  /** Builds the file names for the group of event corresponding
   * to a FED and a starting lumi block.
   * @param fedId FED ID of the event set
   * @param lumiBlock starting luminoisty block of the event set
   * @param [out] tmpName name of the file to use when filling it.
   * @param [out] finalName name of the file once completed.
   */
  void streamFileName(int fedId, edm::LuminosityBlockNumber_t lumiBlock, 
                      std::string& tmpName, std::string& finalName);


  /** Checks if an ECAL DCC event is empty. It is considered as
   * empty if it does not contains FE data ("tower" block). So
   * an event containing SRP or TCC data can be tagged as empty by
   * this method.
   * @dccLen, if not null filled with the event length read from the
   * DCC header.
   * @nTowerBlocks if not null, filled with number of tower blocks
   * @return true if event is empty, false otherwise
   */
  bool isDccEventEmpty(const FEDRawData& data, size_t* dccLen = 0,
		       int* nTowerBlocks = 0) const;
  
  /** Computes the list of FEDs which data must be written out.
   * @param data CMS raw event
   * @param fedIds [out] list of FEDs to keep
   */
  void getOutputFedList(const edm::Event& event,
                        const FEDRawDataCollection& data,
                        std::vector<unsigned>& fedIds) const;

  /** Read index table of an LMF file.
   * @param in LMF file whose index table must be read.
   * @param inName name of the in file
   * @param outRcd record of the output file. whose the index table must be
   * copied to.
   * @param err if not nul, in case of failure filled with the error message.
   * @return true in case of success, false otherwise
   */
  bool readIndexTable(std::ifstream& in, std::string& inName,
                      OutStreamRecord& outRcd, std::string* err);

  /** Writes index table in LMF output file. stream must be positionned
   * to the place for the index table (end of file).
   * @param out stream of output file.
   * @param indices index table
   */
  bool writeIndexTable(std::ofstream& out,
		       std::vector<IndexRecord>& indices);

  bool renameAsBackup(const std::string& fileName,
                      std::string& newFileName);
  

  /** Gets format version of an LMF file. Position of file is preserved.
   * @param in stream to read the file
   * @param fileName name of the file. Used in error message.
   * @return version or -1 in case of error.
   */
  int readFormatVersion(std::ifstream& in,
                        const std::string& fileName);

  /** Help function to format a date
   */
  static std::string toString(uint64_t t);

  /** Opens output streams associated to a lumi block according to
   * already existing files. To be used when previously processed luminosity
   * block is not
   * @param lumiBlock ID of the luminosity block whose output streams
   * must be reopened.
   */
  void restoreStreamsOfLumiBlock(int lumiBlock);

  /** Retrieves DCCs which were fully read out (>=68 readout channels).
   * @param data DCC data collection
   * @return FED ids.
   */
  std::vector<int> getFullyReadoutDccs(const FEDRawDataCollection& data) const;
  
  //fields
private:
  /** Lower bound of ECAL DCC FED ID
   */
  static const int ecalDccFedIdMin_;

  /** Lower bound of ECAL DCC FED ID
   */
  static const int ecalDccFedIdMax_;

  /** Trigger type of calibration event. -1 if unkown.
   */
  int detailedTrigType_;

  /** File for logging
   */
  std::ofstream logFile_;

  /** Luminosity block of event under processing
   */
  edm::LuminosityBlockNumber_t lumiBlock_;

  /** List of output stream to write sorted
   * data
   */
  OutStreamList outStreamList_;

  /** Data format version of lmf output file
   */
  unsigned char formatVersion_;

  /** Top directory for output files
   */
  std::string outputDir_;

  /** Subdirectories for output file of each FED
   */
  std::vector<std::string> fedSubDirs_;

  /** Name of the file to log the processing time
   */
  std::string timeLogFile_;

  /** Switch to disable writing output file (for test purpose).
   */
  bool disableOutput_;

  /** Run number of event under process
   */
  edm::RunNumber_t runNumber_;

  /** Buffer for timing
   */
  timeval timer_;

  /** Output stream to log code timing
   */
  std::ofstream timeLog_;

  /** Switch for code timing.
   */
  bool timing_;

  /** name of file where list of output file is listed to
   */
  std::string outputListFile_;

  /**Switch for logging paths of the output files
   */
  bool doOutputList_;

  /** Debugging message verbosity level
   */
  int verbosity_;
  
  /** stream where list of output file is listed to
   */
  std::ofstream outputList_;

#if 0
  /**Switch for logging Orbit Id of first calibration event
   */
  bool storeFirstOrbitId_;

  /** stream where to store orbit ID of first calibration event
   */
  std::ofstream firstOrbitOut_;

  /** Path to the file to store orbit ID of first calibration event
   */
  std::string firstOrbitOutputFile_;

  /** Buffer to compute minimal orbit ID of process events
   */
  uint32_t minOrbitId_;

#endif

  /** Number of "No fully readout DCC error"
   */
  int iNoFullReadoutDccError_;


  /** Maximum number of "No fully readout DCC error" message in a run
   */
  int maxFullReadoutDccError_;  


  /** number of "ECAL DCC data" message in a run
   */
  int iNoEcalDataMess_;

  /** Maximum number of "ECAL DCC data" message in a run
   */
  int maxNoEcalDataMess_;

  /** Tolerance on lumi block spanning of a FED sequence. Subsequent events
   * of a same FED must span at most on 2*lumiBlockSpan_+1 luminosity blocks.
   *
   * <ul><li>It is important that the laser sequence scane does not pass twice on the same FED
   * within the 2*lumiBlockSpan_+1. Failing this requirement will result mixing event of
   * different passes in the same output file.
   *    <li>The number of input files opened simultinuously is proportional to 2*lumiBlockSpan_+1.
   * So increasing lumiBlockSpan_ will also increase the number of opened files and may have
   * some impact of sorting time performances.
   * </ul>
   */
  int lumiBlockSpan_;

  edm::InputTag fedRawDataCollectionTag_;

  /** FED ID associated to Matacq data
   */
  static const int matacqFedId_ = 655;

  /**
   */
  static const int indexOffset32_;

  /** Limit of number of events to prevent exhausting memory
   * with indexTable_ in case of file corruption.
   */
  static const unsigned maxEvents_ = 1<<20;

  /** Statistics on event processing
   */
  struct stats_t {
    /// number of events read out
    double nRead;
    /// number of events written out
    double nWritten;
    ///number of events with at least one DCC with an invalid Detailed trigger
    ///type value
    double nInvalidDccStrict;
    ///number of events whose DCC ID determination from DCC headers fails because
    ///of a large descrepancy between the different DCCs of because of invalid values
    double nInvalidDccWeak;
    ///number of events whose DCC ID was restored based on FED block sizes
    double nRestoredDcc;
  } stats_;
  static stats_t stats_init;
};
  
#endif //EVENT_SELECT_H not defined
