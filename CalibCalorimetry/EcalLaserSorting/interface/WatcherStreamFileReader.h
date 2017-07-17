#ifndef IOPool_Streamer_StreamerFileReader_h
#define IOPool_Streamer_StreamerFileReader_h

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <iostream>
#include <deque>

/** This module is an source module reading continously file 
 * as they are copied in the input directory.
 * The processed file is moved to directoryt inprocessDir before being
 * processed. Once it is processed it is moved to processedDir.
 * To prevent processing files before their transfer is finished, it is waited
 * than file size is stable during one second before the file is processed. 
 * This protection is obviously not full proof, especially to transfer lag.
 */

namespace edm{
  class StreamerInputFile;
}

class WatcherStreamFileReader{
public:
  WatcherStreamFileReader(edm::ParameterSet const& pset);
  ~WatcherStreamFileReader();

  const InitMsgView* getHeader(); 
  const EventMsgView* getNextEvent();
  const bool newHeader(); 

  edm::StreamerInputFile* getInputFile();

  void closeFile();
  
private:
  /** Directory to look for streamer files
   */
  std::string inputDir_;
  
  /** Streamer file name pattern list
   */
  std::vector<std::string> filePatterns_; 

  /** Directory where file are moved during processing
   */
  std::string inprocessDir_;


  /** Directory where file must be moved once processed
   */
  std::string processedDir_;

  /** Directory where file must be moved if file is unreadble (e.g empty size)
   */
  std::string corruptedDir_;
  
  /** Cached input file stream
   */
  std::auto_ptr<edm::StreamerInputFile> streamerInputFile_;

  std::string fileName_;

  std::string tokenFile_;

  int timeOut_;
  
  std::deque<std::string> filesInQueue_;

  bool end_;
  
  int verbosity_;

  std::string fileListCmd_;

  std::string curDir_;

};

#endif

