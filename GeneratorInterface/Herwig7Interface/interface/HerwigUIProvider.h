/**
Marco A. Harrendorf
**/
#ifndef SRC_HERWIG_UIProvider_H
#define SRC_HERWIG_UIProvider_H

#include <Herwig/API/HerwigUI.h>
#include <iostream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace Herwig {

/**
 * HerwigUIProvider is an implementation of the HerwigUI interface getting its input from CMSSW configs.
 * 
 */
class HerwigUIProvider : public HerwigUI {
public:

  /// Constructor from the arguments provided by CMSSW
  HerwigUIProvider(const edm::ParameterSet &pset, std::string inputFileName, RunMode::Mode);

  /// Destructor to leave a clean ThePEG::Repository behind
  ~HerwigUIProvider();

  /// Requested Herwig run mode
  RunMode::Mode runMode() const { return runMode_; }
  void setSeed(long seed){ seed_=seed; }


  /// Try to resume execution from an earlier interrupted run.
  bool resume() const { return resume_; }

  /// Require verbose progress markers
  bool tics() const { return tics_; }

  /// A user-defined tag to append to the run name.
  std::string tag() const { return tag_; }

  /// Name of the file to be read
  std::string inputfile() const { return inputfile_; }

  /// Repository name to operate on
  std::string repository() const { return repository_; }

  /// Name of the setup file to be read, to modify the repository
  std::string setupfile() const { return setupfile_; }
 
  std::string integrationList() const { return integrationList_; }


  const std::vector<std::string> & 
  prependReadDirectories() const { return prependReadDirectories_; }

  const std::vector<std::string> & 
  appendReadDirectories() const { return appendReadDirectories_; }

  long N() const { return  nEvents_; }
  int seed() const { return seed_; }
  int jobs() const { return jobs_; }
  unsigned int jobSize() const { return jobsize_; }
  unsigned int maxJobs() const { return maxjobs_; }  

  void quitWithHelp() const;

  void quit() const;

   /// Return the standard out stream to be used
  virtual std::ostream& outStream() const { return std::cout; }

   /// Return the standard err stream to be used
  virtual std::ostream& errStream() const { return std::cerr; }

  /// Return the standard in stream to be used
  virtual std::istream& inStream() const { return std::cin; }

  /**
  *  Change run mode of Herwig
  *  Additionally pass the name of the new input file
  *
  *  Function sets jobsize_ and maxjobs_ accordingly to the run modes.
  **/
  void setRunMode(RunMode::Mode runMode, const edm::ParameterSet &pset, std::string inputFile = "");


private:

  RunMode::Mode runMode_;

  bool resume_;
  bool tics_;
  std::string tag_;

  std::string inputfile_;
  std::string repository_;
  std::string setupfile_;

  std::string integrationList_;

  std::vector<std::string> prependReadDirectories_;
  std::vector<std::string> appendReadDirectories_;

  long nEvents_;
  int seed_;
  int jobs_;
  unsigned int jobsize_;
  unsigned int maxjobs_;

};

}

#endif
