#ifndef FWCore_MessageLogger_JobReport_h
#define FWCore_MessageLogger_JobReport_h
// -*- C++ -*-
//
// Package:     MessageLogger
// Class  :     JobReport
//
/**\class edm::JobReport

Description: A service that collections job handling information.

Usage:
The JobReport service collects 'job handling' information (currently
file handling) from several sources, collates the information, and
at appropriate intervals, reports the information to the job report,
through the MessageLogger.

CMS-THREADING Some notes about thread safety:

1. It is assumed non-experts will turn the job report on
and look at the XML output and NOT call the available public
functions. Many of the available service functions can be
called at times that are not thread safe (causing data
races or interleaved output). The following notes are for
Framework experts and detail the assumptions made when
modifying JobReport to run in the multithreaded Framework.

2. We assume the primary input source performs its activity
serially. There can be multiple secondary input sources running
concurrently.

3. We assume the following sequence of activities where the lines
of asterisks indicate synchronization points:

****************
open primary input file
nothing else running concurrently
Also respondToOpenInputFiles serially
****************
open output files serially
nothing else running concurrently
****************
The primary source runs its other tasks concurrently
with the secondary sources running their tasks and
modules running their tasks.
****************
close primary input file
nothing else running concurrently
****************
close output files serially
nothing else running concurrently
****************
repeat the above (the output file opens and closes
are optional except for the first and last)
***********************
All endJob and postEndJob activities occur serially
not concurrently
************************

4. We assume that a single instance of an OutputModule
will only be running on one thread at a time.
Other instances of that type of OutputModule may be running
concurrently. There are several functions where this is
an issue. We have discussed that in the future we might
decide to run the OutputModule for multiple events
concurrently. That would require further modifications
of either the JobReport or the OutputModule.

5. For Primary and SecondaryFile input sources (not
SecondarySource) the calls to reportBranchName from the
delayed reader need to be serialized.

*/
//
// Original Author:  Marc Paterno
//

/*
Changes Log 1: 2009/01/14 10:29:00, Natalia Garcia Nebot
        Modified and added some methods to report CPU and memory information from /proc/cpuinfo
        and /proc/meminfo files and Memory statistics
*/

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "FWCore/Utilities/interface/InputType.h"

#include <atomic>
#include <cstddef>
#include <iosfwd>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_vector.h"

namespace edm {

    class JobReport {
    public:

      typedef unsigned int RunNumber;
      typedef std::size_t Token;

      struct RunReport {
        RunNumber runNumber;
        std::set<unsigned int> lumiSections;
      };

      /**\struct InputFile

      Description: Holds information about an InputFile.

      Usage: The struct InputFile is a collection of the information that
      Data Handling wishes to accumulate about the use of a file that has
      been opened for input.
      */

      struct InputFile {
        typedef std::vector<std::string> StringVector;

        std::string     logicalFileName;
        std::string     physicalFileName;
        std::string     catalog;
        std::string     inputType; // primaryFiles, secondaryFiles, mixingFiles
        std::string     inputSourceClassName; // class which created the file
        std::string     moduleLabel;   // name of class instance
        std::string     guid;
        std::size_t     numEventsRead;
        StringVector    branchNames;
        std::map<RunNumber, RunReport> runReports;
        bool            fileHasBeenClosed;
        std::set<std::string> fastClonedBranches;
      };

      /**\struct OutputFile

      Description: Holds information about an OutputFile.

      Usage: The struct OutputFile is a collection of the information that
      Data Handling wishes to accumulate about the use of a file that has
      been opened for output.
      */

      struct OutputFile {

        typedef InputFile::StringVector StringVector;

        std::string     logicalFileName;
        std::string     physicalFileName;
        std::string     catalog;
        std::string     outputModuleClassName;
        std::string     moduleLabel;   // name of class instance
        std::string     guid;
        std::string     dataType;
        std::string     branchHash;
        std::size_t     numEventsWritten;
        StringVector    branchNames;
        std::vector<Token> contributingInputs;
        tbb::concurrent_vector<Token> contributingInputsSecSource;
        std::map<std::string, bool> fastCopyingInputs;
        std::map<RunNumber, RunReport> runReports;
        bool            fileHasBeenClosed;
      };

      class AtomicLongLong {
      public:
        AtomicLongLong() : value_(0) {}
        AtomicLongLong(AtomicLongLong const& r) : value_(r.value_.load()) {}
        std::atomic<long long>& value() { return value_; }
        std::atomic<long long> const& value() const { return value_; }
      private:
        std::atomic<long long> value_;
      };

      struct JobReportImpl {

        JobReportImpl& operator=(JobReportImpl const&) = delete;
        JobReportImpl(JobReportImpl const&) = delete;

        InputFile& getInputFileForToken(InputType inputType, Token t);
        OutputFile& getOutputFileForToken(Token t);

        /*
         * Add the input file token provided to every output
         * file currently available.
         * Used whenever a new input file is opened, it's token
         * is added to all open output files as a contributor
         */
        void insertInputForOutputs(InputType inputType, Token t);

        /*
         * Associate a Lumi Section to all open output files
         *
         */
        void associateLumiSection(JobReport::Token token, unsigned int runNumber, unsigned int lumiSection);

        /*
         * Associate a Lumi Section to all open input files
         *
         */
        void associateInputLumiSection(unsigned int runNumber, unsigned int lumiSection);

        /*
         * Associate a run to all open output files
         */
        void associateRun(JobReport::Token token, unsigned int runNumber);

        /*
         * Associate a run to all open output files
         */
        void associateInputRun(unsigned int runNumber);

        /*
         * Write an InputFile object to the Logger
         * Generate XML string for InputFile instance and dispatch to
         * job report via MessageLogger
         */
        void writeInputFile(InputFile const& f);

        /*
         * Write an OutputFile object to the Logger
         * Generate an XML string for the OutputFile provided and
         * dispatch it to the logger
         * Contributing input tokens are resolved to the input LFN and PFN
         *
         * TODO: We have not yet addressed the issue where we cleanup not
         * contributing input files.
         * Also, it is possible to get fake input to output file mappings
         * if an input file is open already when a new output file is opened
         * but the input gets closed without contributing events to the
         * output file due to filtering etc.
         *
         */
        void writeOutputFile(OutputFile const& f);

        /*
         *  Flush all open files to logger in event of a problem.
         */
        void flushFiles(void);

        JobReportImpl(std::ostream* iOst): printedReadBranches_(false), ost_(iOst) {}

        std::vector<InputFile> inputFiles_;
        tbb::concurrent_vector<InputFile> inputFilesSecSource_;
        std::vector<OutputFile> outputFiles_;
        std::map<std::string, long long> readBranches_;
        std::map<std::string, long long> readBranchesSecFile_;
        tbb::concurrent_unordered_map<std::string, AtomicLongLong> readBranchesSecSource_;
        bool printedReadBranches_;
        std::vector<InputFile>::size_type lastOpenedPrimaryInputFile_;
        std::ostream* ost_;
      };

      JobReport();
      //Does not take ownership of pointer
      JobReport(std::ostream* outputStream);

      JobReport& operator=(JobReport const&) = delete;
      JobReport(JobReport const&) = delete;

      ~JobReport();

      /// New output file for child
      void childAfterFork(std::string const& jobReportFile, unsigned int childIndex, unsigned int numberOfChildren);

      void parentBeforeFork(std::string const& jobReportFile, unsigned int numberOfChildren);

      void parentAfterFork(std::string const& jobReportFile);

      /// Report that an input file has been opened.
      /// The returned Token should be used for later identification
      /// of this file.
      Token inputFileOpened(std::string const& physicalFileName,
                            std::string const& logicalFileName,
                            std::string const& catalog,
                            std::string const& inputType,
                            std::string const& inputSourceClassName,
                            std::string const& moduleLabel,
                            std::string const& guid,
                            std::vector<std::string> const& branchNames);

      /// Report that an event has been read from
      /// the file identified by the given Token.
      void eventReadFromFile(InputType inputType, Token fileToken);

      ///
      /// Report the data type of a file after it has been opened
      /// Needed since the data type isn't known until an event has been
      /// read
      // CMS-THREADING Current implementation requires an instance of an
      // OuputModule run on only one thread at a time.
      void reportDataType(Token fileToken, std::string const& dataType);


      /// Report that the input file identified by the given Token has
      /// been closed. An exception will be thrown if the given Token
      /// was not obtained from inputFileOpened.
      void inputFileClosed(InputType inputType, Token fileToken);

      /// Report that an output file has been opened.
      /// The returned Token should be used for later identification
      /// of this file.
      Token outputFileOpened(std::string const& physicalFileName,
                             std::string const& logicalFileName,
                             std::string const& catalog,
                             std::string const& outputModuleClassName,
                             std::string const& moduleLabel,
                             std::string const& guid,
                             std::string const& dataType,
                             std::string const& branchHash,
                             std::vector<std::string> const& branchNames);

      /// Report that the event with the given id has been written to
      /// the file identified by the given Token.
      // CMS-THREADING Current implementation requires an instance of an
      // OuputModule run on only one thread at a time.
      void eventWrittenToFile(Token fileToken, RunNumber_t run, EventNumber_t event);

      /// Report that the output file identified by the given Token has
      /// been closed. An exception will be thrown if the given Token
      /// was not obtained from outputFileOpened.
      void outputFileClosed(Token fileToken);

      void reportSkippedEvent(RunNumber_t run, EventNumber_t event);

      /// API for reporting a Run to the job report.
      /// for output files, call only if Run is written to
      /// the output file
      ///
      void reportRunNumber(JobReport::Token token, unsigned int run);

      /// API for reporting a Lumi Section to the job report.
      /// for output files, call only if lumi section is written to
      /// the output file
      ///
      void reportLumiSection(JobReport::Token token, unsigned int run, unsigned int lumiSectId);

      ///
      /// API for reporting a Lumi Section to the job report.
      /// for input files, call only if lumi section is physically read
      /// from the input file
      ///
      void reportInputLumiSection(unsigned int run, unsigned int lumiSectId);

      ///
      /// API to report a run read from input
      ///
      void reportInputRunNumber(unsigned int run);

      ///
      /// Report an exception, providing details of the problem as
      /// a short description (Eg "XXXError") and a long description
      /// (Eg "XXX crashed because...")
      /// Also overload this method to accept an optional standard exit code
      void reportError(std::string const& shortDesc,
                       std::string const& longDesc,
                       int const& exitCode);

      ///
      /// Report Skipped File
      ///
      /// Report that a file has been skipped due to it not being
      /// found.
      void reportSkippedFile(std::string const& pfn, std::string const& lfn);

      void reportFallbackAttempt(std::string const& pfn, std::string const& lfn, std::string const& err);

      void reportAnalysisFile(std::string const& fileName,
                              std::map<std::string, std::string> const& fileData) ;

      ///
      /// Report Memory statistics
      /// ALTERNATE FORM - USING THIS MAY NECESSITATE CHANGES IN PARSING XML!
      /// Invoked by the Memory service to send an end of job
      /// summary about memory usage for inclusion in the job report
      ///
      void reportMemoryInfo(std::vector<std::string> const& memoryData);

      ///
      /// Report Message statistics
      /// Invoked by the MessageLogger service to send an end of job
      /// summary about numbers of various categories messages issued
      /// for inclusion in the job report
      ///
      void reportMessageInfo(std::map<std::string, double> const& messageData);

      ///
      /// Report read branches
      /// Invoked by the source that reads ROOT/EDM files at the end of a job
      /// to report which product branches in the Events tree have been read,
      /// with a count of the number of events for which the branch was read.
      ///
      void reportReadBranches();

      ///  Inform the job report that a branch has been read.
      void reportReadBranch(InputType inputType, std::string const& branchName);

      ///  Inform the job report that branches have been fast Cloned.
      void reportFastClonedBranches(std::set<std::string> const& fastClonedBranches, long long nEvents);

      ///
      /// Report the name of the random engine persistency file
      ///
      void reportRandomStateFile(std::string const& name);

      /*
       * Report information about fast copying. Called for each open output file
       * whenever an input file is opened.
       */
      void reportFastCopyingStatus(Token t, std::string const& inputFileName, bool fastCopying);

      ///
      /// Performance Reports
      ///
      /// Two categories:  Summary for entire job and module
      /// for a specific module
      /// Each one requires a performance metric class such
      /// as Timing, Memory, CPU, Trigger etc.
      void reportPerformanceSummary(std::string const&  metricClass,
                                    std::map<std::string, std::string> const& metrics);

      void reportPerformanceForModule(std::string const&  metricClass,
                                      std::string const&  moduleName,
                                      std::map<std::string, std::string> const& metrics);

      /// debug/test util
      std::string dumpFiles(void);

   protected:
      boost::scoped_ptr<JobReportImpl>& impl() {return impl_;}

   private:
      boost::scoped_ptr<JobReportImpl> impl_;
      std::mutex write_mutex;
   };

   std::ostream& operator<< (std::ostream& os, JobReport::InputFile const& f);
   std::ostream& operator<< (std::ostream& os, JobReport::OutputFile const& f);
}
#endif
