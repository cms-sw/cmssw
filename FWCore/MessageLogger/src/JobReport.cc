
// -*- C++ -*-
//
//
// 10/23/07 mf        In an attempt to get clues about (or ene) the
//                does-not-output-branches behavior, changed the
//                generic os<< lines in JobReport::JobReportImpl::writeOutputFile
//                to direct use of LogInfo.
//
// 4/8/08   mf        Encase the logdesc for in <CDATA> ... </CDATA>
//
// 6/19/08  mf        reportMessageInfo()
//
// 24 June 2008   ewv  Correct format for CDATA and for second instance of reportError

//
// Original Author:  Marc Paterno
//

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/Map.h"
#include "FWCore/Utilities/interface/EDMException.h"

// The part of tinyxml used in JobReport was reviewed and
// determined to be threadsafe.
#include "FWCore/Utilities/interface/tinyxml.h"

#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace edm {
  /*
   * Note that output formatting is spattered across these classes
   * If something outside these classes requires access to the
   * same formatting then we need to refactor it into a common library
   */
  template <typename S, typename T>
  S& formatFile(T const& f, S& os) {

    if(f.fileHasBeenClosed) {
      os << "\n<State  Value=\"closed\"/>";
    } else {
      os << "\n<State  Value=\"open\"/>";
    }

    os << "\n<LFN>" << TiXmlText(f.logicalFileName) << "</LFN>";
    os << "\n<PFN>" << TiXmlText(f.physicalFileName) << "</PFN>";
    os << "\n<Catalog>" << TiXmlText(f.catalog) << "</Catalog>";
    os << "\n<ModuleLabel>" << TiXmlText(f.moduleLabel) << "</ModuleLabel>";
    os << "\n<GUID>" << f.guid << "</GUID>";
    os << "\n<Branches>";
    for(auto const& branch : f.branchNames) {
      os << "\n  <Branch>" << TiXmlText(branch) << "</Branch>";
    }
    os << "\n</Branches>";
    return os;
  }
  /*
   * Note that output formatting is spattered across these classes
   * If something outside these classes requires access to the
   * same formatting then we need to refactor it into a common library
   */
  template <typename S>
  S& print(S& os, JobReport::InputFile const& f) {

    os << "\n<InputFile>";
    formatFile(f, os);
    os << "\n<InputType>" << f.inputType << "</InputType>";
    os << "\n<InputSourceClass>" << TiXmlText(f.inputSourceClassName)
       << "</InputSourceClass>";
    os << "\n<EventsRead>" << f.numEventsRead << "</EventsRead>";
    return os;
  }

  template <typename S>
  S& print(S& os, JobReport::OutputFile const& f) {
    formatFile(f, os);
    os << "\n<OutputModuleClass>"
       << TiXmlText(f.outputModuleClassName)
       << "</OutputModuleClass>";
    os << "\n<TotalEvents>"
       << f.numEventsWritten
       << "</TotalEvents>\n";
    os << "\n<DataType>"
       << TiXmlText(f.dataType)
       << "</DataType>\n";
    os << "\n<BranchHash>"
       << TiXmlText(f.branchHash)
       << "</BranchHash>\n";
    return os;
  }

  template <typename S>
  S& print(S& os,
           JobReport::RunReport const& rep) {
    os << "\n<Run ID=\""
       << rep.runNumber
       << "\">\n";

    for(auto il : rep.lumiSections) {
      os << "   <LumiSection ID=\"" << il << "\"/>\n";
    }
    os << "</Run>\n";
    return os;
  }

  std::ostream& operator<< (std::ostream& os, JobReport::InputFile const& f) {
    return print(os,f);
  }
  std::ostream& operator<< (std::ostream& os, JobReport::OutputFile const& f) {
    return print(os,f);
  }
  std::ostream& operator<< (std::ostream& os, JobReport::RunReport const& f) {
    return print(os,f);
  }

  JobReport::InputFile& JobReport::JobReportImpl::getInputFileForToken(InputType inputType, JobReport::Token t) {

    InputFile* inputFile = nullptr;
    if(inputType == InputType::SecondarySource) {
      if(t >= inputFilesSecSource_.size()) {
        throw edm::Exception(edm::errors::LogicError)
          << "Access reported for secondary source input file with token "
          << t
          << " but no matching input file is found\n";
      }
      inputFile = &inputFilesSecSource_[t];
    } else {
      if(t >= inputFiles_.size()) {
        throw edm::Exception(edm::errors::LogicError)
          << "Access reported for input file with token "
          << t
          << " but no matching input file is found\n";
      }
      inputFile = &inputFiles_[t];
    }
    if(inputFile->fileHasBeenClosed) {
      throw edm::Exception(edm::errors::LogicError)
        << "Access reported for input file with token "
        << t
        << " after this file has been closed.\n"
        << "File record follows:\n"
        << *inputFile
        << '\n';
    }
    return *inputFile;
  }

  JobReport::OutputFile& JobReport::JobReportImpl::getOutputFileForToken(JobReport::Token t) {
    if(t >= outputFiles_.size()) {
      throw edm::Exception(edm::errors::LogicError)
        << "Access reported for output file with token "
        << t
        << " but no matching output file is found\n";
    }
    if(outputFiles_[t].fileHasBeenClosed) {
      throw edm::Exception(edm::errors::LogicError)
        << "Access reported for output file with token "
        << t
        << " after this file has been closed.\n"
        << "File record follows:\n"
        << outputFiles_[t]
        << '\n';
    }
    return outputFiles_[t];
  }

  /*
   * Add the input file token provided to every output
   * file currently available.
   * Used whenever a new input file is opened, it's token
   * is added to all open output files as a contributor
   */
  void JobReport::JobReportImpl::insertInputForOutputs(InputType inputType, JobReport::Token t) {
    for(auto & outputFile : outputFiles_) {
      if(!outputFile.fileHasBeenClosed) {
        if(inputType == InputType::SecondarySource) {
          outputFile.contributingInputsSecSource.push_back(t);
        } else {
          outputFile.contributingInputs.push_back(t);
        }
      }
    }
  }

  /*
   * Write anJobReport::InputFile object to the Logger
   * Generate XML string forJobReport::InputFile instance and dispatch to
   * job report via MessageLogger
   */
  void JobReport::JobReportImpl::writeInputFile(JobReport::InputFile const& f) {
    if(ost_) {
      *ost_ << f ;
      *ost_ << "\n<Runs>";
      for(auto const& runReport : f.runReports) {
        *ost_ << runReport.second;
      }
      *ost_ << "\n</Runs>\n";
      *ost_ << "</InputFile>\n";
      *ost_ << std::flush;
    }
  }

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
  void JobReport::JobReportImpl::writeOutputFile(JobReport::OutputFile const& f) {
    if(ost_) {
      *ost_ << "\n<File>";
      *ost_ << f;

      *ost_ << "\n<Runs>";
      for(auto const& runReport : f.runReports) {
        *ost_ << runReport.second;
      }
      *ost_ << "\n</Runs>\n";

      *ost_ << "\n<Inputs>";
      for(auto token : f.contributingInputs) {
        JobReport::InputFile inpFile = inputFiles_.at(token);
        *ost_ << "\n<Input>";
        *ost_ << "\n  <LFN>" << TiXmlText(inpFile.logicalFileName) << "</LFN>";
        *ost_ << "\n  <PFN>" << TiXmlText(inpFile.physicalFileName) << "</PFN>";
        *ost_ << "\n  <FastCopying>" << findOrDefault(f.fastCopyingInputs, inpFile.physicalFileName) << "</FastCopying>";
        *ost_ << "\n</Input>";
      }
      for(auto token : f.contributingInputsSecSource) {
        JobReport::InputFile inpFile = inputFilesSecSource_.at(token);
        *ost_ << "\n<Input>";
        *ost_ << "\n  <LFN>" << TiXmlText(inpFile.logicalFileName) << "</LFN>";
        *ost_ << "\n  <PFN>" << TiXmlText(inpFile.physicalFileName) << "</PFN>";
        *ost_ << "\n  <FastCopying>" << findOrDefault(f.fastCopyingInputs, inpFile.physicalFileName) << "</FastCopying>";
        *ost_ << "\n</Input>";
      }
      *ost_ << "\n</Inputs>";
      *ost_ << "\n</File>\n";
    }
  }

  /*
   *  Flush all open files to logger in event of a problem.
   *  Called from JobReport dtor to flush any remaining open files
   */
  void JobReport::JobReportImpl::flushFiles(void) {
    for(auto const& inputFile : inputFiles_) {
      if(!(inputFile.fileHasBeenClosed)) {
        writeInputFile(inputFile);
      }
    }
    for(auto const& inputFile : inputFilesSecSource_) {
      if(!(inputFile.fileHasBeenClosed)) {
        writeInputFile(inputFile);
      }
    }
    for(auto const& outputFile : outputFiles_) {
      if(!(outputFile.fileHasBeenClosed)) {
        writeOutputFile(outputFile);
      }
    }
  }

  void JobReport::JobReportImpl::associateRun(JobReport::Token token, unsigned int runNumber) {
    std::map<RunNumber, RunReport>& theMap = outputFiles_.at(token).runReports;
    std::map<RunNumber, RunReport>::iterator iter(theMap.lower_bound(runNumber));
    if(iter == theMap.end() || runNumber < iter->first) {    // not found
      theMap.emplace_hint(iter, runNumber, JobReport::RunReport{ runNumber, {}});  // insert it
    }
  }

  void JobReport::JobReportImpl::associateInputRun(unsigned int runNumber) {
    for(auto& inputFile : inputFiles_) {
      if(!inputFile.fileHasBeenClosed) {
        std::map<RunNumber, RunReport>& theMap = inputFile.runReports;
        std::map<RunNumber, RunReport>::iterator iter(theMap.lower_bound(runNumber));
        if(iter == theMap.end() || runNumber < iter->first) {    // not found
          theMap.emplace_hint(iter, runNumber, JobReport::RunReport{ runNumber, {}});  // insert it
        }
      }
    }
  }

  void JobReport::JobReportImpl::associateLumiSection(JobReport::Token token, unsigned int runNumber, unsigned int lumiSect) {
    std::map<RunNumber, RunReport>& theMap = outputFiles_.at(token).runReports;
    std::map<RunNumber, RunReport>::iterator iter(theMap.lower_bound(runNumber));
    if(iter == theMap.end() || runNumber < iter->first) {    // not found
      theMap.emplace_hint(iter, runNumber, JobReport::RunReport{ runNumber, {lumiSect}});  // insert it
    } else {
      iter->second.lumiSections.insert(lumiSect);
    }
  }

  void JobReport::JobReportImpl::associateInputLumiSection(unsigned int runNumber, unsigned int lumiSect) {
    for(auto& inputFile : inputFiles_) {
      if(!inputFile.fileHasBeenClosed) {
        std::map<RunNumber, RunReport>& theMap = inputFile.runReports;
        std::map<RunNumber, RunReport>::iterator iter(theMap.lower_bound(runNumber));
        if(iter == theMap.end() || runNumber < iter->first) {    // not found
          theMap.emplace_hint(iter, runNumber, JobReport::RunReport{ runNumber, {lumiSect}});  // insert it
        } else {
          iter->second.lumiSections.insert(lumiSect);
        }
      }
    }
  }

  JobReport::~JobReport() {
    impl_->flushFiles();
    if(impl_->ost_) {
      *(impl_->ost_) << "</FrameworkJobReport>\n" << std::flush;
    }
  }

    JobReport::JobReport() :
      impl_(new JobReportImpl(0)) {
    }

    JobReport::JobReport(std::ostream* iOstream) : impl_(new JobReportImpl(iOstream)) {
      if(impl_->ost_) {
        *(impl_->ost_) << "<FrameworkJobReport>\n";
      }
    }

  namespace {
    void
    toFileName(std::string const& jobReportFile, unsigned int childIndex, unsigned int numberOfChildren, std::ostringstream& ofilename) {
      char filler = ofilename.fill();
      unsigned int numberOfDigitsInIndex = 0U;
      while (numberOfChildren != 0) {
        ++numberOfDigitsInIndex;
        numberOfChildren /= 10;
      }
      if(numberOfDigitsInIndex == 0) {
        numberOfDigitsInIndex = 3; // Protect against zero numberOfChildren
      }
      std::string::size_type offset = jobReportFile.rfind('.');
      if(offset == std::string::npos) {
        ofilename << jobReportFile;
        ofilename << '_' << std::setw(numberOfDigitsInIndex) << std::setfill('0') << childIndex << std::setfill(filler);
      } else {
        ofilename << jobReportFile.substr(0, offset);
        ofilename << '_' << std::setw(numberOfDigitsInIndex) << std::setfill('0') << childIndex << std::setfill(filler);
        ofilename << jobReportFile.substr(offset);
      }
    }
  }

  void
  JobReport::parentBeforeFork(std::string const& jobReportFile, unsigned int numberOfChildren) {
    if(impl_->ost_) {
      *(impl_->ost_) << "<ChildProcessFiles>\n";
      for(unsigned int i = 0; i < numberOfChildren; ++i) {
        std::ostringstream ofilename;
        toFileName(jobReportFile, i, numberOfChildren, ofilename);
        *(impl_->ost_) << "  <ChildProcessFile>" << ofilename.str() << "</ChildProcessFile>\n";
      }
      *(impl_->ost_) << "</ChildProcessFiles>\n";
      *(impl_->ost_) << "</FrameworkJobReport>\n";
      std::ofstream* p = dynamic_cast<std::ofstream *>(impl_->ost_);
      if(p) {
        p->close();
      }
    }
  }

  void
  JobReport::parentAfterFork(std::string const& /*jobReportFile*/) {
  }

  void
  JobReport::childAfterFork(std::string const& jobReportFile, unsigned int childIndex, unsigned int numberOfChildren) {
    std::ofstream* p = dynamic_cast<std::ofstream*>(impl_->ost_);
    if(!p) return;
    std::ostringstream ofilename;
    toFileName(jobReportFile, childIndex, numberOfChildren, ofilename);
    p->open(ofilename.str().c_str());
    *p << "<FrameworkJobReport>\n";
  }

  JobReport::Token
  JobReport::inputFileOpened(std::string const& physicalFileName,
                             std::string const& logicalFileName,
                             std::string const& catalog,
                             std::string const& inputType,
                             std::string const& inputSourceClassName,
                             std::string const& moduleLabel,
                             std::string const& guid,
                             std::vector<std::string> const& branchNames) {

    InputType theInputType = InputType::Primary;
    InputFile* newFile = nullptr;
    JobReport::Token newToken = 0;

    if (inputType == "mixingFiles") {
      theInputType = InputType::SecondarySource;
      impl_->inputFilesSecSource_.push_back(InputFile());
      newFile = &impl_->inputFilesSecSource_.back();
      newToken = impl_->inputFilesSecSource_.size() - 1;
    } else {
      if (inputType == "secondaryFiles") {
        theInputType = InputType::SecondaryFile;
      }
      impl_->inputFiles_.emplace_back();
      newFile = &impl_->inputFiles_.back();
      newToken = impl_->inputFiles_.size() - 1;
    }
    
    if(theInputType == InputType::Primary) {
      impl_->lastOpenedPrimaryInputFile_ = impl_->inputFiles_.size() - 1;
    }
    newFile->logicalFileName      = logicalFileName;
    newFile->physicalFileName     = physicalFileName;
    newFile->catalog              = catalog;
    newFile->inputType            = inputType;
    newFile->inputSourceClassName = inputSourceClassName;
    newFile->moduleLabel          = moduleLabel;
    newFile->guid                 = guid;
    newFile->numEventsRead        = 0;
    newFile->branchNames          = branchNames;
    newFile->fileHasBeenClosed    = false;

    // Add the new input file token to all output files
    //  currently open.
    impl_->insertInputForOutputs(theInputType, newToken);
    return newToken;
  }

  void
  JobReport::eventReadFromFile(InputType inputType, JobReport::Token fileToken) {
    JobReport::InputFile& f = impl_->getInputFileForToken(inputType, fileToken);
    ++f.numEventsRead;
  }

  void
  JobReport::reportDataType(Token fileToken, std::string const& dataType) {
    JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
    f.dataType = dataType;
  }

  void
  JobReport::inputFileClosed(InputType inputType, JobReport::Token fileToken) {
    JobReport::InputFile& f = impl_->getInputFileForToken(inputType, fileToken);
    f.fileHasBeenClosed = true;
    if(inputType == InputType::Primary) {
      impl_->writeInputFile(f);
    } else {
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        impl_->writeInputFile(f);
      }
    }
  }

  JobReport::Token
  JobReport::outputFileOpened(std::string const& physicalFileName,
                              std::string const& logicalFileName,
                              std::string const& catalog,
                              std::string const& outputModuleClassName,
                              std::string const& moduleLabel,
                              std::string const& guid,
                              std::string const& dataType,
                              std::string const& branchHash,
                              std::vector<std::string> const& branchNames) {
    impl_->outputFiles_.emplace_back();
    JobReport::OutputFile& r = impl_->outputFiles_.back();

    r.logicalFileName       = logicalFileName;
    r.physicalFileName      = physicalFileName;
    r.catalog               = catalog;
    r.outputModuleClassName = outputModuleClassName;
    r.moduleLabel           = moduleLabel;
    r.guid           = guid;
    r.dataType = dataType;
    r.branchHash = branchHash;
    r.numEventsWritten      = 0;
    r.branchNames           = branchNames;
    r.fileHasBeenClosed     = false;
    //
    // Init list of contributors to list of open input file Tokens
    //
    for(std::vector<Token>::size_type i = 0, iEnd = impl_->inputFiles_.size(); i < iEnd; ++i) {
      if(!impl_->inputFiles_[i].fileHasBeenClosed) {
        r.contributingInputs.push_back(i);
      }
    }
    for(tbb::concurrent_vector<Token>::size_type i = 0, iEnd = impl_->inputFilesSecSource_.size(); i < iEnd; ++i) {
      if(!impl_->inputFilesSecSource_[i].fileHasBeenClosed) {
        r.contributingInputsSecSource.push_back(i);
      }
    }
    return impl_->outputFiles_.size()-1;
  }

  void
  JobReport::eventWrittenToFile(JobReport::Token fileToken, unsigned int /*run*/, unsigned int) {
    JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
    ++f.numEventsWritten;
  }

  void
  JobReport::outputFileClosed(JobReport::Token fileToken) {
    JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
    f.fileHasBeenClosed = true;
    impl_->writeOutputFile(f);
  }

  void
  JobReport::reportSkippedEvent(unsigned int run, unsigned int event) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        msg << "<SkippedEvent Run=\"" << run << "\"";
        msg << " Event=\"" << event << "\" />\n";
        msg << std::flush;
      }
    }
  }

  void
  JobReport:: reportFastCopyingStatus(JobReport::Token fileToken, std::string const& inputFileName, bool fastCopying) {
    JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
    f.fastCopyingInputs.insert(std::make_pair(inputFileName, fastCopying));
  }

  void
  JobReport::reportLumiSection(JobReport::Token token, unsigned int run, unsigned int lumiSectId) {
    impl_->associateLumiSection(token, run, lumiSectId);
  }

  void
  JobReport::reportInputLumiSection(unsigned int run, unsigned int lumiSectId) {
    impl_->associateInputLumiSection(run, lumiSectId);
  }

  void
  JobReport::reportRunNumber(JobReport::Token token, unsigned int run) {
    impl_->associateRun(token, run);
  }

  void
  JobReport::reportInputRunNumber(unsigned int run) {
    impl_->associateInputRun(run);
  }

  void
  JobReport::reportAnalysisFile(std::string const& fileName, std::map<std::string, std::string> const& fileData) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        msg << "<AnalysisFile>\n"
            << "  <FileName>" << TiXmlText(fileName) << "</FileName>\n";

        typedef std::map<std::string, std::string>::const_iterator const_iterator;
        for(const_iterator pos = fileData.begin(), posEnd = fileData.end(); pos != posEnd; ++pos) {
          msg <<  "  <" << pos->first
              <<  "  Value=\"" << pos->second  << "\" />"
              <<  "\n";
        }
        msg << "</AnalysisFile>\n";
        msg << std::flush;
      }
    }
  }

  void
  JobReport::reportError(std::string const& shortDesc,
                         std::string const& longDesc,
                         int const& exitCode) {
    if(impl_->ost_) {
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        std::ostream& msg = *(impl_->ost_);
        msg << "<FrameworkError ExitStatus=\""<< exitCode
            << "\" Type=\"" << shortDesc << "\" >\n";
        msg << "<![CDATA[\n" << longDesc << "\n]]>\n";
        msg << "</FrameworkError>\n";
        msg << std::flush;
      }
    }
  }

  void
  JobReport::reportSkippedFile(std::string const& pfn,
                               std::string const& lfn) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      TiXmlElement skipped("SkippedFile");
      skipped.SetAttribute("Pfn", pfn);
      skipped.SetAttribute("Lfn", lfn);
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        msg << skipped << "\n";
        msg << std::flush;
      }
    }
  }

  void
  JobReport::reportFallbackAttempt(std::string const& pfn, std::string const& lfn, std::string const& err) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      TiXmlElement fallback("FallbackAttempt");
      fallback.SetAttribute("Pfn", pfn);
      fallback.SetAttribute("Lfn", lfn);
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        msg << fallback << "\n";
        msg << "<![CDATA[\n" << err << "\n]]>\n";
        msg << std::flush;
      }
    }
  }

  void
  JobReport::reportMemoryInfo(std::vector<std::string> const& memoryData) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<MemoryService>\n";

      typedef std::vector<std::string>::const_iterator const_iterator;
      for(const_iterator pos = memoryData.begin(), posEnd = memoryData.end(); pos != posEnd; ++pos) {
        msg << *pos << "\n";
      }
      msg << "</MemoryService>\n";
      msg << std::flush;
    }
  }

  void
  JobReport::reportMessageInfo(std::map<std::string, double> const& messageData) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<MessageSummary>\n";
      typedef std::map<std::string, double>::const_iterator const_iterator;
      for(const_iterator pos = messageData.begin(), posEnd = messageData.end(); pos != posEnd; ++pos) {
        msg <<  "  <" << pos->first
        <<  "  Value=\"" << pos->second  << "\" />"
        <<  "\n";
      }
      msg << "</MessageSummary>\n";
      msg << std::flush;
    }
  }

  void
  JobReport::reportReadBranches() {
    if(impl_->printedReadBranches_) return;
    impl_->printedReadBranches_ = true;
    if(impl_->ost_) {
      std::ostream& ost = *(impl_->ost_);
      ost << "<ReadBranches>\n";
      for(auto const& iBranch : impl_->readBranches_) {
        TiXmlElement branch("Branch");
        branch.SetAttribute("Name", iBranch.first);
        branch.SetAttribute("ReadCount", iBranch.second);
        ost << branch << "\n";
      }
      for(auto const& iBranch : impl_->readBranchesSecFile_) {
        TiXmlElement branch("Branch");
        branch.SetAttribute("Name", iBranch.first);
        branch.SetAttribute("ReadCount", iBranch.second);
        ost << branch << "\n";
      }
      ost << "</ReadBranches>\n";
      if(!impl_->readBranchesSecSource_.empty()) {
        ost << "<SecondarySourceReadBranches>\n";
        for(auto const& iBranch : impl_->readBranchesSecSource_) {
          TiXmlElement branch("Branch");
          branch.SetAttribute("Name", iBranch.first);
          branch.SetAttribute("ReadCount", iBranch.second.value().load());
          ost << branch << "\n";
        }
        ost << "</SecondarySourceReadBranches>\n";
      }
      ost << std::flush;
    }
  }

  void
  JobReport::reportReadBranch(InputType inputType, std::string const& branchName) {
    if(inputType == InputType::Primary) {
      // Fast cloned branches have already been reported.
      std::set<std::string> const& clonedBranches = impl_->inputFiles_.at(impl_->lastOpenedPrimaryInputFile_).fastClonedBranches;
      if(clonedBranches.find(branchName) == clonedBranches.end()) {
        ++impl_->readBranches_[branchName];
      }
    } else if (inputType == InputType::SecondaryFile) {
      ++impl_->readBranchesSecFile_[branchName];
    } else if (inputType == InputType::SecondarySource) {
      ++impl_->readBranchesSecSource_[branchName].value();
    }
  }

  void
  JobReport::reportFastClonedBranches(std::set<std::string> const& fastClonedBranches, long long nEvents) {
    std::set<std::string>& clonedBranches = impl_->inputFiles_.at(impl_->lastOpenedPrimaryInputFile_).fastClonedBranches;
    for(std::set<std::string>::const_iterator it = fastClonedBranches.begin(), itEnd = fastClonedBranches.end();
        it != itEnd; ++it) {
      if(clonedBranches.insert(*it).second) {
        impl_->readBranches_[*it] += nEvents;
      }
    }
  }

  void JobReport::reportRandomStateFile(std::string const& name) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        msg << "<RandomServiceStateFile>\n"
            << TiXmlText(name) << "\n"
            <<  "</RandomServiceStateFile>\n";
        msg << std::flush;
      }
    }
  }

  void
  JobReport::reportPerformanceSummary(std::string const& metricClass,
                                      std::map<std::string, std::string> const& metrics) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<PerformanceReport>\n"
        << "  <PerformanceSummary Metric=\"" << metricClass << "\">\n";

      typedef std::map<std::string, std::string>::const_iterator const_iterator;
      for(const_iterator iter = metrics.begin(), iterEnd = metrics.end(); iter != iterEnd; ++iter) {
        msg << "    <Metric Name=\"" << iter->first << "\" "
        << "Value=\"" << iter->second << "\"/>\n";
      }

      msg << "  </PerformanceSummary>\n"
        << "</PerformanceReport>\n";
      msg << std::flush;
      //LogInfo("FwkJob") << msg.str();
    }
  }

  void
  JobReport::reportPerformanceForModule(std::string const&  metricClass,
                                        std::string const&  moduleName,
                                        std::map<std::string, std::string> const& metrics) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<PerformanceReport>\n"
        << "  <PerformanceModule Metric=\"" << metricClass << "\" "
        << " Module=\"" << moduleName << "\" >\n";

      typedef std::map<std::string, std::string>::const_iterator const_iterator;
      for(const_iterator iter = metrics.begin(), iterEnd = metrics.end(); iter != iterEnd; ++iter) {
        msg << "    <Metric Name=\"" << iter->first << "\" "
        << "Value=\"" << iter->second << "\"/>\n";
      }

      msg << "  </PerformanceModule>\n"
        << "</PerformanceReport>\n";
      msg << std::flush;
      //LogInfo("FwkJob") << msg.str();
    }
  }

  std::string
  JobReport::dumpFiles(void) {
    std::ostringstream msg;

    typedef std::vector<JobReport::OutputFile>::iterator iterator;

    for(iterator f = impl_->outputFiles_.begin(), fEnd = impl_->outputFiles_.end(); f != fEnd; ++f) {

      msg << "\n<File>";
      msg << *f;

      msg << "\n<LumiSections>";
      msg << "\n<Inputs>";
      typedef std::vector<JobReport::Token>::iterator iterator;
      for(iterator iInput = f->contributingInputs.begin(),
          iInputEnd = f->contributingInputs.end();
          iInput != iInputEnd; ++iInput) {
        JobReport::InputFile inpFile = impl_->inputFiles_[*iInput];
        msg << "\n<Input>";
        msg << "\n  <LFN>" << TiXmlText(inpFile.logicalFileName) << "</LFN>";
        msg << "\n  <PFN>" << TiXmlText(inpFile.physicalFileName) << "</PFN>";
        msg << "\n  <FastCopying>" << findOrDefault(f->fastCopyingInputs, inpFile.physicalFileName) << "</FastCopying>";
        msg << "\n</Input>";
      }
      msg << "\n</Inputs>";
      msg << "\n</File>";

    }
    return msg.str();
  }

} //namspace edm
