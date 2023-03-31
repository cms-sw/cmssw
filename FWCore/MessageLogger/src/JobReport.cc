
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
#include "tinyxml2.h"
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
    tinyxml2::XMLDocument doc;
    if (f.fileHasBeenClosed) {
      os << "\n<State  Value=\"closed\"/>";
    } else {
      os << "\n<State  Value=\"open\"/>";
    }
    os << "\n<LFN>" << doc.NewText(f.logicalFileName.c_str())->Value() << "</LFN>";
    os << "\n<PFN>" << doc.NewText(f.physicalFileName.c_str())->Value() << "</PFN>";
    os << "\n<Catalog>" << doc.NewText(f.catalog.c_str())->Value() << "</Catalog>";
    os << "\n<ModuleLabel>" << doc.NewText(f.moduleLabel.c_str())->Value() << "</ModuleLabel>";
    os << "\n<GUID>" << f.guid << "</GUID>";
    os << "\n<Branches>";
    for (auto const& branch : f.branchNames) {
      os << "\n  <Branch>" << doc.NewText(branch.c_str())->Value() << "</Branch>";
      doc.DeleteChildren();
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
    tinyxml2::XMLDocument doc;
    os << "\n<InputFile>";
    formatFile(f, os);
    os << "\n<InputType>" << f.inputType << "</InputType>";
    os << "\n<InputSourceClass>" << doc.NewText(f.inputSourceClassName.c_str())->Value() << "</InputSourceClass>";
    os << "\n<EventsRead>" << f.numEventsRead << "</EventsRead>";
    return os;
  }

  template <typename S>
  S& print(S& os, JobReport::OutputFile const& f) {
    tinyxml2::XMLDocument doc;
    formatFile(f, os);
    os << "\n<OutputModuleClass>" << doc.NewText(f.outputModuleClassName.c_str())->Value() << "</OutputModuleClass>";
    os << "\n<TotalEvents>" << f.numEventsWritten << "</TotalEvents>\n";
    os << "\n<DataType>" << doc.NewText(f.dataType.c_str())->Value() << "</DataType>\n";
    os << "\n<BranchHash>" << doc.NewText(f.branchHash.c_str())->Value() << "</BranchHash>\n";
    return os;
  }

  template <typename S>
  S& print(S& os, JobReport::RunReport const& rep) {
    os << "\n<Run ID=\"" << rep.runNumber << "\">\n";

    for (auto const& il : rep.lumiSectionsToNEvents) {
      if (std::numeric_limits<unsigned long>::max() == il.second) {
        os << "   <LumiSection ID=\"" << il.first << "\"/>\n";

      } else {
        os << "   <LumiSection ID=\"" << il.first << "\" NEvents=\"" << il.second << "\"/>\n";
      }
    }
    os << "</Run>\n";
    return os;
  }

  std::ostream& operator<<(std::ostream& os, JobReport::InputFile const& f) { return print(os, f); }
  std::ostream& operator<<(std::ostream& os, JobReport::OutputFile const& f) { return print(os, f); }
  std::ostream& operator<<(std::ostream& os, JobReport::RunReport const& f) { return print(os, f); }

  JobReport::InputFile& JobReport::JobReportImpl::getInputFileForToken(InputType inputType, JobReport::Token t) {
    InputFile* inputFile = nullptr;
    if (inputType == InputType::SecondarySource) {
      if (t >= inputFilesSecSource_.size()) {
        throw edm::Exception(edm::errors::LogicError) << "Access reported for secondary source input file with token "
                                                      << t << " but no matching input file is found\n";
      }
      inputFile = &inputFilesSecSource_[t];
    } else {
      if (t >= inputFiles_.size()) {
        throw edm::Exception(edm::errors::LogicError)
            << "Access reported for input file with token " << t << " but no matching input file is found\n";
      }
      inputFile = &inputFiles_[t];
    }
    if (inputFile->fileHasBeenClosed) {
      throw edm::Exception(edm::errors::LogicError)
          << "Access reported for input file with token " << t << " after this file has been closed.\n"
          << "File record follows:\n"
          << *inputFile << '\n';
    }
    return *inputFile;
  }

  JobReport::OutputFile& JobReport::JobReportImpl::getOutputFileForToken(JobReport::Token t) {
    if (t >= outputFiles_.size()) {
      throw edm::Exception(edm::errors::LogicError)
          << "Access reported for output file with token " << t << " but no matching output file is found\n";
    }
    if (outputFiles_[t].fileHasBeenClosed) {
      throw edm::Exception(edm::errors::LogicError)
          << "Access reported for output file with token " << t << " after this file has been closed.\n"
          << "File record follows:\n"
          << outputFiles_[t] << '\n';
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
    for (auto& outputFile : outputFiles_) {
      if (!outputFile.fileHasBeenClosed) {
        if (inputType == InputType::SecondarySource) {
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
    if (ost_) {
      *ost_ << f;
      *ost_ << "\n<Runs>";
      for (auto const& runReport : f.runReports) {
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
    tinyxml2::XMLDocument doc;
    if (ost_) {
      *ost_ << "\n<File>";
      *ost_ << f;

      *ost_ << "\n<Runs>";
      for (auto const& runReport : f.runReports) {
        *ost_ << runReport.second;
      }
      *ost_ << "\n</Runs>\n";

      *ost_ << "\n<Inputs>";
      for (auto token : f.contributingInputs) {
        JobReport::InputFile inpFile = inputFiles_.at(token);
        *ost_ << "\n<Input>";
        *ost_ << "\n  <LFN>" << doc.NewText(inpFile.logicalFileName.c_str())->Value() << "</LFN>";
        *ost_ << "\n  <PFN>" << doc.NewText(inpFile.physicalFileName.c_str())->Value() << "</PFN>";
        *ost_ << "\n  <FastCopying>" << findOrDefault(f.fastCopyingInputs, inpFile.physicalFileName)
              << "</FastCopying>";
        *ost_ << "\n</Input>";
        doc.DeleteChildren();
      }
      for (auto token : f.contributingInputsSecSource) {
        JobReport::InputFile inpFile = inputFilesSecSource_.at(token);
        *ost_ << "\n<Input>";
        *ost_ << "\n  <LFN>" << doc.NewText(inpFile.logicalFileName.c_str())->Value() << "</LFN>";
        *ost_ << "\n  <PFN>" << doc.NewText(inpFile.physicalFileName.c_str())->Value() << "</PFN>";
        *ost_ << "\n  <FastCopying>" << findOrDefault(f.fastCopyingInputs, inpFile.physicalFileName)
              << "</FastCopying>";
        *ost_ << "\n</Input>";
        doc.DeleteChildren();
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
    for (auto const& inputFile : inputFiles_) {
      if (!(inputFile.fileHasBeenClosed)) {
        writeInputFile(inputFile);
      }
    }
    for (auto const& inputFile : inputFilesSecSource_) {
      if (!(inputFile.fileHasBeenClosed)) {
        writeInputFile(inputFile);
      }
    }
    for (auto const& outputFile : outputFiles_) {
      if (!(outputFile.fileHasBeenClosed)) {
        writeOutputFile(outputFile);
      }
    }
  }

  void JobReport::JobReportImpl::associateRun(JobReport::Token token, unsigned int runNumber) {
    auto& theMap = outputFiles_.at(token).runReports;
    std::map<RunNumber, RunReport>::iterator iter(theMap.lower_bound(runNumber));
    if (iter == theMap.end() || runNumber < iter->first) {                        // not found
      theMap.emplace_hint(iter, runNumber, JobReport::RunReport{runNumber, {}});  // insert it
    }
  }

  void JobReport::JobReportImpl::associateInputRun(unsigned int runNumber) {
    for (auto& inputFile : inputFiles_) {
      if (!inputFile.fileHasBeenClosed) {
        std::map<RunNumber, RunReport>& theMap = inputFile.runReports;
        std::map<RunNumber, RunReport>::iterator iter(theMap.lower_bound(runNumber));
        if (iter == theMap.end() || runNumber < iter->first) {                        // not found
          theMap.emplace_hint(iter, runNumber, JobReport::RunReport{runNumber, {}});  // insert it
        }
      }
    }
  }

  void JobReport::JobReportImpl::associateLumiSection(JobReport::Token token,
                                                      unsigned int runNumber,
                                                      unsigned int lumiSect,
                                                      unsigned long nEvents) {
    auto& theMap = outputFiles_.at(token).runReports;
    std::map<RunNumber, RunReport>::iterator iter(theMap.lower_bound(runNumber));
    if (iter == theMap.end() || runNumber < iter->first) {                                             // not found
      theMap.emplace_hint(iter, runNumber, JobReport::RunReport{runNumber, {{{lumiSect, nEvents}}}});  // insert it
    } else {
      iter->second.lumiSectionsToNEvents[lumiSect] += nEvents;
    }
  }

  void JobReport::JobReportImpl::associateInputLumiSection(unsigned int runNumber, unsigned int lumiSect) {
    for (auto& inputFile : inputFiles_) {
      if (!inputFile.fileHasBeenClosed) {
        std::map<RunNumber, RunReport>& theMap = inputFile.runReports;
        std::map<RunNumber, RunReport>::iterator iter(theMap.lower_bound(runNumber));
        if (iter == theMap.end() || runNumber < iter->first) {  // not found
          theMap.emplace_hint(
              iter,
              runNumber,
              JobReport::RunReport{runNumber, {{lumiSect, std::numeric_limits<unsigned long>::max()}}});  // insert it
        } else {
          iter->second.lumiSectionsToNEvents[lumiSect] = std::numeric_limits<unsigned long>::max();
        }
      }
    }
  }

  static constexpr std::string_view kJobReportEndElement = "</FrameworkJobReport>\n";
  static constexpr int kMinSizeOfComment = 8;

  JobReport::~JobReport() {
    impl_->flushFiles();
    if (impl_->ost_) {
      //are we actually at the end of the file?
      auto pos = impl_->ost_->tellp();
      impl_->ost_->seekp(0, std::ios_base::end);
      auto endpos = impl_->ost_->tellp();
      impl_->ost_->seekp(pos);
      if ((endpos - pos) > static_cast<long int>(kJobReportEndElement.size())) {
        //need to add some padding so use a comment element
        // comment is used since white spaces are converted to a special node
        // while comments are usually ignored by xml parsers
        auto padding = (endpos - pos) - (kJobReportEndElement.size() + kMinSizeOfComment);
        *(impl_->ost_) << "<!--";
        for (int i = padding; i > 0; --i) {
          (*impl_->ost_) << ' ';
        }
        *(impl_->ost_) << "-->\n";
      }
      *(impl_->ost_) << kJobReportEndElement << std::flush;
    }
  }

  void JobReport::temporarilyCloseXML() {
    if (impl_->ost_) {
      //remember where we were
      auto pos = impl_->ost_->tellp();
      if (not errorLogged_) {
        *(impl_->ost_) << "<FrameworkError ExitStatus=\"8901\" Type=\"UnexpectedJobTermination\"/>\n";
      }
      *(impl_->ost_) << kJobReportEndElement << std::flush;

      //overwrite above during next write.
      impl_->ost_->seekp(pos);
    }
  }

  JobReport::JobReport() : impl_(new JobReportImpl(nullptr)) {}

  JobReport::JobReport(std::ostream* iOstream) : impl_(new JobReportImpl(iOstream)) {
    if (impl_->ost_) {
      *(impl_->ost_) << "<FrameworkJobReport>\n";
    }
    temporarilyCloseXML();
  }

  JobReport::Token JobReport::inputFileOpened(std::string const& physicalFileName,
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
      auto itr = impl_->inputFilesSecSource_.push_back(InputFile());
      newFile = &(*itr);
      newToken = itr - impl_->inputFilesSecSource_.begin();
    } else {
      if (inputType == "secondaryFiles") {
        theInputType = InputType::SecondaryFile;
      }
      impl_->inputFiles_.emplace_back();
      newFile = &impl_->inputFiles_.back();
      newToken = impl_->inputFiles_.size() - 1;
    }

    if (theInputType == InputType::Primary) {
      impl_->lastOpenedPrimaryInputFile_ = impl_->inputFiles_.size() - 1;
    }
    newFile->logicalFileName = logicalFileName;
    newFile->physicalFileName = physicalFileName;
    newFile->catalog = catalog;
    newFile->inputType = inputType;
    newFile->inputSourceClassName = inputSourceClassName;
    newFile->moduleLabel = moduleLabel;
    newFile->guid = guid;
    newFile->numEventsRead = 0;
    newFile->branchNames = branchNames;
    newFile->fileHasBeenClosed = false;

    // Add the new input file token to all output files
    //  currently open.
    impl_->insertInputForOutputs(theInputType, newToken);
    return newToken;
  }

  void JobReport::eventReadFromFile(InputType inputType, JobReport::Token fileToken) {
    JobReport::InputFile& f = impl_->getInputFileForToken(inputType, fileToken);
    ++f.numEventsRead;
  }

  void JobReport::reportDataType(Token fileToken, std::string const& dataType) {
    JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
    f.dataType = dataType;
  }

  void JobReport::inputFileClosed(InputType inputType, JobReport::Token fileToken) {
    JobReport::InputFile& f = impl_->getInputFileForToken(inputType, fileToken);
    f.fileHasBeenClosed = true;
    std::lock_guard<std::mutex> lock(write_mutex);
    if (inputType == InputType::Primary) {
      impl_->writeInputFile(f);
    } else {
      impl_->writeInputFile(f);
    }
    temporarilyCloseXML();
  }

  JobReport::Token JobReport::outputFileOpened(std::string const& physicalFileName,
                                               std::string const& logicalFileName,
                                               std::string const& catalog,
                                               std::string const& outputModuleClassName,
                                               std::string const& moduleLabel,
                                               std::string const& guid,
                                               std::string const& dataType,
                                               std::string const& branchHash,
                                               std::vector<std::string> const& branchNames) {
    auto itr = impl_->outputFiles_.emplace_back();
    JobReport::OutputFile& r = *itr;

    r.logicalFileName = logicalFileName;
    r.physicalFileName = physicalFileName;
    r.catalog = catalog;
    r.outputModuleClassName = outputModuleClassName;
    r.moduleLabel = moduleLabel;
    r.guid = guid;
    r.dataType = dataType;
    r.branchHash = branchHash;
    r.numEventsWritten = 0;
    r.branchNames = branchNames;
    r.fileHasBeenClosed = false;
    //
    // Init list of contributors to list of open input file Tokens
    //
    for (std::vector<Token>::size_type i = 0, iEnd = impl_->inputFiles_.size(); i < iEnd; ++i) {
      if (!impl_->inputFiles_[i].fileHasBeenClosed) {
        r.contributingInputs.push_back(i);
      }
    }
    for (oneapi::tbb::concurrent_vector<Token>::size_type i = 0, iEnd = impl_->inputFilesSecSource_.size(); i < iEnd;
         ++i) {
      if (!impl_->inputFilesSecSource_[i].fileHasBeenClosed) {
        r.contributingInputsSecSource.push_back(i);
      }
    }
    return itr - impl_->outputFiles_.begin();
  }

  void JobReport::eventWrittenToFile(JobReport::Token fileToken, RunNumber_t /*run*/, EventNumber_t) {
    JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
    ++f.numEventsWritten;
  }

  void JobReport::outputFileClosed(JobReport::Token fileToken) {
    JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
    f.fileHasBeenClosed = true;
    std::lock_guard<std::mutex> lock(write_mutex);
    impl_->writeOutputFile(f);
    temporarilyCloseXML();
  }

  void JobReport::reportSkippedEvent(RunNumber_t run, EventNumber_t event) {
    if (impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        msg << "<SkippedEvent Run=\"" << run << "\"";
        msg << " Event=\"" << event << "\" />\n";
        temporarilyCloseXML();
      }
    }
  }

  void JobReport::reportFastCopyingStatus(JobReport::Token fileToken,
                                          std::string const& inputFileName,
                                          bool fastCopying) {
    JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
    f.fastCopyingInputs.insert(std::make_pair(inputFileName, fastCopying));
  }

  void JobReport::reportLumiSection(JobReport::Token token,
                                    unsigned int run,
                                    unsigned int lumiSectId,
                                    unsigned long nEvents) {
    impl_->associateLumiSection(token, run, lumiSectId, nEvents);
  }

  void JobReport::reportInputLumiSection(unsigned int run, unsigned int lumiSectId) {
    impl_->associateInputLumiSection(run, lumiSectId);
  }

  void JobReport::reportRunNumber(JobReport::Token token, unsigned int run) { impl_->associateRun(token, run); }

  void JobReport::reportInputRunNumber(unsigned int run) { impl_->associateInputRun(run); }

  void JobReport::reportAnalysisFile(std::string const& fileName, std::map<std::string, std::string> const& fileData) {
    tinyxml2::XMLDocument doc;
    if (impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        msg << "<AnalysisFile>\n"
            << "  <FileName>" << doc.NewText(fileName.c_str())->Value() << "</FileName>\n";

        typedef std::map<std::string, std::string>::const_iterator const_iterator;
        for (const_iterator pos = fileData.begin(), posEnd = fileData.end(); pos != posEnd; ++pos) {
          msg << "  <" << pos->first << "  Value=\"" << pos->second << "\" />"
              << "\n";
        }
        msg << "</AnalysisFile>\n";
        temporarilyCloseXML();
      }
    }
  }

  void JobReport::reportError(std::string const& shortDesc, std::string const& longDesc, int const& exitCode) {
    if (impl_->ost_) {
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        errorLogged_ = true;
        std::ostream& msg = *(impl_->ost_);
        msg << "<FrameworkError ExitStatus=\"" << exitCode << "\" Type=\"" << shortDesc << "\" >\n";
        msg << "<![CDATA[\n" << longDesc << "\n]]>\n";
        msg << "</FrameworkError>\n";
        temporarilyCloseXML();
      }
    }
  }

  void JobReport::reportShutdownSignal() {
    if (impl_->ost_) {
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        std::ostream& msg = *(impl_->ost_);
        msg << "<ShutdownSignal/>\n";
        temporarilyCloseXML();
      }
    }
  }

  void JobReport::reportSkippedFile(std::string const& pfn, std::string const& lfn) {
    if (impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      tinyxml2::XMLDocument doc;
      tinyxml2::XMLPrinter printer;
      tinyxml2::XMLElement* skipped = doc.NewElement("SkippedFile");
      skipped->SetAttribute("Pfn", pfn.c_str());
      skipped->SetAttribute("Lfn", lfn.c_str());
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        skipped->Accept(&printer);
        msg << printer.CStr();
        temporarilyCloseXML();
      }
    }
  }

  void JobReport::reportFallbackAttempt(std::string const& pfn, std::string const& lfn, std::string const& err) {
    if (impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      tinyxml2::XMLDocument doc;
      tinyxml2::XMLPrinter printer;
      tinyxml2::XMLElement* fallback = doc.NewElement("FallbackAttempt");
      fallback->SetAttribute("Pfn", pfn.c_str());
      fallback->SetAttribute("Lfn", lfn.c_str());
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        fallback->Accept(&printer);
        msg << printer.CStr();
        msg << "<![CDATA[\n" << err << "\n]]>\n";
        temporarilyCloseXML();
      }
    }
  }

  void JobReport::reportMemoryInfo(std::vector<std::string> const& memoryData) {
    if (impl_->ost_) {
      std::lock_guard<std::mutex> lock(write_mutex);
      std::ostream& msg = *(impl_->ost_);
      msg << "<MemoryService>\n";

      typedef std::vector<std::string>::const_iterator const_iterator;
      for (const_iterator pos = memoryData.begin(), posEnd = memoryData.end(); pos != posEnd; ++pos) {
        msg << *pos << "\n";
      }
      msg << "</MemoryService>\n";
      temporarilyCloseXML();
    }
  }

  void JobReport::reportMessageInfo(std::map<std::string, double> const& messageData) {
    if (impl_->ost_) {
      std::lock_guard<std::mutex> lock(write_mutex);
      std::ostream& msg = *(impl_->ost_);
      msg << "<MessageSummary>\n";
      typedef std::map<std::string, double>::const_iterator const_iterator;
      for (const_iterator pos = messageData.begin(), posEnd = messageData.end(); pos != posEnd; ++pos) {
        msg << "  <" << pos->first << "  Value=\"" << pos->second << "\" />"
            << "\n";
      }
      msg << "</MessageSummary>\n";
      temporarilyCloseXML();
    }
  }

  void JobReport::reportReadBranches() {
    bool expected = false;
    if (not impl_->printedReadBranches_.compare_exchange_strong(expected, true))
      return;
    if (impl_->ost_) {
      std::lock_guard<std::mutex> lock(write_mutex);
      std::ostream& ost = *(impl_->ost_);
      ost << "<ReadBranches>\n";
      tinyxml2::XMLDocument doc;
      tinyxml2::XMLPrinter printer;
      for (auto const& iBranch : impl_->readBranches_) {
        tinyxml2::XMLElement* branch = doc.NewElement("Branch");
        branch->SetAttribute("Name", iBranch.first.c_str());
        branch->SetAttribute("ReadCount", int64_t(iBranch.second));
        branch->Accept(&printer);
        ost << printer.CStr();
        printer.ClearBuffer();
      }
      for (auto const& iBranch : impl_->readBranchesSecFile_) {
        tinyxml2::XMLElement* branch = doc.NewElement("Branch");
        branch->SetAttribute("Name", iBranch.first.c_str());
        branch->SetAttribute("ReadCount", int64_t(iBranch.second));
        branch->Accept(&printer);
        ost << printer.CStr();
        printer.ClearBuffer();
      }
      ost << "</ReadBranches>\n";
      if (!impl_->readBranchesSecSource_.empty()) {
        ost << "<SecondarySourceReadBranches>\n";
        for (auto const& iBranch : impl_->readBranchesSecSource_) {
          tinyxml2::XMLElement* branch = doc.NewElement("Branch");
          branch->SetAttribute("Name", iBranch.first.c_str());
          branch->SetAttribute("ReadCount", int64_t(iBranch.second.value().load()));
          branch->Accept(&printer);
          ost << printer.CStr();
          printer.ClearBuffer();
        }
        ost << "</SecondarySourceReadBranches>\n";
      }
      temporarilyCloseXML();
    }
  }

  void JobReport::reportReadBranch(InputType inputType, std::string const& branchName) {
    if (inputType == InputType::Primary) {
      // Fast cloned branches have already been reported.
      std::set<std::string> const& clonedBranches =
          impl_->inputFiles_.at(impl_->lastOpenedPrimaryInputFile_).fastClonedBranches;
      if (clonedBranches.find(branchName) == clonedBranches.end()) {
        ++impl_->readBranches_[branchName];
      }
    } else if (inputType == InputType::SecondaryFile) {
      ++impl_->readBranchesSecFile_[branchName];
    } else if (inputType == InputType::SecondarySource) {
      ++impl_->readBranchesSecSource_[branchName].value();
    }
  }

  void JobReport::reportFastClonedBranches(std::set<std::string> const& fastClonedBranches, long long nEvents) {
    std::set<std::string>& clonedBranches =
        impl_->inputFiles_.at(impl_->lastOpenedPrimaryInputFile_).fastClonedBranches;
    for (std::set<std::string>::const_iterator it = fastClonedBranches.begin(), itEnd = fastClonedBranches.end();
         it != itEnd;
         ++it) {
      if (clonedBranches.insert(*it).second) {
        impl_->readBranches_[*it] += nEvents;
      }
    }
  }

  void JobReport::reportRandomStateFile(std::string const& name) {
    tinyxml2::XMLDocument doc;
    if (impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      {
        std::lock_guard<std::mutex> lock(write_mutex);
        msg << "<RandomServiceStateFile>\n"
            << doc.NewText(name.c_str())->Value() << "\n"
            << "</RandomServiceStateFile>\n";
        temporarilyCloseXML();
      }
    }
  }

  void JobReport::reportPerformanceSummary(std::string const& metricClass,
                                           std::map<std::string, std::string> const& metrics) {
    if (impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<PerformanceReport>\n"
          << "  <PerformanceSummary Metric=\"" << metricClass << "\">\n";

      typedef std::map<std::string, std::string>::const_iterator const_iterator;
      for (const_iterator iter = metrics.begin(), iterEnd = metrics.end(); iter != iterEnd; ++iter) {
        msg << "    <Metric Name=\"" << iter->first << "\" "
            << "Value=\"" << iter->second << "\"/>\n";
      }

      msg << "  </PerformanceSummary>\n"
          << "</PerformanceReport>\n";
      temporarilyCloseXML();
    }
  }

  void JobReport::reportPerformanceForModule(std::string const& metricClass,
                                             std::string const& moduleName,
                                             std::map<std::string, std::string> const& metrics) {
    if (impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<PerformanceReport>\n"
          << "  <PerformanceModule Metric=\"" << metricClass << "\" "
          << " Module=\"" << moduleName << "\" >\n";

      typedef std::map<std::string, std::string>::const_iterator const_iterator;
      for (const_iterator iter = metrics.begin(), iterEnd = metrics.end(); iter != iterEnd; ++iter) {
        msg << "    <Metric Name=\"" << iter->first << "\" "
            << "Value=\"" << iter->second << "\"/>\n";
      }

      msg << "  </PerformanceModule>\n"
          << "</PerformanceReport>\n";
      temporarilyCloseXML();
    }
  }

  std::string JobReport::dumpFiles(void) {
    std::ostringstream msg;

    tinyxml2::XMLDocument doc;
    for (auto const& f : impl_->outputFiles_) {
      msg << "\n<File>";
      msg << f;

      msg << "\n<LumiSections>";
      msg << "\n<Inputs>";
      typedef std::vector<JobReport::Token>::iterator iterator;
      for (auto const& iInput : f.contributingInputs) {
        auto const& inpFile = impl_->inputFiles_[iInput];
        msg << "\n<Input>";
        msg << "\n  <LFN>" << doc.NewText(inpFile.logicalFileName.c_str())->Value() << "</LFN>";
        msg << "\n  <PFN>" << doc.NewText(inpFile.physicalFileName.c_str())->Value() << "</PFN>";
        msg << "\n  <FastCopying>" << findOrDefault(f.fastCopyingInputs, inpFile.physicalFileName) << "</FastCopying>";
        msg << "\n</Input>";
        doc.DeleteChildren();
      }
      msg << "\n</Inputs>";
      msg << "\n</File>";
    }
    return msg.str();
  }

}  // namespace edm
