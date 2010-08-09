
// -*- C++ -*-
//
//
// 10/23/07 mf	In an attempt to get clues about (or ene) the
//		does-not-output-branches behavior, changed the
//		generic os<< lines in JobReport::JobReportImpl::writeOutputFile
//		to direct use of LogInfo.
//
// 4/8/08   mf	Encase the logdesc for in <CDATA> ... </CDATA>
//
// 6/19/08  mf	reportMessageInfo()
//
// 24 June 2008   ewv  Correct format for CDATA and for second instance of reportError

//
// Original Author:  Marc Paterno
//

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Map.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace edm {
    /*
     * Note that output formatting is spattered across these classes
     * If something outside these classes requires access to the
     * same formatting then we need to refactor it into a common library
     */
  template <typename S>
    S&
    print(S& os, JobReport::InputFile const& f) {

      os << "\n<InputFile>";
      formatFile(f, os);
      os << "\n<InputType>" << f.inputType << "</InputType>";
      os << "\n<InputSourceClass>" << TiXmlText(f.inputSourceClassName)
	 << "</InputSourceClass>";
      os << "\n<EventsRead>" << f.numEventsRead << "</EventsRead>";
      return os;
    }

  template <typename S>
    S&
    print(S& os, JobReport::OutputFile const& f) {
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
    S&
    print(S& os,
	   JobReport::RunReport const& rep) {
    os << "\n<Run ID=\""
       << rep.runNumber
       << "\">\n";

    typedef std::set<unsigned int>::iterator iterator;
    for (iterator il = rep.lumiSections.begin(), ilEnd = rep.lumiSections.end(); il != ilEnd; ++il) {
      os << "   <LumiSection ID=\"" << *il << "\"/>\n";

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

  //To talk to MessageLogger directly
  edm::MessageSender& operator<< (edm::MessageSender& os, JobReport::InputFile const& f) {
    return print(os,f);
  }
  edm::MessageSender& operator<< (edm::MessageSender& os, JobReport::OutputFile const& f) {
    return print(os,f);
  }
  edm::MessageSender& operator<< (edm::MessageSender& os, JobReport::RunReport const& f) {
    return print(os,f);
  }
//  edm::MessageSender& operator<< (edm::MessageSender& os, JobReport::LumiSectionReport const& rep) {
//    return print(os,rep);
//  }
//

    JobReport::InputFile& JobReport::JobReportImpl::getInputFileForToken(JobReport::Token t) {
	if (t >= inputFiles_.size()) {
	    throw edm::Exception(edm::errors::LogicError)
	      << "Access reported for input file with token "
	      << t
	      << " but no matching input file is found\n";
	}

	if (inputFiles_[t].fileHasBeenClosed) {
	    throw edm::Exception(edm::errors::LogicError)
	      << "Access reported for input file with token "
	      << t
	      << " after this file has been closed.\n"
	      << "File record follows:\n"
	      << inputFiles_[t]
	      << '\n';
	}

      return inputFiles_[t];
    }

    JobReport::OutputFile& JobReport::JobReportImpl::getOutputFileForToken(JobReport::Token t) {
	if (t >= outputFiles_.size()) {
	    throw edm::Exception(edm::errors::LogicError)
	      << "Access reported for output file with token "
	      << t
	      << " but no matching output file is found\n";
	}
	if (outputFiles_[t].fileHasBeenClosed) {
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
    void JobReport::JobReportImpl::insertInputForOutputs(JobReport::Token t) {
	typedef std::vector<JobReport::OutputFile>::iterator iterator;
	for (iterator outFile = outputFiles_.begin(), outFileEnd = outputFiles_.end(); outFile != outFileEnd; ++outFile) {
	  outFile->contributingInputs.push_back(t);
	}
    }
    /*
     * get a vector of Tokens for all currently open
     * input files.
     * Used when a new output file is opened, all currently open
     * input file tokens are used to initialize its list of contributors
     */
    void JobReport::JobReportImpl::openInputFiles(std::vector<JobReport::Token>& result) {
	result.reserve(inputFiles_.size());
	for (unsigned int i = 0; i < inputFiles_.size(); ++i) {
	  if (inputFiles_[i].fileHasBeenClosed == false) {
	    result.push_back(i);
	  }
	}
    }

    /*
     * get a vector of Tokens for all currently open
     * output files.
     *
     */
    void JobReport::JobReportImpl::openOutputFiles(std::vector<JobReport::Token>& result) {
	result.reserve(outputFiles_.size());
	for (unsigned int i = 0; i < outputFiles_.size(); ++i) {
	  if (outputFiles_[i].fileHasBeenClosed == false) {
	    result.push_back(i);
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
	typedef std::map<JobReport::RunNumber, JobReport::RunReport>::const_iterator const_iterator;
	for (const_iterator iRun = f.runReports.begin(), iRunEnd = f.runReports.end(); iRun != iRunEnd; ++iRun) {
	  *ost_ << iRun->second;
	}
	*ost_ << "\n</Runs>\n";
        *ost_ << "</InputFile>\n";
	*ost_ << std::flush;
      }
	//LogInfo("FwkJob") << f;
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
      if (ost_) {
	*ost_ << "\n<File>";
	*ost_ << f;

	*ost_ << "\n<Runs>";
	typedef std::map<JobReport::RunNumber, JobReport::RunReport>::const_iterator const_iterator;
	for (const_iterator iRun = f.runReports.begin(), iRunEnd = f.runReports.end(); iRun != iRunEnd; ++iRun) {
	  *ost_ << iRun->second;
	}
	*ost_ << "\n</Runs>\n";

	*ost_ << "\n<Inputs>";
	for (std::vector<JobReport::Token>::const_iterator
          iInput = f.contributingInputs.begin(),
          iInputEnd = f.contributingInputs.end();
          iInput != iInputEnd; ++iInput) {
	    JobReport::InputFile inpFile = inputFiles_[*iInput];
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
      for (std::vector<JobReport::InputFile>::iterator ipos = inputFiles_.begin(), iposEnd = inputFiles_.end();
          ipos != iposEnd; ++ipos) {
        if (!(ipos->fileHasBeenClosed)) {
          writeInputFile(*ipos);
        }
      }
      for (std::vector<JobReport::OutputFile>::iterator opos = outputFiles_.begin(), oposEnd = outputFiles_.end();
          opos != oposEnd; ++opos) {
	if (!(opos->fileHasBeenClosed)) {
	  writeOutputFile(*opos);
	}
      }
    }

  void JobReport::JobReportImpl::addGeneratorInfo(std::string const& name,
						  std::string const& value) {

    generatorInfo_[name] = value;
  }

  void JobReport::JobReportImpl::writeGeneratorInfo(void) {
    if(ost_) {
      *ost_ << "\n<GeneratorInfo>\n";
      for (std::map<std::string, std::string>::iterator pos = generatorInfo_.begin(),
          posEnd = generatorInfo_.end();
          pos != posEnd;  ++pos) {
        std::ostringstream msg;
        msg << "\n<Data Name=\"" << pos->first
          << "\" Value=\"" << pos->second << "\"/>";
        *ost_ << msg.str();
      }
      *ost_ << "</GeneratorInfo>\n";
    }
  }

  void JobReport::JobReportImpl::associateRun(unsigned int runNumber) {
    std::vector<Token> openFiles;
    openOutputFiles(openFiles);
    for (std::vector<Token>::iterator iToken = openFiles.begin(), iTokenEnd = openFiles.end(); iToken != iTokenEnd; ++iToken) {
      JobReport::OutputFile & theFile = outputFiles_[*iToken];

      //
      // check run is known to file
      // if not, add a run report for that run
      if (theFile.runReports.count(runNumber) == 0) {
	JobReport::RunReport newReport = JobReport::RunReport();
	newReport.runNumber = runNumber;
	theFile.runReports.insert(
		 std::make_pair(runNumber, newReport)
		);
      }

    }
  }

  void JobReport::JobReportImpl::associateInputRun(unsigned int runNumber) {
    std::vector<Token> openFiles;
    openInputFiles(openFiles);
    for (std::vector<Token>::iterator iToken = openFiles.begin(), iTokenEnd = openFiles.end(); iToken != iTokenEnd; ++iToken) {
      JobReport::InputFile & theFile = inputFiles_[*iToken];

      //
      // check run is known to file
      // if not, add a run report for that run
      if (theFile.runReports.count(runNumber) == 0) {
	JobReport::RunReport newReport = JobReport::RunReport();
	newReport.runNumber = runNumber;
	theFile.runReports.insert(std::make_pair(runNumber, newReport));
      }

    }
  }

  void JobReport::JobReportImpl::associateLumiSection(unsigned int runNumber, unsigned int lumiSect) {
    std::vector<Token> openFiles;
    openOutputFiles(openFiles);
    for (std::vector<Token>::iterator iToken = openFiles.begin(), iTokenEnd = openFiles.end(); iToken != iTokenEnd; ++iToken) {
      //
      // Loop over all open output files
      //
      JobReport::OutputFile & theFile = outputFiles_[*iToken];

      //
      // check run is known to file
      // if not, add a run report for that run
      if (theFile.runReports.count(runNumber) == 0) {
	JobReport::RunReport newReport = JobReport::RunReport();
	newReport.runNumber = runNumber;
	theFile.runReports.insert(std::make_pair(runNumber, newReport));
      }

      //
      // Get the run report for this run, now that it either was created
      // or already existed
      std::map<JobReport::RunNumber, JobReport::RunReport>::iterator finder;
      finder = theFile.runReports.find(runNumber);

      //
      // add the lumi section to the report, the lumi list is a Set
      // so duplicates dont matter
      (finder->second).lumiSections.insert(lumiSect);
    }
  }

  void JobReport::JobReportImpl::associateInputLumiSection(unsigned int runNumber, unsigned int lumiSect) {
    std::vector<Token> openFiles;
    openInputFiles(openFiles);
    for (std::vector<Token>::iterator iToken = openFiles.begin(), iTokenEnd = openFiles.end(); iToken != iTokenEnd; ++iToken) {
      //
      // Loop over all open input files
      //
      JobReport::InputFile & theFile = inputFiles_[*iToken];

      //
      // check run is known to file
      // if not, add a run report for that run
      if (theFile.runReports.count(runNumber) == 0) {
	JobReport::RunReport newReport = JobReport::RunReport();
	newReport.runNumber = runNumber;
	theFile.runReports.insert(std::make_pair(runNumber, newReport));
      }

      //
      // Get the run report for this run, now that it either was created
      // or already existed
      std::map<JobReport::RunNumber, JobReport::RunReport>::iterator finder;
      finder = theFile.runReports.find(runNumber);

      //
      // add the lumi section to the report, the lumi list is a Set
      // so duplicates dont matter
      (finder->second).lumiSections.insert(lumiSect);
    }
  }

  JobReport::~JobReport() {
    impl_->writeGeneratorInfo();
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
      unsigned int numberOfDigitsInIndex = 0U;
      while (numberOfChildren != 0) {
        ++numberOfDigitsInIndex;
        numberOfChildren /= 10;
      }
      if (numberOfDigitsInIndex == 0) {
        numberOfDigitsInIndex = 3; // Protect against zero numberOfChildren
      }
      std::string::size_type offset = jobReportFile.rfind('.');
      if (offset == std::string::npos) {
        ofilename << jobReportFile;
        ofilename << '_' << std::setw(numberOfDigitsInIndex) << std::setfill('0') << childIndex;
      } else {
        ofilename << jobReportFile.substr(0, offset);
        ofilename << '_' << std::setw(numberOfDigitsInIndex) << std::setfill('0') << childIndex;
        ofilename << jobReportFile.substr(offset);
      }
    }
  }

    void
    JobReport::parentBeforeFork(std::string const& jobReportFile, unsigned int numberOfChildren) {
      if(impl_->ost_) {
        *(impl_->ost_) << "<ChildProcessFiles>\n";
        for (unsigned int i = 0; i < numberOfChildren; ++i) {
	  std::ostringstream ofilename;
          toFileName(jobReportFile, i, numberOfChildren, ofilename);
          *(impl_->ost_) << "  <ChildProcessFile>" << ofilename.str() << "</ChildProcessFile>\n";
        }
        *(impl_->ost_) << "</ChildProcessFiles>\n";
        *(impl_->ost_) << "</FrameworkJobReport>\n";
        std::ofstream* p = dynamic_cast<std::ofstream *>(impl_->ost_);
        if (p) {
          p->close();
        }
      }
    }

    void
    JobReport::parentAfterFork(std::string const& jobReportFile) {
    }

    void
    JobReport::childAfterFork(std::string const& jobReportFile, unsigned int childIndex, unsigned int numberOfChildren) {
      std::ofstream* p = dynamic_cast<std::ofstream *>(impl_->ost_);
      if (!p) return;
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
      // Do we have to worry about thread safety here? Or is this
      // service used in a way to make this safe?
      impl_->inputFiles_.push_back(JobReport::InputFile());
      JobReport::InputFile& r = impl_->inputFiles_.back();

      r.logicalFileName      = logicalFileName;
      r.physicalFileName     = physicalFileName;
      r.catalog              = catalog;
      r.inputType            = inputType;
      r.inputSourceClassName = inputSourceClassName;
      r.moduleLabel          = moduleLabel;
      r.guid                 = guid;
      // r.runsSeen is not modified
      r.numEventsRead        = 0;
      r.branchNames          = branchNames;
      r.fileHasBeenClosed    = false;

      JobReport::Token newToken = impl_->inputFiles_.size()-1;
        //
       // Add the new input file token to all output files
      //  currently open.
      impl_->insertInputForOutputs(newToken);
      return newToken;
    }

    void
    JobReport::eventReadFromFile(JobReport::Token fileToken, unsigned int run, unsigned int) {
      JobReport::InputFile& f = impl_->getInputFileForToken(fileToken);
      ++f.numEventsRead;
      //f.runsSeen.insert(run);
    }

    void
    JobReport::reportDataType(Token fileToken, std::string const& dataType) {
      JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
      f.dataType = dataType;
    }

    void
    JobReport::inputFileClosed(JobReport::Token fileToken) {
      JobReport::InputFile& f = impl_->getInputFileForToken(fileToken);
      // Dump information to the MessageLogger's JobSummary
      // about this file.
      // After setting the file to 'closed', we will no longer be able
      // to reference it by ID.
      f.fileHasBeenClosed = true;
      impl_->writeInputFile(f);
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
      impl_->outputFiles_.push_back(JobReport::OutputFile());
      JobReport::OutputFile& r = impl_->outputFiles_.back();

      r.logicalFileName       = logicalFileName;
      r.physicalFileName      = physicalFileName;
      r.catalog               = catalog;
      r.outputModuleClassName = outputModuleClassName;
      r.moduleLabel           = moduleLabel;
      r.guid           = guid;
      r.dataType = dataType;
      r.branchHash = branchHash;
      // r.runsSeen is not modified
      r.numEventsWritten      = 0;
      r.branchNames           = branchNames;
      r.fileHasBeenClosed     = false;
        //
       // Init list of contributors to list of open input file Tokens
      //
      impl_->openInputFiles(r.contributingInputs);
      return impl_->outputFiles_.size()-1;
    }

    void
    JobReport::eventWrittenToFile(JobReport::Token fileToken, unsigned int run, unsigned int) {
      JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
      ++f.numEventsWritten;
      //f.runsSeen.insert(run);
    }

    void
    JobReport::outputFileClosed(JobReport::Token fileToken) {
      JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
      // Dump information to the MessageLogger's JobSummary
      // about this file.
      // After setting the file to 'closed', we will no longer be able
      // to reference it by ID.
      f.fileHasBeenClosed = true;
      impl_->writeOutputFile(f);
    }

    void
    JobReport:: reportFastCopyingStatus(JobReport::Token fileToken, std::string const& inputFileName, bool fastCopying) {
      JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
      f.fastCopyingInputs.insert(std::make_pair(inputFileName, fastCopying));
    }

    void
    JobReport::overrideEventsWritten(Token fileToken, int const eventsWritten) {
      // Get the required output file instance using the token
      JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
      // set the eventsWritten parameter to the provided value
      f.numEventsWritten = eventsWritten;
    }

    void
    JobReport::overrideEventsRead(Token fileToken, int const eventsRead) {
      // Get the required input file instance using the token
      JobReport::InputFile& f = impl_->getInputFileForToken(fileToken);
      // set the events read parameter to the provided value
      f.numEventsRead = eventsRead;
    }

    void
    JobReport::overrideContributingInputs(Token outputToken,
					  std::vector<Token> const& inputTokens) {
       // Get the required output file instance using the token
      JobReport::OutputFile& f = impl_->getOutputFileForToken(outputToken);
      // override its contributing inputs data
      f.contributingInputs = inputTokens;
    }

    void
    JobReport::reportSkippedEvent(unsigned int run, unsigned int event) {
      if(impl_->ost_) {
        std::ostream& msg = *(impl_->ost_);
        msg << "<SkippedEvent Run=\"" << run << "\"";
        msg << " Event=\"" << event << "\" />\n";
        msg << std::flush;
        //LogInfo("FwkJob") << msg.str();
      }
    }

  void
  JobReport::reportLumiSection(unsigned int run, unsigned int lumiSectId) {
    impl_->associateLumiSection(run, lumiSectId);
  }

  void
  JobReport::reportInputLumiSection(unsigned int run, unsigned int lumiSectId) {
    impl_->associateInputLumiSection(run, lumiSectId);
  }

  void
  JobReport::reportRunNumber(unsigned int run) {
    impl_->associateRun(run);
  }
  void
  JobReport::reportInputRunNumber(unsigned int run) {
    impl_->associateInputRun(run);
  }

  void
  JobReport::reportError(std::string const& shortDesc,
  			 std::string const& longDesc) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<FrameworkError ExitStatus=\"1\" Type=\"" << shortDesc << "\" >\n";
      msg << "<![CDATA[\n" << longDesc << "\n]]>\n";
      msg << "</FrameworkError>\n";
   //LogError("FwkJob") << msg.str();
      msg << std::flush;
    }
  }

  void
  JobReport::reportAnalysisFile(std::string const& fileName, std::map<std::string, std::string> const& fileData) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      //std::ostringstream msg;
      msg << "<AnalysisFile>\n"
	  << "  <FileName>" << TiXmlText(fileName) << "</FileName>\n";

      typedef std::map<std::string, std::string>::const_iterator const_iterator;
      for (const_iterator pos = fileData.begin(), posEnd = fileData.end(); pos != posEnd; ++pos) {
        msg <<  "  <" << pos->first
	    <<  "  Value=\"" << pos->second  << "\" />"
	    <<  "\n";
      }
      msg << "</AnalysisFile>\n";
      //LogError("FwkJob") << msg.str();
      msg << std::flush;
    }
  }

  void
  JobReport::reportError(std::string const& shortDesc,
			 std::string const& longDesc,
			 int const& exitCode) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      //std::ostringstream msg;
      msg << "<FrameworkError ExitStatus=\""<< exitCode
    	<< "\" Type=\"" << shortDesc << "\" >\n";
      msg << "<![CDATA[\n" << longDesc << "\n]]>\n";
      msg << "</FrameworkError>\n";
      //LogError("FwkJob") << msg.str();
      msg << std::flush;
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
      msg << skipped << "\n";
      msg << std::flush;
      //LogInfo("FwkJob") << msg.str();
    }
  }

  void
  JobReport::reportTimingInfo(std::map<std::string, double> const& timingData) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<TimingService>\n";
      typedef std::map<std::string, double>::const_iterator const_iterator;
      for (const_iterator pos = timingData.begin(), posEnd = timingData.end(); pos != posEnd; ++pos) {
        msg <<  "  <" << pos->first
        <<  "  Value=\"" << pos->second  << "\" />"
        <<  "\n";
      }
      msg << "</TimingService>\n";
      //LogInfo("FwkJob") << msg.str();
      msg << std::flush;
    }
  }

  void
  JobReport::reportMemoryInfo(std::map<std::string, double> const& memoryData, std::map<std::string, double> const& memoryProperties) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<MemoryService>\n";
      typedef std::map<std::string, double>::const_iterator const_iterator;
      for (const_iterator pos = memoryData.begin(), posEnd = memoryData.end(); pos != posEnd; ++pos) {
        msg <<  "  <" << pos->first
        <<  "  Value=\"" << pos->second  << "\" />"
        <<  "\n";
      }
      reportMachineMemoryProperties(memoryProperties);
      msg << "</MemoryService>\n";
      msg << std::flush;
    }
  }

  void
  JobReport::reportMemoryInfo(std::vector<std::string> const& memoryData) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<MemoryService>\n";

      typedef std::vector<std::string>::const_iterator const_iterator;
      for (const_iterator pos = memoryData.begin(), posEnd = memoryData.end(); pos != posEnd; ++pos) {
        msg << *pos << "\n";
      }
      msg << "</MemoryService>\n";
      msg << std::flush;
    }
  }

  void JobReport::reportCPUInfo(std::map<std::string, std::map<std::string, std::string> > const& CPUData) {

    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<CPUService>\n";

      typedef std::map<std::string, std::map<std::string, std::string> >::const_iterator core_iter;

      for (core_iter core_pos = CPUData.begin(), core_posEnd = CPUData.end(); core_pos != core_posEnd; ++core_pos) {
	msg << "  <CPUCore Core=\"" << core_pos->first << "\">\n";
        typedef std::map<std::string, std::string>::const_iterator property_iter;
      	for (property_iter property_pos = core_pos->second.begin(), property_posEnd = core_pos->second.end();
	  property_pos != property_posEnd; ++property_pos) {
       	    msg <<  "    <Property Name=\"" << property_pos->first << "\">" << property_pos->second << "</Property>\n";
        }
	msg << "  </CPUCore>\n";
      }
      msg << "</CPUService>\n";
      msg << std::flush;
    }
  }

  void
  JobReport::reportMachineMemoryProperties(std::map<std::string, double> const& memoryProperties) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "  <Memory>\n";
      typedef std::map<std::string, double>::const_iterator const_iterator;
      for (const_iterator pos = memoryProperties.begin(), posEnd = memoryProperties.end(); pos != posEnd; ++pos) {
       msg <<  "    <Property Name=\"" << pos->first << "\">" << pos->second << "</Property>\n";
      }
      msg << "  </Memory>\n";
      msg << std::flush;
    }
  }

  void
  JobReport::reportMessageInfo(std::map<std::string, double> const& messageData) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<MessageSummary>\n";
      typedef std::map<std::string, double>::const_iterator const_iterator;
      for (const_iterator pos = messageData.begin(), posEnd = messageData.end(); pos != posEnd; ++pos) {
        msg <<  "  <" << pos->first
        <<  "  Value=\"" << pos->second  << "\" />"
        <<  "\n";
      }
      msg << "</MessageSummary>\n";
      msg << std::flush;
    }
  }

  void
  JobReport::reportStorageStats(std::string const& data) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<StorageStatistics>\n"
	<< data << "\n"
	<<  "</StorageStatistics>\n";
      //LogInfo("FwkJob") << msg.str();
      msg << std::flush;
    }
  }
  void
  JobReport::reportGeneratorInfo(std::string const& name, std::string const& value) {
    impl_->addGeneratorInfo(name, value);
  }

  void JobReport::reportRandomStateFile(std::string const& name) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<RandomServiceStateFile>\n"
	<< TiXmlText(name) << "\n"
	<<  "</RandomServiceStateFile>\n";
      //LogInfo("FwkJob") << msg.str();
      msg << std::flush;
    }
  }

  void
  JobReport::reportPSetHash(std::string const& hashValue) {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<PSetHash>"
	<<  hashValue
	<<  "</PSetHash>\n";
      //LogInfo("FwkJob") << msg.str();
      msg << std::flush;
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

    for (iterator f = impl_->outputFiles_.begin(), fEnd = impl_->outputFiles_.end(); f != fEnd; ++f) {

      msg << "\n<File>";
      msg << *f;

      msg << "\n<LumiSections>";
      //typedef std::vector<JobReport::LumiSectionReport>::iterator Iter;
      //for (Iter iLumi = f->lumiSections.begin(),
      //     iLumiEnd = f->lumiSections.end();
      //     iLumi != iLumiEnd;  ++iLumi) {
      //  msg << *iLumi;
      //}
      //msg << "\n</LumiSections>\n";
      msg << "\n<Inputs>";
      typedef std::vector<JobReport::Token>::iterator iterator;
      for (iterator iInput = f->contributingInputs.begin(),
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
