
// -*- C++ -*-
//
// Package:     Services
// Class  :     MessageLogger
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
// $Id: JobReport.cc,v 1.44 2008/09/30 21:42:21 evansde Exp $
//

/*

Changes Log 1: 2009/01/14 10:29:00, Natalia Garcia Nebot
        Modified and added some methods to report CPU and memory information from /proc/cpuinfo
        and /proc/meminfo files and Memory statistics

*/

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>
#include <sstream>



namespace edm
{
    /*
     * Note that output formatting is spattered across these classes
     * If something outside these classes requires access to the
     * same formatting then we need to refactor it into a common library
     */
  template <typename S>
    S&
    print (S& os, JobReport::InputFile const& f) {

      os << "\n<InputFile>";
      formatFile(f, os);
      os << "\n<InputSourceClass>" << f.inputSourceClassName
	 << "</InputSourceClass>";
      os << "\n<EventsRead>" << f.numEventsRead << "</EventsRead>";
      return os;
    }


  template <typename S>
    S&
    print (S& os, JobReport::OutputFile const& f) {
      formatFile(f, os);
      os << "\n<OutputModuleClass>"
			<< f.outputModuleClassName
			<< "</OutputModuleClass>";
      os << "\n<TotalEvents>"
			<< f.numEventsWritten
			<< "</TotalEvents>\n";
      os << "\n<DataType>"
			<< f.dataType
			<< "</DataType>\n";
      os << "\n<BranchHash>"
			<< f.branchHash
			<< "</BranchHash>\n";

      return os;
    }

  template <typename S>
    S&
    print (S& os,
	   JobReport::RunReport const& rep){
    os << "\n<Run ID=\"" 
       <<rep.runNumber
       << "\">\n";
    
    std::set<unsigned int>::iterator il;
    for (il = rep.lumiSections.begin(); il != rep.lumiSections.end();
	 ++il){
      os << "   <LumiSection ID=\"" << *il << "\"/>\n";
      
    }
    
    os << "</Run>\n";
    

	return os;
     }

  std::ostream& operator<< (std::ostream& os, JobReport::InputFile const& f) {
    return print(os,f);
  }
  std::ostream& operator<< (std::ostream& os, JobReport::OutputFile const& f){
    return print(os,f);
  }

  std::ostream& operator<< (std::ostream& os, JobReport::RunReport const& f){
    return print(os,f);
  }

  //To talk to MessageLogger directly
  edm::MessageSender& operator<< (edm::MessageSender& os, JobReport::InputFile const& f) {
    return print(os,f);
  }
  edm::MessageSender& operator<< (edm::MessageSender& os, JobReport::OutputFile const& f){
    return print(os,f);
  }
  edm::MessageSender& operator<< (edm::MessageSender& os, JobReport::RunReport const& f){
    return print(os,f);
  }
//  edm::MessageSender& operator<< (edm::MessageSender& os, JobReport::LumiSectionReport const& rep) {
//    return print(os,rep);
//  }
//


    JobReport::InputFile& JobReport::JobReportImpl::getInputFileForToken(JobReport::Token t) {
	if (t >= inputFiles_.size() ) {
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
	if (t >= outputFiles_.size() ) {
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
	std::vector<JobReport::OutputFile>::iterator outFile;
	for (outFile = outputFiles_.begin();
	     outFile != outputFiles_.end();
	     outFile++){
	  outFile->contributingInputs.push_back(t);
	}
    }
    /*
     * get a vector of Tokens for all currently open
     * input files.
     * Used when a new output file is opened, all currently open
     * input file tokens are used to initialise its list of contributors
     */
    std::vector<JobReport::Token> JobReport::JobReportImpl::openInputFiles(void) {
	std::vector<JobReport::Token> result;
	for (unsigned int i = 0; i < inputFiles_.size(); ++i) {
	  JobReport::InputFile inFile = inputFiles_[i];
	  if ( inFile.fileHasBeenClosed == false){
	    result.push_back(i);
	  }
	}
	return result;
    }

    /*
     * get a vector of Tokens for all currently open
     * output files.
     *
     */
    std::vector<JobReport::Token> JobReport::JobReportImpl::openOutputFiles(void) {
	std::vector<JobReport::Token> result;
	for (unsigned int i = 0; i < outputFiles_.size(); ++i) {
	  JobReport::OutputFile outFile = outputFiles_[i];
	  if ( outFile.fileHasBeenClosed == false){
	    result.push_back(i);
	  }
	}
	return result;
    }

    /*
     * Write anJobReport::InputFile object to the Logger
     * Generate XML string forJobReport::InputFile instance and dispatch to
     * job report via MessageLogger
     */
    void JobReport::JobReportImpl::writeInputFile(JobReport::InputFile const& f){
      if(ost_) {
        *ost_ <<f ;
	*ost_ << "\n<Runs>";
	std::map<JobReport::RunNumber, JobReport::RunReport>::const_iterator iRun;
	for (iRun = f.runReports.begin();
	     iRun != f.runReports.end(); iRun++){
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
        *ost_ <<f;

	*ost_ << "\n<Runs>";
	std::map<JobReport::RunNumber, JobReport::RunReport>::const_iterator iRun;
	for (iRun = f.runReports.begin();
	     iRun != f.runReports.end(); iRun++){
	  *ost_ << iRun->second;
	}
	*ost_ << "\n</Runs>\n";

	*ost_ << "\n<Inputs>";
 	std::vector<JobReport::Token>::const_iterator iInput;
 	for (iInput = f.contributingInputs.begin();
 	     iInput != f.contributingInputs.end(); iInput++) {
 	    JobReport::InputFile inpFile = inputFiles_[*iInput];
 	    *ost_ <<"\n<Input>";
 	    *ost_ <<"\n  <LFN>" << inpFile.logicalFileName << "</LFN>";
 	    *ost_ <<"\n  <PFN>" << inpFile.physicalFileName << "</PFN>";
 	    *ost_ <<"\n</Input>";
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
      std::vector<JobReport::InputFile>::iterator ipos;
      std::vector<JobReport::OutputFile>::iterator opos;
      for (ipos = inputFiles_.begin(); ipos != inputFiles_.end(); ++ipos) {
          if (!(ipos->fileHasBeenClosed)) {
            writeInputFile(*ipos);
          }
      }
      for (opos = outputFiles_.begin(); opos != outputFiles_.end(); ++opos) {
	if (!(opos->fileHasBeenClosed)) {
	  writeOutputFile(*opos);
	}
      }
    }

  void JobReport::JobReportImpl::addGeneratorInfo(std::string const& name,
						  std::string const& value){

    generatorInfo_[name] = value;
  }

  void JobReport::JobReportImpl::writeGeneratorInfo(void){
    if(ost_) {
      *ost_ << "\n<GeneratorInfo>\n";
      std::map<std::string, std::string>::iterator pos;
      for (pos = generatorInfo_.begin(); pos != generatorInfo_.end(); ++pos){
        std::ostringstream msg;
        msg << "\n<Data Name=\"" << pos->first
          << "\" Value=\"" << pos->second << "\"/>";
        *ost_ << msg.str();
      }
      *ost_ << "</GeneratorInfo>\n";
    }
  }

  void JobReport::JobReportImpl::associateRun(unsigned int runNumber){
    std::vector<Token> openFiles = openOutputFiles();
    std::vector<Token>::iterator iToken;
    for (iToken = openFiles.begin(); iToken != openFiles.end(); iToken++){
      JobReport::OutputFile & theFile = outputFiles_[*iToken];
      
      //
      // check run is known to file
      // if not, add a run report for that run     
      if (theFile.runReports.count(runNumber) == 0){
	JobReport::RunReport newReport = JobReport::RunReport();
	newReport.runNumber = runNumber;
	theFile.runReports.insert(
		 std::make_pair(runNumber, newReport)
		 );
      }
      
    }
  }

  void JobReport::JobReportImpl::associateInputRun(unsigned int runNumber){
    std::vector<Token> openFiles = openInputFiles();
    std::vector<Token>::iterator iToken;
    for (iToken = openFiles.begin(); iToken != openFiles.end(); iToken++){
      JobReport::InputFile & theFile = inputFiles_[*iToken];
	
      
      //
      // check run is known to file
      // if not, add a run report for that run     
      if (theFile.runReports.count(runNumber) == 0){
	JobReport::RunReport newReport = JobReport::RunReport();
	newReport.runNumber = runNumber;
	theFile.runReports.insert(
		 std::make_pair(runNumber, newReport)
		 );
      }
      

    }
  }


  void JobReport::JobReportImpl::associateLumiSection(unsigned int runNumber, unsigned int lumiSect){
    std::vector<Token> openFiles = openOutputFiles();
    std::vector<Token>::iterator iToken;
    for (iToken = openFiles.begin(); iToken != openFiles.end(); iToken++){
      //
      // Loop over all open output files
      //
      JobReport::OutputFile & theFile = outputFiles_[*iToken];
      
	
      
      //
      // check run is known to file
      // if not, add a run report for that run     
      if (theFile.runReports.count(runNumber) == 0){
	JobReport::RunReport newReport = JobReport::RunReport();
	newReport.runNumber = runNumber;
	theFile.runReports.insert(
		 std::make_pair(runNumber, newReport)
		 );
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


  void JobReport::JobReportImpl::associateInputLumiSection(unsigned int runNumber, unsigned int lumiSect){
    std::vector<Token> openFiles = openInputFiles();
    std::vector<Token>::iterator iToken;
    for (iToken = openFiles.begin(); iToken != openFiles.end(); iToken++){
      //
      // Loop over all open input files
      //
      JobReport::InputFile & theFile = inputFiles_[*iToken];

      //
      // check run is known to file
      // if not, add a run report for that run     
      if (theFile.runReports.count(runNumber) == 0){
	JobReport::RunReport newReport = JobReport::RunReport();
	newReport.runNumber = runNumber;
	theFile.runReports.insert(
		 std::make_pair(runNumber, newReport)
		 );
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
      *(impl_->ost_)<<"</FrameworkJobReport>\n"<<std::flush;
    }
  }

    JobReport::JobReport() :
      impl_(new JobReportImpl(0)) {
    }

    JobReport::JobReport(std::ostream* iOstream) :
       impl_(new JobReportImpl(iOstream) ) {
         if(impl_->ost_) {
           *(impl_->ost_)<<"<FrameworkJobReport>\n";
         }
       }

    JobReport::Token
    JobReport::inputFileOpened(std::string const& physicalFileName,
			       std::string const& logicalFileName,
			       std::string const& catalog,
			       std::string const& inputSourceClassName,
			       std::string const& moduleLabel,
			       std::string const& guid,
			       std::vector<std::string> const& branchNames)
    {
      // Do we have to worry about thread safety here? Or is this
      // service used in a way to make this safe?
      impl_->inputFiles_.push_back(JobReport::InputFile());
      JobReport::InputFile& r = impl_->inputFiles_.back();

      r.logicalFileName      = logicalFileName;
      r.physicalFileName     = physicalFileName;
      r.catalog              = catalog;
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

    JobReport::Token
    JobReport::inputFileOpened(std::string const& physicalFileName,
			       std::string const& logicalFileName,
			       std::string const& catalog,
			       std::string const& inputSourceClassName,
			       std::string const& moduleLabel,
			       std::vector<std::string> const& branchNames)
    {
      return this->inputFileOpened(physicalFileName,
				   logicalFileName,
				   catalog,
				   inputSourceClassName,
				   moduleLabel,
				   "",
				   branchNames);
    }

    void
    JobReport::eventReadFromFile(JobReport::Token fileToken, unsigned int run, unsigned int)
    {
      JobReport::InputFile& f = impl_->getInputFileForToken(fileToken);
      f.numEventsRead++;
      //f.runsSeen.insert(run);
    }

    void
    JobReport::reportDataType(Token fileToken, std::string const& dataType)
    {
      JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
      f.dataType = dataType;
    }

    void
    JobReport::inputFileClosed(JobReport::Token fileToken)
    {
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
				std::vector<std::string> const& branchNames)
    {
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
      r.contributingInputs = std::vector<JobReport::Token>(impl_->openInputFiles());
      return impl_->outputFiles_.size()-1;
    }

    JobReport::Token
    JobReport::outputFileOpened(std::string const& physicalFileName,
				std::string const& logicalFileName,
				std::string const& catalog,
				std::string const& outputModuleClassName,
				std::string const& moduleLabel,
				std::string const& guid,
				std::string const& dataType,
				std::vector<std::string> const& branchNames)
    {
      return this->outputFileOpened(physicalFileName,
				    logicalFileName,
				    catalog,
				    outputModuleClassName,
				    moduleLabel,
				    guid,
				    "",
				    "NO_BRANCH_HASH",
				    branchNames);


    }

    JobReport::Token
    JobReport::outputFileOpened(std::string const& physicalFileName,
				std::string const& logicalFileName,
				std::string const& catalog,
				std::string const& outputModuleClassName,
				std::string const& moduleLabel,
				std::string const& guid,
				std::vector<std::string> const& branchNames)
    {
      return this->outputFileOpened(physicalFileName,
				    logicalFileName,
				    catalog,
				    outputModuleClassName,
				    moduleLabel,
				    guid,
				    "",
				    branchNames);

    }



  JobReport::Token
  JobReport::outputFileOpened(std::string const& physicalFileName,
			      std::string const& logicalFileName,
			      std::string const& catalog,
			      std::string const& outputModuleClassName,
			      std::string const& moduleLabel,
			      std::vector<std::string> const& branchNames)
  {

    return this->outputFileOpened(physicalFileName,
				  logicalFileName,
				  catalog,
				  outputModuleClassName,
				  moduleLabel,
				  "",
				  "",
				  branchNames);
  }



    void
    JobReport::eventWrittenToFile(JobReport::Token fileToken, unsigned int run, unsigned int)
    {
      JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
      f.numEventsWritten++;
      //f.runsSeen.insert(run);
    }


    void
    JobReport::outputFileClosed(JobReport::Token fileToken)
    {
      JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
      // Dump information to the MessageLogger's JobSummary
      // about this file.
      // After setting the file to 'closed', we will no longer be able
      // to reference it by ID.
      f.fileHasBeenClosed = true;
      impl_->writeOutputFile(f);

    }

    void
    JobReport::overrideEventsWritten(Token fileToken, const int eventsWritten)
    {
      // Get the required output file instance using the token
      JobReport::OutputFile& f = impl_->getOutputFileForToken(fileToken);
      // set the eventsWritten parameter to the provided value
      f.numEventsWritten = eventsWritten;

    }

    void
    JobReport::overrideEventsRead(Token fileToken, const int eventsRead)
    {
      // Get the required input file instance using the token
      JobReport::InputFile& f = impl_->getInputFileForToken(fileToken);
      // set the events read parameter to the provided value
      f.numEventsRead = eventsRead;

    }

    void
    JobReport::overrideContributingInputs(Token outputToken,
					  std::vector<Token> const& inputTokens)
    {
       // Get the required output file instance using the token
      JobReport::OutputFile& f = impl_->getOutputFileForToken(outputToken);
      // override its contributing inputs data
      f.contributingInputs = inputTokens;
    }

    void
    JobReport::reportSkippedEvent(unsigned int run, unsigned int event)
    {
      if(impl_->ost_) {
        std::ostream& msg = *(impl_->ost_);
        msg << "<SkippedEvent Run=\"" << run << "\"";
        msg << " Event=\"" << event << "\" />\n";
        msg <<std::flush;
        //LogInfo("FwkJob") << msg.str();
      }
    }

  void
  JobReport::reportLumiSection(unsigned int run, unsigned int lumiSectId){
    impl_->associateLumiSection(run, lumiSectId);
  }
  void
  JobReport::reportInputLumiSection(unsigned int run, unsigned int lumiSectId){
    impl_->associateInputLumiSection(run, lumiSectId);
  }

  void
  JobReport::reportRunNumber(unsigned int run){
    impl_->associateRun(run);
  }
  void
  JobReport::reportInputRunNumber(unsigned int run){
    impl_->associateInputRun(run);
  }


  void
  JobReport::reportError(std::string const& shortDesc,
  			 std::string const& longDesc)
  {
    if(impl_->ost_) {
      std::ostream& msg =*(impl_->ost_);
      msg << "<FrameworkError ExitStatus=\"1\" Type=\"" << shortDesc <<"\" >\n";
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
	  << "  <FileName>" << fileName <<"</FileName>\n";

      std::map<std::string, std::string>::const_iterator pos;
      for (pos = fileData.begin(); pos != fileData.end(); ++pos){
        msg <<  "  <" << pos->first
	    <<  "  Value=\"" << pos->second  << "\" />"
	    <<  "\n";
      }

      msg << "</AnalysisFile>\n";
      //LogError("FwkJob") << msg.str();
      msg <<std::flush;
    }


  }


  void
  JobReport::reportError(std::string const& shortDesc,
			 std::string const& longDesc,
			 int const& exitCode)
  {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      //std::ostringstream msg;
      msg << "<FrameworkError ExitStatus=\""<< exitCode
    	<<"\" Type=\"" << shortDesc <<"\" >\n";
      msg << "<![CDATA[\n" << longDesc << "\n]]>\n";
      msg << "</FrameworkError>\n";
      //LogError("FwkJob") << msg.str();
      msg <<std::flush;
    }
  }

  void
  JobReport::reportSkippedFile(std::string const& pfn,
			       std::string const& lfn) {

    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<SkippedFile Pfn=\"" << pfn << "\"";
      msg << " Lfn=\"" << lfn << "\" />\n";
      msg <<std::flush;
      //LogInfo("FwkJob") << msg.str();

    }
  }

  void
  JobReport::reportTimingInfo(std::map<std::string, double> const& timingData){

    if(impl_->ost_) {
      std::ostream& msg=*(impl_->ost_);
      msg << "<TimingService>\n";
      std::map<std::string, double>::const_iterator pos;
      for (pos = timingData.begin(); pos != timingData.end(); ++pos){
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
  JobReport::reportMemoryInfo(std::map<std::string, double> const& memoryData, std::map<std::string, double> const& memoryProperties){

    if(impl_->ost_) {
      std::ostream& msg=*(impl_->ost_);
      msg << "<MemoryService>\n";
      std::map<std::string, double>::const_iterator pos;
      for (pos = memoryData.begin(); pos != memoryData.end(); ++pos){
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
  JobReport::reportMemoryInfo(std::vector<std::string> const& memoryData){

    if(impl_->ost_) {
      std::ostream& msg=*(impl_->ost_);
      msg << "<MemoryService>\n";

      std::vector<std::string>::const_iterator pos;
      for (pos = memoryData.begin(); pos != memoryData.end(); ++pos){
        msg << *pos << "\n";
      }
      msg << "</MemoryService>\n";
      msg << std::flush;
    }
  }

  void JobReport::reportCPUInfo(std::map<std::string, std::map<std::string, std::string> > const& CPUData){

    if(impl_->ost_) {
      std::ostream& msg=*(impl_->ost_);
      msg << "<CPUService>\n";

      std::map<std::string, std::map<std::string, std::string> >::const_iterator core_pos;
      std::map<std::string, std::string>::const_iterator property_pos;

      for (core_pos = CPUData.begin(); core_pos != CPUData.end(); ++core_pos){
	msg << "  <CPUCore Core=\"" << core_pos->first << "\">\n";
      	for (property_pos = core_pos->second.begin(); property_pos != core_pos->second.end(); ++property_pos){
       	    msg <<  "    <Property Name=\"" << property_pos->first << "\">" << property_pos->second << "</Property>\n";
        }
	msg << "  </CPUCore>\n";
      }

      msg << "</CPUService>\n";
      msg << std::flush;
    }
  }

  void
  JobReport::reportMachineMemoryProperties(std::map<std::string, double> const& memoryProperties){
    if(impl_->ost_) {
      std::ostream& msg=*(impl_->ost_);
      msg << "  <Memory>\n";
      std::map<std::string, double>::const_iterator pos;
      for (pos = memoryProperties.begin(); pos != memoryProperties.end(); ++pos){
       msg <<  "    <Property Name=\"" << pos->first << "\">" << pos->second << "</Property>\n";
      }
      msg << "  </Memory>\n";
      msg << std::flush;
    }
  }

  void
  JobReport::reportMessageInfo(std::map<std::string, double> const& messageData){

    if(impl_->ost_) {
      std::ostream& msg=*(impl_->ost_);
      msg << "<MessageSummary>\n";
      std::map<std::string, double>::const_iterator pos;
      for (pos = messageData.begin(); pos != messageData.end(); ++pos){
        msg <<  "  <" << pos->first
        <<  "  Value=\"" << pos->second  << "\" />"
        <<  "\n";
      }
      msg << "</MessageSummary>\n";
      msg << std::flush;
    }
  }

  void
  JobReport::reportStorageStats(std::string const& data)
  {
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
  JobReport::reportGeneratorInfo(std::string const&  name, std::string const&  value)
  {

    impl_->addGeneratorInfo(name, value);
  }


  void JobReport::reportRandomStateFile(std::string const& name)
  {
    if(impl_->ost_) {
      std::ostream& msg = *(impl_->ost_);
      msg << "<RandomServiceStateFile>\n"
        << name << "\n"
	<<  "</RandomServiceStateFile>\n";
      //LogInfo("FwkJob") << msg.str();
      msg << std::flush;
    }
  }

  void
  JobReport::reportPSetHash(std::string const& hashValue)
  {
    if(impl_->ost_){
      std::ostream& msg =*(impl_->ost_);
      msg << "<PSetHash>"
        <<  hashValue
	<<  "</PSetHash>\n";
      //LogInfo("FwkJob") << msg.str();
      msg << std::flush;
    }
  }


  void
  JobReport::reportPerformanceSummary(std::string const& metricClass,
				      std::map<std::string, std::string> const& metrics)
  {
    if(impl_->ost_){
      std::ostream& msg =*(impl_->ost_);
      msg << "<PerformanceReport>\n"
        << "  <PerformanceSummary Metric=\"" << metricClass << "\">\n";

      std::map<std::string, std::string>::const_iterator iter;
      for( iter = metrics.begin(); iter != metrics.end(); ++iter ) {
        msg << "    <Metric Name=\"" << iter->first << "\" "
        <<"Value=\"" << iter->second << "\"/>\n";
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
					std::map<std::string, std::string> const& metrics)
  {
    if(impl_->ost_){
      std::ostream& msg =*(impl_->ost_);
      msg << "<PerformanceReport>\n"
        << "  <PerformanceModule Metric=\"" << metricClass << "\" "
	<< " Module=\""<< moduleName << "\" >\n";

      std::map<std::string, std::string>::const_iterator iter;
      for( iter = metrics.begin(); iter != metrics.end(); ++iter ) {
        msg << "    <Metric Name=\"" << iter->first << "\" "
        <<"Value=\"" << iter->second << "\"/>\n";
      }

      msg << "  </PerformanceModule>\n"
	<< "</PerformanceReport>\n";
      msg << std::flush;
      //LogInfo("FwkJob") << msg.str();
    }
  }


  std::string
  JobReport::dumpFiles(void){

    std::ostringstream msg;

    std::vector<JobReport::OutputFile>::iterator f;

    for (f = impl_->outputFiles_.begin();
	 f != impl_->outputFiles_.end(); f++){

      msg << "\n<File>";
      msg << *f;

      msg << "\n<LumiSections>";
      //std::vector<JobReport::LumiSectionReport>::iterator iLumi;
      //for (iLumi = f->lumiSections.begin();
      //     iLumi != f->lumiSections.end(); iLumi++){
      //  msg << *iLumi;
      //}
      //msg << "\n</LumiSections>\n";
      msg << "\n<Inputs>";
      std::vector<JobReport::Token>::iterator iInput;
      for (iInput = f->contributingInputs.begin();
	   iInput != f->contributingInputs.end(); iInput++) {
	JobReport::InputFile inpFile = impl_->inputFiles_[*iInput];
	msg <<"\n<Input>";
	msg <<"\n  <LFN>" << inpFile.logicalFileName << "</LFN>";
	msg <<"\n  <PFN>" << inpFile.physicalFileName << "</PFN>";
	msg <<"\n</Input>";
      }
      msg << "\n</Inputs>";
      msg << "\n</File>";

    }

    return msg.str();

  }


} //namspace edm
