// -*- C++ -*-
//
// Package:     Services
// Class  :     JobReport
// 
//
// Original Author:  Marc Paterno
// $Id: JobReport.cc,v 1.1 2006/03/21 22:11:35 paterno Exp $
//


#include "FWCore/Services/interface/JobReport.h"
#include "FWCore/Services/interface/Files.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <sstream>
#include <algorithm>

using std::string;
using std::vector;

      
namespace edm
{
  namespace service
  {
 
    
    
    struct JobReport::JobReportImpl
    {
      
      
      // Helper functions to be called from the xFileClosed functions,
      // after the appropriate logging is done.
      /*
       * Note: We want to keep input files, so we can resolve tokens
       * when we close output files
       *
       * Note2: Erasing the Output files on close invalidates the Tokens
       * for existing output files since they are based on position in 
       * the vector, so I have left off implementing these methods for now
       */
      void removeInputFileRecord_(Token t); 
      void removeOutputFileRecord_(Token t);

      InputFile& getInputFileForToken(Token t)
      {
	if (t >= inputFiles_.size() )
	  {
	    throw edm::Exception(edm::errors::LogicError)
	      << "Access reported for input file with token "
	      << t
	      << " but no matching input file is found\n";
	  }

	 if (inputFiles_[t].fileHasBeenClosed)
	  {
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

      OutputFile& getOutputFileForToken(Token t)
      {
	if (t >= outputFiles_.size() )
	  {
	    throw edm::Exception(edm::errors::LogicError)
	      << "Access reported for output file with token "
	      << t
	      << " but no matching output file is found\n";
	  }
	if (outputFiles_[t].fileHasBeenClosed)
	  {
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
      void insertInputForOutputs( Token t )
      {
	std::vector<OutputFile>::iterator outFile;
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
      std::vector<Token> openInputFiles(void){
	std::vector<Token> result;
	for (unsigned int i = 0; i < inputFiles_.size(); i++){
	  InputFile inFile = inputFiles_[i];
	  if ( inFile.fileHasBeenClosed == false){
	    result.push_back(i);
	  }
	}
	return result;
      }

      /*
       * Write an InputFile object to the Logger 
       * Generate XML string for InputFile instance and dispatch to 
       * job report via MessageLogger
       */
      void writeInputFile(InputFile & f){
	LogInfo("FwkJob") << f;
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
      void writeOutputFile(OutputFile & f){
	LogInfo("FwkJob") << "\n<File>";
	LogInfo("FwkJob") << f;
 	LogInfo("FwkJob") << "\n<Inputs>";
	
 	std::vector<Token>::iterator iInput;
 	for (iInput = f.contributingInputs.begin(); 
 	     iInput != f.contributingInputs.end(); iInput++)
 	  {
 	    InputFile inpFile = inputFiles_[*iInput];
 	    LogInfo("FwkJob") <<"\n<Input>";
 	    LogInfo("FwkJob") <<"\n  <LFN>" << inpFile.logicalFileName << "</LFN>";
 	    LogInfo("FwkJob") <<"\n  <PFN>" << inpFile.physicalFileName << "</PFN>";
 	    LogInfo("FwkJob") <<"\n</Input>";
 	  }
 	LogInfo("FwkJob") << "\n</Inputs>";
 	LogInfo("FwkJob") << "\n</File>";
      }
      
      /*
       *  Flush all open files to logger in event of a problem.
       */
      void flushFiles(void){
	std::vector<InputFile>::iterator ipos;
	std::vector<OutputFile>::iterator opos;

	for (ipos = inputFiles_.begin(); ipos != inputFiles_.end(); ++ipos)
	  {
	    if ( !(ipos->fileHasBeenClosed) ){
	      writeInputFile(*ipos);
	    }
	  }
	for (opos = outputFiles_.begin(); opos != outputFiles_.end(); ++opos)
	  {
	    if ( !(opos->fileHasBeenClosed) ){
	      writeOutputFile(*opos);
	    }
	  }
      };

      std::vector<InputFile> inputFiles_;
      std::vector<OutputFile> outputFiles_;
    };

    
    


    JobReport::~JobReport() {}

    JobReport::JobReport(ParameterSet const& /*ps*/,
			 ActivityRegistry& reg) :
      impl(new JobReportImpl)
    {
      reg.watchPostEndJob(this, &JobReport::postEndJob);
      reg.watchJobFailure(this, &JobReport::frameworkShutdownOnFailure);

      // We don't handle PreProcessEvent, because we have to know *which
      // input file* was the event read from. Only the InputSource that
      // did the reading knows this.
    }

    void 
    JobReport::postEndJob()
    {
      // This will be called at end-of-job (obviously).
      // Dump information to the MessageLogger's JobSummary.

      // ... not yet implemented ...      

      // Maybe we should have a member function called from both
      // postEndJob() and frameworkShutdownOnFailure(), so that common
      // elements are reported through common code.

        //
       // Any files that are still open should be flushed to the report
      //
      impl->flushFiles();

    }


    Token
    JobReport::inputFileOpened(string const& physicalFileName,
			       string const& logicalFileName,
			       string const& catalog,
			       string const& inputSourceClassName,
			       string const& moduleLabel,
			       vector<string> const& branchNames)
    {
      // Do we have to worry about thread safety here? Or is this
      // service used in a way to make this safe?
      impl->inputFiles_.push_back(InputFile());
      InputFile& r = impl->inputFiles_.back();

      r.logicalFileName      = logicalFileName;
      r.physicalFileName     = physicalFileName;
      r.catalog              = catalog;
      r.inputSourceClassName = inputSourceClassName;
      r.moduleLabel          = moduleLabel;
      // r.runsSeen is not modified
      r.numEventsRead        = 0;
      r.branchNames          = branchNames;
      r.fileHasBeenClosed    = false;
    
      Token newToken = impl->inputFiles_.size()-1;
        //
       // Add the new input file token to all output files
      //  currently open.
      impl->insertInputForOutputs(newToken);
      return newToken;
    }

    void
    JobReport::eventReadFromFile(Token fileToken, edm::EventID const& id)
    {
      InputFile& f = impl->getInputFileForToken(fileToken);
      f.numEventsRead++;
      f.runsSeen.insert(id.run());
    }

    void
    JobReport::inputFileClosed(Token fileToken)
    {
      InputFile& f = impl->getInputFileForToken(fileToken);
      // Dump information to the MessageLogger's JobSummary
      // about this file.
      // After setting the file to 'closed', we will no longer be able
      // to reference it by ID.
      f.fileHasBeenClosed = true;
      impl->writeInputFile(f);
    }

    Token 
    JobReport::outputFileOpened(string const& physicalFileName,
				string const& logicalFileName,
				string const& catalog,
				string const& outputModuleClassName,
				string const& moduleLabel,
				vector<string> const& branchNames)
    {
      impl->outputFiles_.push_back(OutputFile());
      OutputFile& r = impl->outputFiles_.back();
      
      r.logicalFileName       = logicalFileName;
      r.physicalFileName      = physicalFileName;
      r.catalog               = catalog;
      r.outputModuleClassName = outputModuleClassName;
      r.moduleLabel           = moduleLabel;
      // r.runsSeen is not modified
      r.numEventsWritten      = 0;
      r.branchNames           = branchNames;
      r.fileHasBeenClosed     = false;
        //
       // Init list of contributors to list of open input file Tokens
      //
      r.contributingInputs = std::vector<Token>(impl->openInputFiles());
      return impl->outputFiles_.size()-1;
    }


    void
    JobReport::eventWrittenToFile(Token fileToken, edm::EventID const& id)
    {
      OutputFile& f = impl->getOutputFileForToken(fileToken);
      f.numEventsWritten++;
      f.runsSeen.insert(id.run());
    }


    void
    JobReport::outputFileClosed(Token fileToken)
    {
      OutputFile& f = impl->getOutputFileForToken(fileToken);
      // Dump information to the MessageLogger's JobSummary
      // about this file.
      // After setting the file to 'closed', we will no longer be able
      // to reference it by ID.
      f.fileHasBeenClosed = true;
      impl->writeOutputFile(f);

    }

    void
    JobReport::frameworkShutdownOnFailure()
    {
      // Dump information to the MessageLogger's JobSummary
      // about the files that aren't already closed,
      // and whatever summary information is wanted.

      // Maybe we should have a member function called from both
      // postEndJob() and frameworkShutdownOnFailure(), so that common
      // elements are reported through common code.
      impl->flushFiles();
    }
  } // namespace service
} //namspace edm
