// -*- C++ -*-
//
// Package:     Services
// Class  :     JobReport
// 
//
// Original Author:  Marc Paterno
// $Id: JobReport.cc,v 1.6 2006/03/05 16:42:27 chrjones Exp $
//


#include "FWCore/Services/interface/JobReport.h"
#include "FWCore/Services/interface/Files.h"

#include <iostream>


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
  
      return impl->inputFiles_.size()-1;
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

      // ... not yet implemented ...

      // After setting the file to 'closed', we will no longer be able
      // to reference it by ID.
      f.fileHasBeenClosed = true;
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

      // ... not yet implemented ...      
      // After setting the file to 'closed', we will no longer be able
      // to reference it by ID.
      f.fileHasBeenClosed = true;
    }

    void
    JobReport::frameworkShutdownOnFailure()
    {
      // Dump information to the MessageLogger's JobSummary
      // about the files that aren't already closed,
      // and whatever summary information is wanted.

      // ... not yet implemented ...      

      // Maybe we should have a member function called from both
      // postEndJob() and frameworkShutdownOnFailure(), so that common
      // elements are reported through common code.

    }
  } // namespace service
} //namspace edm
