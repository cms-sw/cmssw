#ifndef Services_JobReport_h
#define Services_JobReport_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     JobReport
// 
/**\class JobReport JobReport.h FWCore/Services/interface/JobReport.h

Description: A service that collections job handling information.

Usage:
The JobReport service collects 'job handling' information (currently
file handling) from several sources, collates the information, and
at appropriate intervales, reports the information to the job report,
through the MessageLogger.

*/

//
// Original Author:  Marc Paterno
// $Id: JobReport.h,v 1.1 2006/03/21 22:11:35 paterno Exp $
//

#include <cstddef>
#include <string>

#include "boost/scoped_ptr.hpp"


#include "FWCore/Services/interface/ServicesFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"


namespace edm 
{
  namespace service 
  {
    class JobReport 
    {
    public:

      JobReport(ParameterSet const& ps, ActivityRegistry& reg);

      ~JobReport();
         
      void postEndJob();

      /// Report that an input file has been opened.
      /// The returned Token should be used for later identification
      /// of this file.
      Token inputFileOpened(std::string const& physicalFileName,
			    std::string const& logicalFileName,
			    std::string const& catalog,
			    std::string const& inputSourceClassName,
			    std::string const& moduleLabel,
			    std::vector<std::string> const& branchNames);

      /// Report that the event with the given id has been read from
      /// the file identified by the given Token.
      void eventReadFromFile(Token fileToken, edm::EventID const& id);

      /// Report that the input file identified by the given Token has
      /// been closed. An exception will be thrown if the given Token
      /// was not obtained from inputFileOpened.
      void inputFileClosed(Token fileToken);

      /// Report that an output file has been opened.
      /// The returned Token should be used for later identification
      /// of this file.
      Token outputFileOpened(std::string const& physicalFileName,
			     std::string const& logicalFileName,
			     std::string const& catalog,
			     std::string const& outputModuleClassName,
			     std::string const& moduleLabel,
			     std::vector<std::string> const& branchNames);

      /// Report that the event with the given id has been written to
      /// the file identified by the given Token.
      void eventWrittenToFile(Token fileToken, edm::EventID const& id);
      
      /// Report that the output file identified by the given Token has
      /// been closed. An exception will be thrown if the given Token
      /// was not obtained from outputFileOpened.
      void outputFileClosed(Token fileToken);

      void frameworkShutdownOnFailure();
      void reportSkippedEvent(edm::EventID const& id);



    private:
      struct JobReportImpl;
      boost::scoped_ptr<JobReportImpl> impl;
    };
  }
}

#endif
