#if !defined(STOR_PARAMETER_H)
#define STOR_PARAMETER_H

// Created by Markus Klute on 2007 Jan 29.
// $Id: Parameter.h,v 1.6 2008/04/21 12:17:24 loizides Exp $
//
// holds configuration parameter for StorageManager
//

#include <toolbox/net/Utils.h>
#include <string>

namespace stor 
{
  class Parameter
    {
    public:
      Parameter() :
	fileCatalog_("summaryCatalog.txt"),
	smInstance_("0"),
	hostName_(toolbox::net::getHostName()),
	initialSafetyLevel_(0),
        fileName_("storageManager"),
        filePath_("/scratch2/cheung"),
        setupLabel_("mtcc"),
	maxFileSize_(-1),
        highWaterMark_(0.9),
        lumiSectionTimeOut_(10.0),
        exactFileSizeTest_(false)
      {
        // strip domainame
         std::string::size_type pos = hostName_.find('.');  
         if ( pos != std::string::npos ) {  
           std::string basename = hostName_.substr(0,pos);  
           hostName_ = basename;
         }
      }

      const std::string& fileCatalog()        const {return fileCatalog_;}
      const std::string& smInstance()         const {return smInstance_;}
      const std::string& host()               const {return hostName_;}
      const std::string& fileName()           const {return fileName_;}
      const std::string& filePath()           const {return filePath_;}
      const std::string& setupLabel()         const {return setupLabel_;}
      int    maxFileSize()             const {return maxFileSize_;} 
      double highWaterMark()           const {return highWaterMark_;}
      double lumiSectionTimeOut()      const {return lumiSectionTimeOut_;}
      bool exactFileSizeTest()         const {return exactFileSizeTest_;}
      int initialSafetyLevel()         const {return initialSafetyLevel_;}

      // not efficient to pass object but tolerable
      void setFileCatalog       (std::string x) {fileCatalog_=x;}
      void setSmInstance        (std::string x) {smInstance_=x;}
      void setfileName          (std::string x) {fileName_=x;}
      void setfilePath          (std::string x) {filePath_=x;}
      void setsetupLabel        (std::string x) {setupLabel_=x;}
      void setmaxFileSize               (int x) {maxFileSize_=x;}
      void sethighWaterMark          (double x) {highWaterMark_=x;}
      void setlumiSectionTimeOut     (double x) {lumiSectionTimeOut_=x;}
      void setExactFileSizeTest        (bool x) {exactFileSizeTest_=x;}
      void initialSafetyLevel   (int i)         {initialSafetyLevel_=i;}

   private:
      std::string fileCatalog_;
      std::string smInstance_;
      std::string hostName_;
      int         initialSafetyLevel_;
      std::string fileName_;
      std::string filePath_;
      std::string setupLabel_;
      int         maxFileSize_; 
      double      highWaterMark_;
      double      lumiSectionTimeOut_;
      bool        exactFileSizeTest_;
    }; 
}

#endif

