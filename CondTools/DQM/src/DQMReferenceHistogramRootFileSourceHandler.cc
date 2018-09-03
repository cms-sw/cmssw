#include "CondTools/DQM/interface/DQMReferenceHistogramRootFileSourceHandler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <iostream>
#include <sstream>
#include <vector>

namespace popcon {
  DQMReferenceHistogramRootFileSourceHandler::DQMReferenceHistogramRootFileSourceHandler(const edm::ParameterSet & pset):
    m_name(pset.getUntrackedParameter<std::string>("name","DQMReferenceHistogramRootFileSourceHandler")),
    m_file(pset.getUntrackedParameter<std::string>("ROOTFile","./file.root")),
    m_zip(pset.getUntrackedParameter<bool>("zip",false)),
    m_since(pset.getUntrackedParameter<unsigned long long>("firstSince",1)),
    m_debugMode(pset.getUntrackedParameter<bool>("debug",false)) {
  }
  
  DQMReferenceHistogramRootFileSourceHandler::~DQMReferenceHistogramRootFileSourceHandler() {}
  
  void DQMReferenceHistogramRootFileSourceHandler::getNewObjects() {
    edm::LogInfo("DQMReferenceHistogramRootFileSourceHandler") << "[DQMReferenceHistogramRootFileSourceHandler::getNewObjects] for PopCon application " << m_name;
    if(m_debugMode){
	std::stringstream ss;
	ss << "\n------- " << m_name 
	   << " - > getNewObjects\n";
	if (this->tagInfo().size > 0){
	  //check what is already inside of the database
	  ss << "\ngot offlineInfo "<< this->tagInfo().name 
	     << ",\n size " << this->tagInfo().size 
	     << ",\n" << this->tagInfo().token 
	     << ",\n last object valid since " << this->tagInfo().lastInterval.first 
	     << ",\n token " << this->tagInfo().lastPayloadToken 
	     << ",\n UserText " << this->userTextLog()
	     << ";\n last entry info regarding the payload (if existing):" 
	     << ",\n logId"<<this->logDBEntry().logId 
	     << ",\n last record with the correct tag (if existing) has been written in the db " << this->logDBEntry().destinationDB
	     << ",\n provenance " << this->logDBEntry().provenance
	     << ",\n usertext " << this->logDBEntry().usertext
	     << ",\n iovtag " << this->logDBEntry().iovtag
	     << ",\n timetype " << this->logDBEntry().iovtimetype
	     << ",\n payload index " << this->logDBEntry().payloadIdx
	     << ",\n payload class " << this->logDBEntry().payloadClass 
	     << ",\n payload token " << this->logDBEntry().payloadToken
	     << ",\n execution time " << this->logDBEntry().exectime
	     << ",\n execution message " << this->logDBEntry().execmessage
	     << std::endl;
	  Ref payload = this->lastPayload();
	  ss << "size of last payload " << payload->size() << std::endl;
	} else {
	  ss << " First object for this tag " << std::endl;
	}
	edm::LogInfo("DQMReferenceHistogramRootFileSourceHandler") << ss.str();
    }
    edm::LogInfo("DQMReferenceHistogramRootFileSourceHandler") << "runnumber/first since = " << m_since << std::endl;
    if(m_since<=this->tagInfo().lastInterval.first){
      edm::LogInfo("DQMReferenceHistogramRootFileSourceHandler") 
	<< "[DQMReferenceHistogramRootFileSourceHandler::getNewObjects] \nthe current starting iov " << m_since
	<< "\nis not compatible with the last iov ("  
	<< this->tagInfo().lastInterval.first << ") open for the object " 
	<< this->logDBEntry().payloadClass << " \nin the db " 
	<< this->logDBEntry().destinationDB << " \n NO TRANSFER NEEDED"
	<< std::endl;
      return;
      }
    edm::LogInfo("DQMReferenceHistogramRootFileSourceHandler") 
      << "[DQMReferenceHistogramRootFileSourceHandler::getNewObjects] " << m_name << " getting data to be transferred "  << std::endl;
    FileBlob* rootFile = new FileBlob(m_file,m_zip);
    /*if(!this->tagInfo().size)
      m_since=1;
    else
      if (m_debugMode)
      m_since=this->tagInfo().lastInterval.first+1; */
    if(rootFile->size() != 0){
      edm::LogInfo("DQMReferenceHistogramRootFileSourceHandler") << "setting runnumber/first since = " << m_since << std::endl;
      this->m_to_transfer.push_back(std::make_pair(rootFile,m_since));
    } else {
      edm::LogError("DQMSummarySourceHandler") << "Root file " << m_file << " does not exist" << std::endl;
      delete rootFile;
    }
    edm::LogInfo("DQMSummarySourceHandler") << "------- " 
					    << m_name << " - > getNewObjects" 
					    << std::endl;
  }
  
  std::string DQMReferenceHistogramRootFileSourceHandler::id() const {return m_name;}
}
