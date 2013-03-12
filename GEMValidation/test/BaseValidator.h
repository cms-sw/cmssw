#ifndef _BaseValidator_h_
#define _BaseValidator_h_

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TH1D.h"
#include "TPad.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TArrayD.h"
#include "TMath.h"
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "TCut.h"

class BaseValidator
{
 public:
  ~BaseValidator() {}
  
  void makeValidationPlots();
  void makeValidationReport();

  void setInFileName(const std::string file){infile_ = file;}
  void setOutFileName(const std::string file){outfile_ = file;}
  void setTitle(const std::string title){title_ = title;}
  void setPriority(const std::string priority){priority_ = priority;}
  void setDateOfRequest(const std::string date){dateOfRequest_ = date;}
  void setDescription(const std::string desc){description_ = desc;}
  void setLinkToTwiki(const std::string link){linkToTwiki_ = link;}
  void setProductionStartDate(const std::string date){productionStartDate_ = date;}
  void setResponsible(const std::string resp){responsible_ = resp;}
  void setDataSetPath(const std::string dsp){dataSetPath_ = dsp;}
  void setProductionEndDate(const std::string date){productionEndDate_ = date;}
  void setDataSetSize(const std::string dss){dataSetSize_ = dss;}
  void setNumberOfEvents(const int n){numberOfEvents_ = n;}
  void setNumberOfJobs(const int n){numberOfJobs_ = n;}
  void setTimeToComplete(const std::string time){timeToComplete_ = time;}
  void setCrabConfiguration(const std::string crab){crabConfig_ = crab;}
  void setObsolete(bool obs){obsolete_ = obs;}
  void setDateOfObsoletion(const std::string date){dateOfObsoletion_ = date;}
  void setReasonForObsoletion(const std::string reason){reasonForObsoletion_ = reason;}
  void setDeleted(bool del){deleted_ = del;}
  
  std::string getInFileName(){return infile_;}
  std::string getOutFileName(){return outfile_;}
  std::string getTitle(){return title_;}
  std::string getPriority(){return priority_;}
  std::string getDateOfRequest(){return dateOfRequest_;}
  std::string getDescription(){return description_;}
  std::string getLinkToTwiki(){return linkToTwiki_;}
  std::string getProductionStartDate(){return productionStartDate_;}
  std::string getResponsible(){return responsible_;}
  std::string getDataSetPath(){return dataSetPath_;}
  std::string getProductionEndDate(){return productionEndDate_;}
  std::string getDataSetSize(){return dataSetSize_;}
  int getNumberOfEvents(){return numberOfEvents_;}
  int getNumberOfJobs(){return numberOfJobs_;}
  std::string getTimeToComplete(){return timeToComplete_;}
  std::string getCrabConfiguration(){return crabConfig_;}
  bool isObsolete(){return obsolete_;}
  std::string getDateOfObsoletion(){return dateOfObsoletion_;}
  std::string getReasonForObsoletion(){return reasonForObsoletion_;}
  bool isDeleted(){return deleted_;}
  
 private:
  std::string infile_;
  std::string outfile_;
  std::string title_;
  std::string priority_;
  std::string dateOfRequest_;
  std::string description_;
  std::string linkToTwiki_;
  std::string productionStartDate_;
  std::string responsible_;
  std::string dataSetPath_;
  std::string productionEndDate_;
  std::string dataSetSize_;
  int numberOfEvents_;
  int numberOfJobs_;    
  std::string timeToComplete_;
  std::string crabConfig_;
  bool obsolete_;
  std::string dateOfObsoletion_;
  std::string reasonForObsoletion_;
  bool deleted_;
};

#endif
