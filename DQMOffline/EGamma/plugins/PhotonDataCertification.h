#ifndef PhotonDataCertification_H
#define PhotonDataCertification_H

// system include files
#include <memory>


#include "CommonTools/UtilAlgos/interface/TFileService.h"
//
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

//root include files
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
//


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//DQM services
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"


// forward declarations
class TFile;
class TH1F;
class TH2F;
class TProfile;
class TTree;

//
// class decleration
//

class PhotonDataCertification : public DQMEDHarvester {

public:
  explicit PhotonDataCertification(const edm::ParameterSet& pset);
  ~PhotonDataCertification();

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

private:

  edm::ParameterSet parameters_;

  bool verbose_;
  MonitorElement* reportSummary_;
  MonitorElement* reportSummaryMap_;
  float invMassZtest(std::string path, TString name, DQMStore::IGetter &);


 // ----------member data ---------------------------
};


#endif
