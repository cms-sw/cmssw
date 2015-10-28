#ifndef BTVHLTOfflineSource_H
#define BTVHLTOfflineSource_H
/*
  BTVHLTOffline DQM code
*/  
//
// Originally created by:  Anne-Catherine Le Bihan
//                         June 2015
// Following the structure used in JetMetHLTOfflineSource


// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

class BTVHLTOfflineSource : public DQMEDAnalyzer {
 public:
  explicit BTVHLTOfflineSource(const edm::ParameterSet&);
  ~BTVHLTOfflineSource();

 private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const & run, edm::EventSetup const & c) override;
  virtual void dqmBeginRun(edm::Run const& run, edm::EventSetup const& c) override;

  bool verbose_;
  std::string dirname_;
  std::string processname_;
  std::string pathname_;
  std::string filtername_; 
  
  std::vector<std::pair<std::string, std::string> > custompathnamepairs_;
  
  edm::InputTag triggerSummaryLabel_;
  edm::InputTag triggerResultsLabel_;

  edm::EDGetTokenT<reco::JetTagCollection> offlineCSVTokenPF_;
  edm::EDGetTokenT<reco::JetTagCollection> offlineCSVTokenCalo_;

  edm::EDGetTokenT<std::vector<reco::Vertex> > hltFastPVToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > hltPFPVToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > hltCaloPVToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > offlinePVToken_;
  
  edm::EDGetTokenT <edm::TriggerResults> triggerResultsToken;
  edm::EDGetTokenT <edm::TriggerResults> triggerResultsFUToken;
  edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryToken;
  edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryFUToken;
  
  edm::EDGetTokenT<reco::JetTagCollection> csvCaloTagsToken_;
  edm::EDGetTokenT<reco::JetTagCollection> csvPfTagsToken_;
  edm::Handle<reco::JetTagCollection> csvCaloTags;
  edm::Handle<reco::JetTagCollection> csvPfTags;
  
  HLTConfigProvider hltConfig_;
  edm::Handle<edm::TriggerResults> triggerResults_;
  edm::TriggerNames triggerNames_;
  edm::Handle<trigger::TriggerEvent> triggerObj_;
  
  class PathInfo {
    PathInfo():
      prescaleUsed_(-1),
      pathName_("unset"),
      filterName_("unset"),
      processName_("unset"),
      objectType_(-1),
      triggerType_("unset")
      {};
  
  public:
    void setHistos(
		   MonitorElement* const CSV, MonitorElement* const Pt, MonitorElement* const Eta,
		   MonitorElement* const CSV_RECOvsHLT, MonitorElement* const PVz, MonitorElement* const fastPVz,
		   MonitorElement* const PVz_HLTMinusRECO, MonitorElement* const fastPVz_HLTMinusRECO
		   )
    { CSV_ = CSV; Pt_ = Pt; Eta_ = Eta; CSV_RECOvsHLT_ = CSV_RECOvsHLT; PVz_ = PVz; fastPVz_ = fastPVz;
      PVz_HLTMinusRECO_ = PVz_HLTMinusRECO; fastPVz_HLTMinusRECO_ = fastPVz_HLTMinusRECO;
    };


    ~PathInfo() {};
    PathInfo(int prescaleUsed,
	     std::string pathName,
	     std::string filterName,
	     std::string processName,
	     size_t type,
	     std::string triggerType):
      prescaleUsed_(prescaleUsed),
      pathName_(pathName),
      filterName_(filterName),
      processName_(processName),
      objectType_(type),
      triggerType_(triggerType){};

      MonitorElement * getMEhisto_CSV()               { return CSV_;}
      MonitorElement * getMEhisto_Pt()                { return Pt_; }
      MonitorElement * getMEhisto_Eta()               { return Eta_;}
      MonitorElement * getMEhisto_CSV_RECOvsHLT()     { return CSV_RECOvsHLT_;}
      MonitorElement * getMEhisto_PVz()               { return PVz_;}
      MonitorElement * getMEhisto_fastPVz()           { return fastPVz_;}
      MonitorElement * getMEhisto_PVz_HLTMinusRECO()      { return PVz_HLTMinusRECO_;}
      MonitorElement * getMEhisto_fastPVz_HLTMinusRECO()  { return fastPVz_HLTMinusRECO_;}
     
      const std::string getLabel(void ) const {
	return filterName_;
      }
      void setLabel(std::string labelName){
	filterName_ = labelName;
	return;
      }
      const std::string getPath(void ) const {
	return pathName_;
      }
      const int getprescaleUsed(void) const {
	return prescaleUsed_;
      }
      const std::string getProcess(void ) const {
	return processName_;
      }
      const int getObjectType(void ) const {
	return objectType_;
      }
      const std::string getTriggerType(void ) const {
	return triggerType_;
      }
      const edm::InputTag getTag(void) const{
	edm::InputTag tagName(filterName_,"",processName_);
	return tagName;
      }
      bool operator==(const std::string v)
      {
	return v==pathName_;
      }

  private:
  
      int prescaleUsed_;
      std::string pathName_;
      std::string filterName_;
      std::string processName_;
      int objectType_;
      std::string triggerType_;

      MonitorElement*  CSV_;
      MonitorElement*  Pt_;
      MonitorElement*  Eta_;
      MonitorElement*  CSV_RECOvsHLT_;   
      MonitorElement*  PVz_;
      MonitorElement*  fastPVz_;
      MonitorElement*  PVz_HLTMinusRECO_;
      MonitorElement*  fastPVz_HLTMinusRECO_;
     
      };
 
  class PathInfoCollection: public std::vector<PathInfo> {
  public:
    PathInfoCollection(): std::vector<PathInfo>()
      {};
      std::vector<PathInfo>::iterator find(std::string pathName) {
        return std::find(begin(), end(), pathName);
      }
  };
  PathInfoCollection hltPathsAll_;

 };
#endif
