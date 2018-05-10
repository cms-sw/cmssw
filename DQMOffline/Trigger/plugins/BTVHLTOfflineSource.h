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
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

// user include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class BTVHLTOfflineSource : public DQMEDAnalyzer {
 public:
  explicit BTVHLTOfflineSource(const edm::ParameterSet&);
  ~BTVHLTOfflineSource() override;

 private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const & run, edm::EventSetup const & c) override;
  void dqmBeginRun(edm::Run const& run, edm::EventSetup const& c) override;

  bool verbose_;
  std::string dirname_;
  std::string processname_;
  std::string pathname_;
  std::string filtername_;

  std::vector<std::pair<std::string, std::string> > custompathnamepairs_;

  edm::InputTag triggerSummaryLabel_;
  edm::InputTag triggerResultsLabel_;

  float turnon_threshold_loose_;
  float turnon_threshold_medium_;
  float turnon_threshold_tight_;

  edm::EDGetTokenT<reco::JetTagCollection> offlineDiscrTokenb_;
  edm::EDGetTokenT<reco::JetTagCollection> offlineDiscrTokenbb_;

  edm::EDGetTokenT<std::vector<reco::Vertex> > hltFastPVToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > hltPFPVToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > hltCaloPVToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > offlinePVToken_;

  edm::EDGetTokenT <edm::TriggerResults> triggerResultsToken;
  edm::EDGetTokenT <edm::TriggerResults> triggerResultsFUToken;
  edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryToken;
  edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryFUToken;

  edm::EDGetTokenT<std::vector<reco::ShallowTagInfo> > shallowTagInfosTokenCalo_;
  edm::EDGetTokenT<std::vector<reco::ShallowTagInfo> > shallowTagInfosTokenPf_;

  edm::Handle<std::vector<reco::ShallowTagInfo> > shallowTagInfosCalo;
  edm::Handle<std::vector<reco::ShallowTagInfo> > shallowTagInfosPf;

  // edm::EDGetTokenT<std::vector<reco::TemplatedSecondaryVertexTagInfo<reco::IPTagInfo<edm::RefVector<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,reco::JTATagInfo>,reco::Vertex> > >
  //      caloTagInfosToken_;
  // edm::EDGetTokenT<std::vector<reco::TemplatedSecondaryVertexTagInfo<reco::IPTagInfo<edm::RefVector<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,reco::JTATagInfo>,reco::Vertex> > >
  //      pfTagInfosToken_;

  edm::Handle<std::vector<reco::TemplatedSecondaryVertexTagInfo<reco::IPTagInfo<edm::RefVector<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,reco::JTATagInfo>,reco::Vertex> > >
       caloTagInfos;
  edm::Handle<std::vector<reco::TemplatedSecondaryVertexTagInfo<reco::IPTagInfo<edm::RefVector<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> >,reco::JTATagInfo>,reco::Vertex> > >
       pfTagInfos;

  edm::EDGetTokenT<reco::JetTagCollection> caloTagsToken_;
  edm::EDGetTokenT<reco::JetTagCollection> pfTagsToken_;
  edm::Handle<reco::JetTagCollection> caloTags;
  edm::Handle<reco::JetTagCollection> pfTags;

  HLTConfigProvider hltConfig_;
  edm::Handle<edm::TriggerResults> triggerResults_;
  edm::TriggerNames triggerNames_;
  edm::Handle<trigger::TriggerEvent> triggerObj_;

  class PathInfo : public TriggerDQMBase {
    PathInfo():
      prescaleUsed_(-1),
      pathName_("unset"),
      filterName_("unset"),
      processName_("unset"),
      objectType_(-1),
      triggerType_("unset")
      {};

  public:
    ~PathInfo() = default;;
    PathInfo(int prescaleUsed,
        std::string pathName,
        std::string filterName,
        std::string processName,
        size_t type,
        std::string triggerType):
      prescaleUsed_(prescaleUsed),
      pathName_(std::move(pathName)),
      filterName_(std::move(filterName)),
      processName_(std::move(processName)),
      objectType_(type),
      triggerType_(std::move(triggerType)) {}

    const std::string getLabel( )                const  {return filterName_;}
    void setLabel(std::string labelName)                {filterName_ = std::move(labelName);}
    const std::string getPath( )                 const  {return pathName_;}
    const int getprescaleUsed()                  const  {return prescaleUsed_;}
    const std::string getProcess( )              const  {return processName_;}
    const int getObjectType( )                   const  {return objectType_;}
    const std::string getTriggerType( )          const  {return triggerType_;}
    const edm::InputTag getTag()                 const  {return edm::InputTag(filterName_,"",processName_);}
    const bool operator== (const std::string& v) const  {return v==pathName_;}

    MonitorElement*  Discr = nullptr;
    MonitorElement*  Pt = nullptr;
    MonitorElement*  Eta = nullptr;
    MonitorElement*  Discr_HLTvsRECO = nullptr;
    MonitorElement*  Discr_HLTMinusRECO = nullptr;
    ObjME            Discr_turnon_loose;
    ObjME            Discr_turnon_medium;
    ObjME            Discr_turnon_tight;
    MonitorElement*  PVz = nullptr;
    MonitorElement*  fastPVz = nullptr;
    MonitorElement*  PVz_HLTMinusRECO = nullptr;
    MonitorElement*  fastPVz_HLTMinusRECO = nullptr;
    MonitorElement*  n_vtx = nullptr;
    MonitorElement*  vtx_mass = nullptr;
    MonitorElement*  n_vtx_trks = nullptr;
    MonitorElement*  n_sel_tracks = nullptr;
    MonitorElement*  h_3d_ip_distance = nullptr;
    MonitorElement*  h_3d_ip_error = nullptr;
    MonitorElement*  h_3d_ip_sig = nullptr;
    // MonitorElement*  n_pixel_hits_;
    // MonitorElement*  n_total_hits_;

  private:
    int prescaleUsed_;
    std::string pathName_;
    std::string filterName_;
    std::string processName_;
    int objectType_;
    std::string triggerType_;
  };

  class PathInfoCollection: public std::vector<PathInfo> {
  public:
    PathInfoCollection(): std::vector<PathInfo>()
      {};
      std::vector<PathInfo>::iterator find(const std::string& pathName) {
        return std::find(begin(), end(), pathName);
      }
  };
  PathInfoCollection hltPathsAll_;

 };
#endif
