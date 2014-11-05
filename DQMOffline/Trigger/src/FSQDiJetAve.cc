// -*- C++ -*-
//
// Package:    DQMOffline/FSQDiJetAve
// Class:      FSQDiJetAve
// 
/**\class FSQDiJetAve FSQDiJetAve.cc DQMOffline/FSQDiJetAve/plugins/FSQDiJetAve.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Tomasz Fruboes
//         Created:  Tue, 04 Nov 2014 11:36:27 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMOffline/Trigger/interface/FSQDiJetAve.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"


using namespace edm;
//#using namespace reco;
using namespace std;
//using namespace trigger;


namespace FSQ {

struct HLTConfigDataContainer {
    HLTConfigProvider m_hltConfig;

};

//################################################################################################
//
// Base Handler class
//
//################################################################################################
class BaseHandler {
    public:
        BaseHandler();
        BaseHandler(const edm::ParameterSet& iConfig);
        virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const HLTConfigDataContainer &hc) = 0;
};
//################################################################################################
//
// Handle objects saved into hlt event by hlt filters
//
//################################################################################################
class HLTHandler: public BaseHandler {
    public:
        HLTHandler(const edm::ParameterSet& iConfig):
            m_singleObjectSelection(iConfig.getParameter<std::string>("singleObjectsPreselection")),
            m_combinedObjectSelection(iConfig.getParameter<std::string>("combinedObjectSelection")),
            m_combinedObjectSortFunction(iConfig.getParameter<std::string>("combinedObjectSortCriteria"))
        {
             std::string type = iConfig.getParameter<std::string>("handlerType");
             if (type != "FromHLT") {
                throw cms::Exception("FSQ - HLTHandler: wrong " + type);
             }

             m_dqmhistolabel = iConfig.getParameter<std::string>("dqmhistolabel");
             m_filterPartialName = iConfig.getParameter<std::string>("partialFilterName"); // std::string find is used to match filter
                                                                                           // there should be just one matching filter
                                                                                           //  in path
                                                                                           
             m_pathPartialName  = iConfig.getParameter<std::string>("partialPathName");
             m_combinedObjectDimension = iConfig.getParameter<int>("combinedObjectDimension");
        }


        void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const HLTConfigDataContainer &hc){

            // 1. Find matching path. Inside matchin path find matching filter
            std::string filterFullName = "";
            std::vector<std::string> filtersForThisPath;
            //int pathIndex = -1;
            int numPathMatches = 0;
            int numFilterMatches = 0;
            for (unsigned int i = 0; i < hc.m_hltConfig.size(); ++i) {
                if (hc.m_hltConfig.triggerName(i).find(m_pathPartialName) == std::string::npos) continue;
                //pathIndex = i;
                ++numPathMatches;
                std::vector<std::string > moduleLabels = hc.m_hltConfig.moduleLabels(i);
                for (unsigned int iMod = 0; iMod <moduleLabels.size(); ++iMod){
                    if ("EDFilter" ==  hc.m_hltConfig.moduleEDMType(moduleLabels.at(iMod))) {
                        filtersForThisPath.push_back(moduleLabels.at(iMod));
                        if ( moduleLabels.at(iMod).find(m_filterPartialName)!= std::string::npos  ){
                            filterFullName = moduleLabels.at(iMod);
                            ++numFilterMatches;
                        }
                    }
                }
            }




        }

    private:
        typedef trigger::TriggerObject TCandidateType;

        std::string m_dqmhistolabel;
        std::string m_pathPartialName; //#("HLT_DiPFJetAve30_HFJEC_");
        std::string m_filterPartialName; //#("ForHFJECBase"); // Calo jet preFilter


        int m_combinedObjectDimension;

        StringCutObjectSelector<TCandidateType>  m_singleObjectSelection;
        StringCutObjectSelector<std::vector<TCandidateType> >  m_combinedObjectSelection;
        StringObjectFunction<std::vector<TCandidateType> >     m_combinedObjectSortFunction;
};

}

//################################################################################################
//
// Plugin functions
//
//################################################################################################
FSQDiJetAve::FSQDiJetAve(const edm::ParameterSet& iConfig):
  m_isSetup(false)
{
   //now do what ever initialization is needed
  m_dbe = Service < DQMStore > ().operator->();
  m_dirname = iConfig.getUntrackedParameter("dirname",std::string("HLT/FSQ/DiJETAve/"));
  m_dbe->setCurrentFolder(m_dirname);
  m_useGenWeight = iConfig.getUntrackedParameter("useGenWeight", false);


  processname_         = iConfig.getParameter<std::string>("processname");
  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
  triggerSummaryToken  = consumes <trigger::TriggerEvent> (triggerSummaryLabel_);
  triggerResultsToken  = consumes <edm::TriggerResults>   (triggerResultsLabel_);

  triggerSummaryFUToken= consumes <trigger::TriggerEvent> (edm::InputTag(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(),std::string("FU")));
  triggerResultsFUToken= consumes <edm::TriggerResults>   (edm::InputTag(triggerResultsLabel_.label(),triggerResultsLabel_.instance(),std::string("FU")));

}


FSQDiJetAve::~FSQDiJetAve()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
FSQDiJetAve::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   static int cnt = 0;
   cnt += 1;
   float w = 1./float(cnt);

    std::cout << w << std::endl;
    m_me["test"]->Fill(cnt, w);


  //---------- triggerResults ----------
  iEvent.getByToken(triggerResultsToken, triggerResults_);
  if(!triggerResults_.isValid()) {
    iEvent.getByToken(triggerResultsFUToken,triggerResults_);
    if(!triggerResults_.isValid()) {
      edm::LogInfo("FSQDiJetAve") << "TriggerResults not found, "
	"skipping event";
      return;
    }
  }
  
  //---------- triggerResults ----------
  //int npath;
  if(&triggerResults_) {  
    // Check how many HLT triggers are in triggerResults
    //npath = triggerResults_->size();
    triggerNames_ = iEvent.triggerNames(*triggerResults_);
  } 
  else {
    edm::LogInfo("FSQDiJetAve") << "TriggerResults::HLT not found";
    return;
  } 
  
  //---------- triggerSummary ----------
  iEvent.getByToken(triggerSummaryToken,triggerObj_);
  if(!triggerObj_.isValid()) {
    iEvent.getByToken(triggerSummaryFUToken,triggerObj_);
    if(!triggerObj_.isValid()) {
      edm::LogInfo("FSQDiJetAve") << "TriggerEvent not found, ";
      return;
    }
  } 
  





}


// ------------ method called once each job just before starting event loop  ------------
void 
FSQDiJetAve::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
FSQDiJetAve::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
//*
void 
FSQDiJetAve::beginRun(edm::Run const&, edm::EventSetup const&)
{

    if (!m_isSetup){
      m_me["test"]= m_dbe->book1D("test", "test", 100, 0, 100);
      m_isSetup = true;
    }

}
//*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
FSQDiJetAve::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
FSQDiJetAve::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
FSQDiJetAve::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
FSQDiJetAve::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
//DEFINE_FWK_MODULE(FSQDiJetAve);
