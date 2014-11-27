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
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DQMOffline/Trigger/interface/FSQDiJetAve.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include <DataFormats/TrackReco/interface/Track.h>
#include <DataFormats/EgammaCandidates/interface/Photon.h>
#include <DataFormats/MuonReco/interface/Muon.h>

#include <boost/algorithm/string.hpp>


using namespace edm;
using namespace std;


namespace FSQ {

//################################################################################################
//
// Base Handler class
//
//################################################################################################
class BaseHandler {
    public:
        BaseHandler();
        ~BaseHandler(){
            delete m_expression;
        }
        BaseHandler(const edm::ParameterSet& iConfig,  triggerExpression::Data & eventCache):
              m_expression(triggerExpression::parse( iConfig.getParameter<std::string>("triggerSelection")))
         {
              // extract list of used paths
              std::vector<std::string> strs;
              std::string triggerSelection = iConfig.getParameter<std::string>("triggerSelection");
              boost::split(strs, triggerSelection, boost::is_any_of("\t ,`!@#$%^&*()~/\\"));
              for (unsigned int iToken = 0; iToken < strs.size();++iToken ){
                    if (strs.at(iToken).find("HLT_")==0){
                        m_usedPaths.insert(strs.at(iToken));
                    }
              }

              m_eventCache = &eventCache;
              std::string pathPartialName  = iConfig.getParameter<std::string>("partialPathName");
              m_dirname = iConfig.getUntrackedParameter("mainDQMDirname",std::string("HLT/FSQ/"))+pathPartialName + "/";

        };
        virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
                             const HLTConfigProvider&  hltConfig,
                             const trigger::TriggerEvent& trgEvent,
                             const edm::TriggerResults & triggerResults, 
                             const edm::TriggerNames  & triggerNames,
                             float weight) = 0;
        virtual void book(DQMStore::IBooker & booker) = 0;

        triggerExpression::Evaluator * m_expression;
        triggerExpression::Data * m_eventCache;

        std::string m_dirname;

        std::map<std::string,  MonitorElement*> m_histos;
         std::set<std::string> m_usedPaths;

        
};
//################################################################################################
//
// Handle objects saved into hlt event by hlt filters
//
//################################################################################################
//
template <class TInputCandidateType, class TOutputCandidateType>
class HandlerTemplate: public BaseHandler {
    private:
        std::string m_dqmhistolabel;
        std::string m_pathPartialName; //#("HLT_DiPFJetAve30_HFJEC_");
        std::string m_filterPartialName; //#("ForHFJECBase"); // Calo jet preFilter

        int m_combinedObjectDimension;

        StringCutObjectSelector<TInputCandidateType>  m_singleObjectSelection;
        StringCutObjectSelector<std::vector<TOutputCandidateType> >  m_combinedObjectSelection;
        StringObjectFunction<std::vector<TOutputCandidateType> >     m_combinedObjectSortFunction;
        // TODO: auto ptr
        std::map<std::string, std::shared_ptr<StringObjectFunction<std::vector<TOutputCandidateType> > > > m_plotters;


        std::vector< edm::ParameterSet > m_drawables;
        bool m_isSetup;
        edm::InputTag m_input;

    public:
        HandlerTemplate(const edm::ParameterSet& iConfig, triggerExpression::Data & eventCache):
            BaseHandler(iConfig, eventCache),
            m_singleObjectSelection(iConfig.getParameter<std::string>("singleObjectsPreselection")),
            m_combinedObjectSelection(iConfig.getParameter<std::string>("combinedObjectSelection")),
            m_combinedObjectSortFunction(iConfig.getParameter<std::string>("combinedObjectSortCriteria"))
        {
             std::string type = iConfig.getParameter<std::string>("handlerType");
             if (type != "FromHLT") {
                m_input = iConfig.getParameter<edm::InputTag>("inputCol");
             }

             m_dqmhistolabel = iConfig.getParameter<std::string>("dqmhistolabel");
             m_filterPartialName = iConfig.getParameter<std::string>("partialFilterName"); // std::string find is used to match filter
             m_pathPartialName  = iConfig.getParameter<std::string>("partialPathName");
             m_combinedObjectDimension = iConfig.getParameter<int>("combinedObjectDimension");
             m_drawables = iConfig.getParameter<  std::vector< edm::ParameterSet > >("drawables");
             m_isSetup = false;
        }

        void book(DQMStore::IBooker & booker){
            if(!m_isSetup){
                booker.setCurrentFolder(m_dirname);
                m_isSetup = true;
                for (unsigned int i = 0; i < m_drawables.size(); ++i){
                    std::string histoName = m_dqmhistolabel + "_" +m_drawables.at(i).getParameter<std::string>("name");
                    std::string expression = m_drawables.at(i).getParameter<std::string>("expression");
                    int bins =  m_drawables.at(i).getParameter<int>("bins");
                    double rangeLow  =  m_drawables.at(i).getParameter<double>("min");
                    double rangeHigh =  m_drawables.at(i).getParameter<double>("max");

                    m_histos[histoName] =  booker.book1D(histoName, histoName, bins, rangeLow, rangeHigh);
                    StringObjectFunction<std::vector<TOutputCandidateType> > * func 
                            = new StringObjectFunction<std::vector<TOutputCandidateType> >(expression);
                    m_plotters[histoName] =  std::shared_ptr<StringObjectFunction<std::vector<TOutputCandidateType> > >(func);

                }   
            }
        }

        // Notes:
        //  - FIXME this function should take only event/ event setup (?)
        //  - FIXME responsibility to apply preselection should be elsewhere
        //          hard to fix, since we dont want to copy all objects due to
        //          performance reasons
        //
        //           implementation below working when in/out types are equal
        //           in other cases you must provide specialized version (see below)
        void getFilteredCands(TInputCandidateType *, std::vector<TOutputCandidateType> & cands,
                     const edm::Event& iEvent,
                     const edm::EventSetup& iSetup,
                     const HLTConfigProvider&  hltConfig,
                     const trigger::TriggerEvent& trgEvent)
        {
               Handle<std::vector<TInputCandidateType> > hIn;
               iEvent.getByLabel(InputTag(m_input), hIn);
               if(!hIn.isValid()) {
                  edm::LogError("FSQDiJetAve") << "product not found: "<<  m_input.encode();
                  return;  
               }


               for (unsigned int i = 0; i<hIn->size(); ++i) {
                    bool preselection = m_singleObjectSelection(hIn->at(i));
                    if (preselection){
                        cands.push_back(hIn->at(i));
                    }
               }
        }

        std::vector<std::string > findPathAndFilter(const HLTConfigProvider&  hltConfig){
            std::vector<std::string> ret(2,"");
            std::string filterFullName = "";
            std::string pathFullName = "";
            std::vector<std::string> filtersForThisPath;
            //int pathIndex = -1;
            int numPathMatches = 0;
            int numFilterMatches = 0;
            for (unsigned int i = 0; i < hltConfig.size(); ++i) {
                if (hltConfig.triggerName(i).find(m_pathPartialName) == std::string::npos) continue;
                pathFullName = hltConfig.triggerName(i);
                //pathIndex = i;
                ++numPathMatches;
                std::vector<std::string > moduleLabels = hltConfig.moduleLabels(i);
                for (unsigned int iMod = 0; iMod <moduleLabels.size(); ++iMod){
                    if ("EDFilter" ==  hltConfig.moduleEDMType(moduleLabels.at(iMod))) {
                        filtersForThisPath.push_back(moduleLabels.at(iMod));
                        if ( moduleLabels.at(iMod).find(m_filterPartialName)!= std::string::npos  ){
                            filterFullName = moduleLabels.at(iMod);
                            ++numFilterMatches;
                        }
                    }
                }
            }

            // LogWarning or LogError?
            if (numPathMatches != 1) {
                  edm::LogError("FSQDiJetAve") << "Problem: found " << numPathMatches
                    << " paths matching " << m_pathPartialName << std::endl;
                  return ret;   
            }
            ret[0] = pathFullName;
            if (numFilterMatches != 1) {
                  edm::LogError("FSQDiJetAve") << "Problem: found " << numFilterMatches
                    << " filter matching " << m_filterPartialName
                    << " in path "<< m_pathPartialName << std::endl;
                  return ret;
            }
            ret[1] = filterFullName;
            return ret;
        }


        // xxx
        void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup,
                     const HLTConfigProvider&  hltConfig,
                     const trigger::TriggerEvent& trgEvent,
                     const edm::TriggerResults & triggerResults, 
                     const edm::TriggerNames  & triggerNames,
                     float weight)
        {
            unsigned int found = 0;
            for (unsigned int i = 0; i<triggerNames.size(); ++i){
                std::set<std::string>::iterator itUsedPaths = m_usedPaths.begin();
                for(; itUsedPaths != m_usedPaths.end(); ++itUsedPaths){ 
                    if (triggerNames.triggerName(i).find(*itUsedPaths)!= std::string::npos ){
                        ++found;
                        break;
                    }
                }

                if (found == m_usedPaths.size()) break;
            }
            if (found != m_usedPaths.size()){
                    edm::LogInfo("FSQDiJetAve") << "One of requested paths not found, skipping event";
                    return;
            }

            if (m_eventCache->configurationUpdated()) {
                m_expression->init(*m_eventCache);
            }
            if (not (*m_expression)(*m_eventCache)) return;


            /*
            std::vector<std::string> pathAndFilter = findPathAndFilter(hltConfig);

            std::string pathFullName = pathAndFilter[0];
            if (pathFullName == "") {
                return;
            }
            unsigned indexNum = triggerNames.triggerIndex(pathFullName);
            if(indexNum >= triggerNames.size()){
                  edm::LogError("FSQDiJetAve") << "Problem determining trigger index for " << pathFullName << " " << m_pathPartialName;
            }
            if (!triggerResults.accept(indexNum)) return;*/

            std::vector<TOutputCandidateType> cands;
            getFilteredCands((TInputCandidateType *)0, cands, iEvent, iSetup, hltConfig, trgEvent);

            if (cands.size()==0) return;

            std::vector<TOutputCandidateType> bestCombinationFromCands = getBestCombination(cands);
            if (bestCombinationFromCands.size()==0) return;

            // plot 
            std::map<std::string,  MonitorElement*>::iterator it, itE;
            it = m_histos.begin();
            itE = m_histos.end();
            for (;it!=itE;++it){
                float val = (*m_plotters[it->first])(bestCombinationFromCands);
                it->second->Fill(val, weight);
            }

        }


        std::vector<TOutputCandidateType> getBestCombination(std::vector<TOutputCandidateType> & cands ){
            int columnSize = cands.size();
            std::vector<int> currentCombination(m_combinedObjectDimension, 0);
            std::vector<int> bestCombination(m_combinedObjectDimension, -1);

            int maxCombinations = 1;
            int cnt = 0;
            while (cnt < m_combinedObjectDimension){
                cnt += 1;
                maxCombinations *= columnSize;
            }

            cnt = 0;
            float bestCombinedCandVal = -1;
            while ( cnt < maxCombinations){
                cnt += 1;

                // 1. Check if current combination contains duplicates
                std::vector<int> currentCombinationCopy(currentCombination);
                std::vector<int>::iterator it;
                std::sort(currentCombinationCopy.begin(), currentCombinationCopy.end());
                it = std::unique(currentCombinationCopy.begin(), currentCombinationCopy.end());
                currentCombinationCopy.resize( std::distance(currentCombinationCopy.begin(),it) );
                bool duplicatesPresent = currentCombination.size() != currentCombinationCopy.size();


                // 2. If no duplicates found - 
                //          - check if current combination passes the cut
                //          - rank current combination
                if (!duplicatesPresent) { // no duplicates, we can consider this combined object
                    /*
                    std::cout << cnt << " " << duplicatesPresent << " ";
                    for (int i = 0; i< dimension; ++i){
                        std::cout << cands.at(currentCombination.at(i));
                    }
                    std::cout << std::endl;
                    // */
                    std::vector<TOutputCandidateType > currentCombinationFromCands;
                    for (int i = 0; i<m_combinedObjectDimension;++i){
                        currentCombinationFromCands.push_back( cands.at(currentCombination.at(i)));
                    }
                    bool isOK = m_combinedObjectSelection(currentCombinationFromCands);
                    if (isOK){
                        float curVal = m_combinedObjectSortFunction(currentCombinationFromCands);
                        // FIXME
                        if (curVal < 0) {
                            edm::LogError("FSQDiJetAve") << "Problem: ranking function returned negative value: " << curVal << std::endl;
                        } else if (curVal > bestCombinedCandVal){
                            //std::cout << curVal << " " << bestCombinedCandVal << std::endl;
                            bestCombinedCandVal = curVal;
                            bestCombination = currentCombination;
                        }
                    }
                }

                // 3. Prepare next combination to test
                //    note to future self: less error prone method with modulo
                currentCombination.at(m_combinedObjectDimension-1)+=1; // increase last number
                int carry = 0;
                for (int i = m_combinedObjectDimension-1; i>=0; --i){  // iterate over all numbers, check if we are out of range
                    currentCombination.at(i)+= carry;
                    carry = 0;
                    if (currentCombination.at(i)>=columnSize){
                        carry = 1;
                        currentCombination.at(i) = 0;
                    }
                }
            } // combinations loop ends

            //XXX
            std::vector<TOutputCandidateType > bestCombinationFromCands;
            if (bestCombination.size()!=0 && bestCombination.at(0)>=0){
                for (int i = 0; i<m_combinedObjectDimension;++i){
                          bestCombinationFromCands.push_back( cands.at(bestCombination.at(i)));
                }
            }
            return bestCombinationFromCands;
        }

};
//#############################################################################
//
// Read any object inheriting from reco::Candidate. Save p4
//
//#############################################################################
template<>
void HandlerTemplate<reco::Candidate::LorentzVector, reco::Candidate::LorentzVector>::getFilteredCands(
             reco::Candidate::LorentzVector *, // pass a dummy pointer, makes possible to select correct getFilteredCands
             std::vector<reco::Candidate::LorentzVector> & cands, // output collection
             const edm::Event& iEvent,  
             const edm::EventSetup& iSetup,
             const HLTConfigProvider&  hltConfig,
             const trigger::TriggerEvent& trgEvent)
{  
   Handle<View<reco::Candidate> > hIn;
   iEvent.getByLabel(InputTag(m_input), hIn);
   if(!hIn.isValid()) {
      edm::LogError("FSQDiJetAve") << "product not found: "<<  m_input.encode();
      return;
   }
   for (unsigned int i = 0; i<hIn->size(); ++i) {
        bool preselection = m_singleObjectSelection(hIn->at(i).p4());
        if (preselection){
            cands.push_back(hIn->at(i).p4());
        }
   }
}
//#############################################################################
//
// Count any object inheriting from reco::Track. Save into std::vector<int>
// note: this is similar to recoCand counter (code duplication is hard to 
//       avoid in this case)
//
//#############################################################################
template<>
void HandlerTemplate<reco::Track, int >::getFilteredCands(
             reco::Track *, // pass a dummy pointer, makes possible to select correct getFilteredCands
             std::vector<int > & cands, // output collection
             const edm::Event& iEvent,  
             const edm::EventSetup& iSetup,
             const HLTConfigProvider&  hltConfig,
             const trigger::TriggerEvent& trgEvent)
{  
   cands.clear();
   cands.push_back(0);

   Handle<std::vector<reco::Track > > hIn;
   iEvent.getByLabel(InputTag(m_input), hIn);
   if(!hIn.isValid()) {
      edm::LogError("FSQDiJetAve") << "product not found: "<<  m_input.encode();
      return;
   }
   for (unsigned int i = 0; i<hIn->size(); ++i) {
        bool preselection = m_singleObjectSelection(hIn->at(i));
        if (preselection){
            cands.at(0)+=1;
        }
   }
}
//#############################################################################
//
// Count any object inheriting from reco::Candidate. Save into std::vector<int>
//
//#############################################################################
template<>
void HandlerTemplate<reco::Candidate::LorentzVector, int >::getFilteredCands(
             reco::Candidate::LorentzVector *, // pass a dummy pointer, makes possible to select correct getFilteredCands
             std::vector<int > & cands, // output collection
             const edm::Event& iEvent,  
             const edm::EventSetup& iSetup,
             const HLTConfigProvider&  hltConfig,
             const trigger::TriggerEvent& trgEvent)
{  
   cands.clear();
   cands.push_back(0);

   Handle<View<reco::Candidate> > hIn;
   iEvent.getByLabel(InputTag(m_input), hIn);
   if(!hIn.isValid()) {
      edm::LogError("FSQDiJetAve") << "product not found: "<<  m_input.encode();
      return;
   }
   for (unsigned int i = 0; i<hIn->size(); ++i) {
        bool preselection = m_singleObjectSelection(hIn->at(i).p4());
        if (preselection){
            cands.at(0)+=1;
        }
   }
}
//#############################################################################
//
// Read and save trigger::TriggerObject from triggerEvent
//
//#############################################################################
template<>
void HandlerTemplate<trigger::TriggerObject, trigger::TriggerObject>::getFilteredCands(
             trigger::TriggerObject *, 
             std::vector<trigger::TriggerObject> &cands, 
             const edm::Event& iEvent,  
             const edm::EventSetup& iSetup,
             const HLTConfigProvider&  hltConfig,
             const trigger::TriggerEvent& trgEvent)
{
    
    // 1. Find matching path. Inside matchin path find matching filter
    std::string filterFullName = findPathAndFilter(hltConfig)[1];
    if (filterFullName == "") {
        return;
    }

    // 2. Fetch HLT objects saved by selected filter. Save those fullfilling preselection
    //      objects are saved in cands variable
    std::string process = trgEvent.usedProcessName(); // broken?
    edm::InputTag hltTag(filterFullName ,"", process);
    
    const int hltIndex = trgEvent.filterIndex(hltTag);
    if ( hltIndex >= trgEvent.sizeFilters() ) {
      edm::LogInfo("FSQDiJetAve") << "Cannot determine hlt index for |" << filterFullName << "|" << process;
      return;
    }

    const trigger::TriggerObjectCollection & toc(trgEvent.getObjects());
    const trigger::Keys & khlt = trgEvent.filterKeys(hltIndex);

    trigger::Keys::const_iterator kj = khlt.begin();

    for(;kj != khlt.end(); ++kj){
        bool preselection = m_singleObjectSelection(toc[*kj]);
        if (preselection){
            cands.push_back( toc[*kj]);
        }
    }

}



typedef HandlerTemplate<trigger::TriggerObject, trigger::TriggerObject> HLTHandler;
typedef HandlerTemplate<reco::Candidate::LorentzVector, reco::Candidate::LorentzVector> RecoCandidateHandler;// in fact reco::Candidate, reco::Candidate::LorentzVector
typedef HandlerTemplate<reco::PFJet, reco::PFJet> RecoPFJetHandler;
typedef HandlerTemplate<reco::Track, reco::Track> RecoTrackHandler;
typedef HandlerTemplate<reco::Photon, reco::Photon> RecoPhotonHandler;
typedef HandlerTemplate<reco::Muon, reco::Muon> RecoMuonHandler;

typedef HandlerTemplate<reco::Candidate::LorentzVector, int > RecoCandidateCounter;
typedef HandlerTemplate<reco::Track, int > RecoTrackCounter;

// muon, genPart
}

//################################################################################################
//
// Plugin functions
//
//################################################################################################
FSQDiJetAve::FSQDiJetAve(const edm::ParameterSet& iConfig):
  m_eventCache(iConfig.getParameterSet("triggerConfiguration") , consumesCollector()),
  m_isSetup(false)
{
  m_useGenWeight = iConfig.getParameter<bool>("useGenWeight");

  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
  triggerSummaryToken  = consumes <trigger::TriggerEvent> (triggerSummaryLabel_);
  triggerResultsToken  = consumes <edm::TriggerResults>   (triggerResultsLabel_);

  triggerSummaryFUToken= consumes <trigger::TriggerEvent> (edm::InputTag(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(),std::string("FU")));
  triggerResultsFUToken= consumes <edm::TriggerResults>   (edm::InputTag(triggerResultsLabel_.label(),triggerResultsLabel_.instance(),std::string("FU")));

  std::vector< edm::ParameterSet > todo  = iConfig.getParameter<  std::vector< edm::ParameterSet > >("todo");
  for (unsigned int i = 0; i < todo.size(); ++i) {
        edm::ParameterSet pset = todo.at(i);
        std::string type = pset.getParameter<std::string>("handlerType");
        if (type == "FromHLT") {
            m_handlers.push_back(std::shared_ptr<FSQ::HLTHandler>(new FSQ::HLTHandler(pset, m_eventCache)));
        }
        else if (type == "RecoCandidateCounter") {
            m_handlers.push_back(std::shared_ptr<FSQ::RecoCandidateCounter>(new FSQ::RecoCandidateCounter(pset, m_eventCache)));
        }
        else if (type == "RecoTrackCounter") {
            m_handlers.push_back(std::shared_ptr<FSQ::RecoTrackCounter>(new FSQ::RecoTrackCounter(pset, m_eventCache)));
        }
        else if (type == "FromRecoCandidate") {
            m_handlers.push_back(std::shared_ptr<FSQ::RecoCandidateHandler>(new FSQ::RecoCandidateHandler(pset, m_eventCache)));
        }
        else if (type == "RecoPFJet") {
            m_handlers.push_back(std::shared_ptr<FSQ::RecoPFJetHandler>(new FSQ::RecoPFJetHandler(pset, m_eventCache)));
        }
        else if (type == "RecoTrack") {
            m_handlers.push_back(std::shared_ptr<FSQ::RecoTrackHandler>(new FSQ::RecoTrackHandler(pset, m_eventCache)));
        } 
        else if (type == "RecoPhoton") {
            m_handlers.push_back(std::shared_ptr<FSQ::RecoPhotonHandler>(new FSQ::RecoPhotonHandler(pset, m_eventCache)));
        } 
        else if (type == "RecoMuon") {
            m_handlers.push_back(std::shared_ptr<FSQ::RecoMuonHandler>(new FSQ::RecoMuonHandler(pset, m_eventCache)));
        } 
        else {
            throw cms::Exception("FSQ DQM handler not know: "+ type);
        }
  }
}


FSQDiJetAve::~FSQDiJetAve()
{}

void
FSQDiJetAve::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   if (not m_eventCache.setEvent(iEvent, iSetup)){
      edm::LogError("FSQDiJetAve") << "Could not setup the filter";
   }

  //---------- triggerResults ----------
  iEvent.getByToken(triggerResultsToken, m_triggerResults);
  if(!m_triggerResults.isValid()) {
    iEvent.getByToken(triggerResultsFUToken, m_triggerResults);
    if(!m_triggerResults.isValid()) {
      edm::LogError("FSQDiJetAve") << "TriggerResults not valid, skippng event";
      return;
    }
  }
  
  //---------- triggerResults ----------
  if(&m_triggerResults) {  
    m_triggerNames = iEvent.triggerNames(*m_triggerResults);
  } 
  else {
    edm::LogError("FSQDiJetAve") << "TriggerResults not found";
    return;
  } 
  
  //---------- triggerSummary ----------
  iEvent.getByToken(triggerSummaryToken, m_trgEvent);
  if(!m_trgEvent.isValid()) {
    iEvent.getByToken(triggerSummaryFUToken, m_trgEvent);
    if(!m_trgEvent.isValid()) {
      edm::LogInfo("FSQDiJetAve") << "TriggerEvent not found, ";
      return;
    }
  } 
  

  float weight = 1.;  
  if (m_useGenWeight){
    edm::Handle<GenEventInfoProduct> hGW;
    iEvent.getByLabel(edm::InputTag("generator"), hGW);
    weight = hGW->weight();
  }

  for (unsigned int i = 0; i < m_handlers.size(); ++i) {
        m_handlers.at(i)->analyze(iEvent, iSetup, m_hltConfig, *m_trgEvent.product(), *m_triggerResults.product(), m_triggerNames, weight);
  }



}

// ------------ method called when starting to processes a run  ------------
//*
void 
FSQDiJetAve::dqmBeginRun(edm::Run const& run, edm::EventSetup const& c)
{

    bool changed(true);
    std::string processName = triggerResultsLabel_.process();
    if (m_hltConfig.init(run, c, processName, changed)) {
        LogDebug("FSQDiJetAve") << "HLTConfigProvider failed to initialize.";
    }



}
void FSQDiJetAve::bookHistograms(DQMStore::IBooker & booker, edm::Run const & run, edm::EventSetup const & c){
    for (unsigned int i = 0; i < m_handlers.size(); ++i) {
        m_handlers.at(i)->book(booker);
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
{}
// */

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
