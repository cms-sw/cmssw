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
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/Candidate/interface/Candidate.h"


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
        BaseHandler(const edm::ParameterSet& iConfig) {
              std::string pathPartialName  = iConfig.getParameter<std::string>("partialPathName");
              m_dirname = iConfig.getUntrackedParameter("mainDQMDirname",std::string("HLT/FSQ/"))+pathPartialName + "/";
              m_dbe = Service < DQMStore > ().operator->();

        };
        virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
                             const HLTConfigProvider&  hltConfig,
                             const trigger::TriggerEvent& trgEvent,
                             float weight) = 0;
        virtual void beginRun() = 0;

        DQMStore * m_dbe;
        std::string m_dirname;

        std::map<std::string,  MonitorElement*> m_histos;

        
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
        //typedef trigger::TriggerObject TInputCandidateType;

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
        HandlerTemplate(const edm::ParameterSet& iConfig):
            BaseHandler(iConfig),
            m_singleObjectSelection(iConfig.getParameter<std::string>("singleObjectsPreselection")),
            m_combinedObjectSelection(iConfig.getParameter<std::string>("combinedObjectSelection")),
            m_combinedObjectSortFunction(iConfig.getParameter<std::string>("combinedObjectSortCriteria"))
        {
             std::string type = iConfig.getParameter<std::string>("handlerType");
             if (type != "FromHLT") {
                m_input = iConfig.getParameter<edm::InputTag>("inputCol");
                //throw cms::Exception("FSQ - HandlerTemplate: wrong " + type);
             }


             m_dqmhistolabel = iConfig.getParameter<std::string>("dqmhistolabel");
             m_filterPartialName = iConfig.getParameter<std::string>("partialFilterName"); // std::string find is used to match filter
                                                                                           // there should be just one matching filter
                                                                                           //  in path
                                                                                           
             m_pathPartialName  = iConfig.getParameter<std::string>("partialPathName");
             m_combinedObjectDimension = iConfig.getParameter<int>("combinedObjectDimension");

             m_drawables = iConfig.getParameter<  std::vector< edm::ParameterSet > >("drawables");
             m_isSetup = false;


        }

        void beginRun(){
            if(!m_isSetup){
                m_dbe->setCurrentFolder(m_dirname);
                m_isSetup = true;
                for (unsigned int i = 0; i < m_drawables.size(); ++i){
                    std::string histoName = m_dqmhistolabel + "_" +m_drawables.at(i).getParameter<std::string>("name");
                    std::string expression = m_drawables.at(i).getParameter<std::string>("expression");
                    int bins =  m_drawables.at(i).getParameter<int>("bins");
                    double rangeLow  =  m_drawables.at(i).getParameter<double>("min");
                    double rangeHigh =  m_drawables.at(i).getParameter<double>("max");

                    m_histos[histoName] =  m_dbe->book1D(histoName, histoName, bins, rangeLow, rangeHigh);
                    StringObjectFunction<std::vector<TInputCandidateType> > * func = new StringObjectFunction<std::vector<TInputCandidateType> >(expression);
                    m_plotters[histoName] =  std::shared_ptr<StringObjectFunction<std::vector<TOutputCandidateType> > >(func);
                }   
            }
        }

        void getFilteredCands(
                     reco::Candidate::LorentzVector *, // pass a dummy pointer, makes possible to select correct getFilteredCands
                     std::vector<reco::Candidate::LorentzVector> & cands, // output collection
                     const edm::Event& iEvent,  
                     const edm::EventSetup& iSetup,
                     const HLTConfigProvider&  hltConfig,
                     const trigger::TriggerEvent& trgEvent)
        {  

           Handle<View<reco::Candidate> > hIn;
           iEvent.getByLabel(InputTag(m_input), hIn);
           for (unsigned int i = 0; i<hIn->size(); ++i) {
                bool preselection = m_singleObjectSelection(hIn->at(i).p4());
                if (preselection){
                    cands.push_back(hIn->at(i).p4());
                }
           }

        }

        // Notes:
        //  - FIXME this function should take only event/ event setup
        //  - FIXME responsibility to apply preselection should be elsewhere
        //          hard to fix, since we dont want to copy all objects due to
        //          performance reasons
        void getFilteredCands(
                     trigger::TriggerObject *, // input object type
                     std::vector<trigger::TriggerObject> &cands, // output collection
                     const edm::Event& iEvent,  
                     const edm::EventSetup& iSetup,
                     const HLTConfigProvider&  hltConfig,
                     const trigger::TriggerEvent& trgEvent)
        {
            
            // 1. Find matching path. Inside matchin path find matching filter
            std::string filterFullName = "";
            std::vector<std::string> filtersForThisPath;
            //int pathIndex = -1;
            int numPathMatches = 0;
            int numFilterMatches = 0;
            for (unsigned int i = 0; i < hltConfig.size(); ++i) {
                if (hltConfig.triggerName(i).find(m_pathPartialName) == std::string::npos) continue;
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
                  return;   
            }
            if (numFilterMatches != 1) {
                  edm::LogError("FSQDiJetAve") << "Problem: found " << numFilterMatches
                    << " filter matching " << m_filterPartialName
                    << " in path "<< m_pathPartialName << std::endl;
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

        // xxx
        void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup,
                     const HLTConfigProvider&  hltConfig,
                     const trigger::TriggerEvent& trgEvent,
                     float weight)
        {


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

            std::vector<TInputCandidateType > bestCombinationFromCands;
            if (bestCombination.size()!=0 && bestCombination.at(0)>=0){
                for (int i = 0; i<m_combinedObjectDimension;++i){
                          bestCombinationFromCands.push_back( cands.at(bestCombination.at(i)));
                }
            }
            return bestCombinationFromCands;
        }

};
typedef HandlerTemplate<trigger::TriggerObject, trigger::TriggerObject> HLTHandler;
typedef HandlerTemplate<reco::Candidate::LorentzVector, reco::Candidate::LorentzVector> RecoCandidateHandler;// in fact reco::Candidate, reco::Candidate::LorentzVector

}

//################################################################################################
//
// Plugin functions
//
//################################################################################################
FSQDiJetAve::FSQDiJetAve(const edm::ParameterSet& iConfig):
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
            m_handlers.push_back(std::shared_ptr<FSQ::HLTHandler>(new FSQ::HLTHandler(pset)));
        }
        else if (type == "FromRecoCandidate") {
            m_handlers.push_back(std::shared_ptr<FSQ::RecoCandidateHandler>(new FSQ::RecoCandidateHandler(pset)));
        } else {
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

  //---------- triggerResults ----------
  iEvent.getByToken(triggerResultsToken, triggerResults_);
  if(!triggerResults_.isValid()) {
    iEvent.getByToken(triggerResultsFUToken,triggerResults_);
    if(!triggerResults_.isValid()) {
      edm::LogInfo("FSQDiJetAve") << "TriggerResults not found, skipping event";
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
        m_handlers.at(i)->analyze(iEvent, iSetup, m_hltConfig, *m_trgEvent.product(), weight);
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
FSQDiJetAve::beginRun(edm::Run const& run, edm::EventSetup const& c)
{

    bool changed(true);
    if (m_hltConfig.init(run, c, "TTT", changed)) {
        LogDebug("FSQDiJetAve") << "HLTConfigProvider failed to initialize.";
    }


    for (unsigned int i = 0; i < m_handlers.size(); ++i) {
        m_handlers.at(i)->beginRun();
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
