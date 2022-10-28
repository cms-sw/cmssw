// -*- C++ -*-
//
// Package:    DQMOffline/FSQDiJetAve
// Class:      FSQDiJetAve
//
/**\class FSQDiJetAve FSQDiJetAve.cc DQMOffline/FSQDiJetAve/plugins/FSQDiJetAve.cc

 Description: DQM source for FSQ triggers

 Implementation:
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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DQMOffline/Trigger/interface/FSQDiJetAve.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include <DataFormats/TrackReco/interface/Track.h>
#include <DataFormats/EgammaCandidates/interface/Photon.h>
#include <DataFormats/MuonReco/interface/Muon.h>
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <boost/algorithm/string.hpp>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

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
    typedef dqm::legacy::MonitorElement MonitorElement;
    typedef dqm::legacy::DQMStore DQMStore;

    BaseHandler();
    virtual ~BaseHandler() = default;
    BaseHandler(const edm::ParameterSet& iConfig, triggerExpression::Data& eventCache)
        : m_expression(triggerExpression::parse(iConfig.getParameter<std::string>("triggerSelection"))) {
      // extract list of used paths
      std::vector<std::string> strs;
      std::string triggerSelection = iConfig.getParameter<std::string>("triggerSelection");
      boost::split(strs, triggerSelection, boost::is_any_of("\t ,`!@#$%^&*()~/\\"));
      for (auto& str : strs) {
        if (str.find("HLT_") == 0) {
          m_usedPaths.insert(str);
        }
      }

      m_eventCache = &eventCache;
      std::string pathPartialName = iConfig.getParameter<std::string>("partialPathName");
      m_dirname = iConfig.getUntrackedParameter("mainDQMDirname", std::string("HLT/FSQ/")) + pathPartialName + "/";
      m_pset = iConfig;
    };
    virtual void analyze(const edm::Event& iEvent,
                         const edm::EventSetup& iSetup,
                         const HLTConfigProvider& hltConfig,
                         const trigger::TriggerEvent& trgEvent,
                         const edm::TriggerResults& triggerResults,
                         const edm::TriggerNames& triggerNames,
                         float weight) = 0;
    virtual void book(DQMStore::IBooker& booker) = 0;
    virtual void getAndStoreTokens(edm::ConsumesCollector&& iC) = 0;

    std::unique_ptr<triggerExpression::Evaluator> m_expression;
    triggerExpression::Data* m_eventCache;
    std::string m_dirname;
    std::map<std::string, MonitorElement*> m_histos;
    std::set<std::string> m_usedPaths;
    edm::ParameterSet m_pset;
  };
  //################################################################################################
  //
  // Handle objects saved into hlt event by hlt filters
  //
  //################################################################################################
  enum SpecialFilters { None, BestVertexMatching, ApplyJEC };
  template <class TInputCandidateType, class TOutputCandidateType, SpecialFilters filter = None>
  class HandlerTemplate : public BaseHandler {
  private:
    std::string m_dqmhistolabel;
    std::string m_pathPartialName;    //#("HLT_DiPFJetAve30_HFJEC_");
    std::string m_filterPartialName;  //#("ForHFJECBase"); // Calo jet preFilter

    int m_combinedObjectDimension;

    StringCutObjectSelector<TInputCandidateType> m_singleObjectSelection;
    StringCutObjectSelector<std::vector<TOutputCandidateType> > m_combinedObjectSelection;
    StringObjectFunction<std::vector<TOutputCandidateType> > m_combinedObjectSortFunction;
    std::map<std::string, std::shared_ptr<StringObjectFunction<std::vector<TOutputCandidateType> > > >
        m_plottersCombinedObject;
    std::map<std::string, std::shared_ptr<StringObjectFunction<TInputCandidateType> > > m_plottersSingleObject;
    /// xxx
    static const int SingleObjectPlotter = 0;
    static const int CombinedObjectPlotter = 1;
    std::map<std::string, int> m_plotterType;
    std::vector<edm::ParameterSet> m_combinedObjectDrawables;
    std::vector<edm::ParameterSet> m_singleObjectDrawables;  // for all single objects passing preselection
    bool m_isSetup;
    edm::InputTag m_input;
    std::map<std::string, edm::EDGetToken> m_tokens;

  public:
    HandlerTemplate(const edm::ParameterSet& iConfig, triggerExpression::Data& eventCache)
        : BaseHandler(iConfig, eventCache),
          m_singleObjectSelection(iConfig.getParameter<std::string>("singleObjectsPreselection")),
          m_combinedObjectSelection(iConfig.getParameter<std::string>("combinedObjectSelection")),
          m_combinedObjectSortFunction(iConfig.getParameter<std::string>("combinedObjectSortCriteria")) {
      std::string type = iConfig.getParameter<std::string>("handlerType");
      if (type != "FromHLT") {
        m_input = iConfig.getParameter<edm::InputTag>("inputCol");
      }

      m_dqmhistolabel = iConfig.getParameter<std::string>("dqmhistolabel");
      m_filterPartialName =
          iConfig.getParameter<std::string>("partialFilterName");  // std::string find is used to match filter
      m_pathPartialName = iConfig.getParameter<std::string>("partialPathName");
      m_combinedObjectDimension = iConfig.getParameter<int>("combinedObjectDimension");
      m_combinedObjectDrawables = iConfig.getParameter<std::vector<edm::ParameterSet> >("combinedObjectDrawables");
      m_singleObjectDrawables = iConfig.getParameter<std::vector<edm::ParameterSet> >("singleObjectDrawables");
      m_isSetup = false;
    }
    ~HandlerTemplate() override = default;
    void book(DQMStore::IBooker& booker) override {
      if (!m_isSetup) {
        booker.setCurrentFolder(m_dirname);
        m_isSetup = true;
        std::vector<std::vector<edm::ParameterSet>*> todo(2, (std::vector<edm::ParameterSet>*)nullptr);
        todo[CombinedObjectPlotter] = &m_combinedObjectDrawables;
        todo[SingleObjectPlotter] = &m_singleObjectDrawables;
        for (size_t ti = 0; ti < todo.size(); ++ti) {
          for (size_t i = 0; i < todo[ti]->size(); ++i) {
            std::string histoName = m_dqmhistolabel + "_" + todo[ti]->at(i).template getParameter<std::string>("name");
            std::string expression = todo[ti]->at(i).template getParameter<std::string>("expression");
            int bins = todo[ti]->at(i).template getParameter<int>("bins");
            double rangeLow = todo[ti]->at(i).template getParameter<double>("min");
            double rangeHigh = todo[ti]->at(i).template getParameter<double>("max");
            m_histos[histoName] = booker.book1D(histoName, histoName, bins, rangeLow, rangeHigh);
            m_plotterType[histoName] = ti;
            if (ti == CombinedObjectPlotter) {
              auto* func = new StringObjectFunction<std::vector<TOutputCandidateType> >(expression);
              m_plottersCombinedObject[histoName] =
                  std::shared_ptr<StringObjectFunction<std::vector<TOutputCandidateType> > >(func);
            } else {
              auto* func = new StringObjectFunction<TInputCandidateType>(expression);
              m_plottersSingleObject[histoName] = std::shared_ptr<StringObjectFunction<TInputCandidateType> >(func);
            }
          }
        }
      }
    }
    void getAndStoreTokens(edm::ConsumesCollector&& iC) override {
      edm::EDGetTokenT<std::vector<TInputCandidateType> > tok = iC.consumes<std::vector<TInputCandidateType> >(m_input);
      m_tokens[m_input.encode()] = edm::EDGetToken(tok);
    }

    //#############################################################################
    // Count objects. To avoid code duplication we do it in a separate template -
    //  - partial specialization not easy...:
    // http://stackoverflow.com/questions/21182729/specializing-single-method-in-a-big-template-class
    //#############################################################################
    template <class T>
    int count(const edm::Event& iEvent, InputTag& input, StringCutObjectSelector<T>& sel, float weight) {
      int ret = 0;
      Handle<std::vector<T> > hIn;
      iEvent.getByToken(m_tokens[input.encode()], hIn);
      if (!hIn.isValid()) {
        edm::LogError("FSQDiJetAve") << "product not found: " << input.encode();
        return -1;  // return nonsense value
      }
      for (size_t i = 0; i < hIn->size(); ++i) {
        bool preselection = sel(hIn->at(i));
        if (preselection) {
          fillSingleObjectPlots(hIn->at(i), weight);
          ret += 1;
        }
      }
      return ret;
    }

    // FIXME (?): code duplication
    void fillSingleObjectPlots(const TInputCandidateType& cand, float weight) {
      std::map<std::string, MonitorElement*>::iterator it, itE;
      it = m_histos.begin();
      itE = m_histos.end();
      for (; it != itE; ++it) {
        if (m_plotterType[it->first] != SingleObjectPlotter)
          continue;
        float val = (*m_plottersSingleObject[it->first])(cand);
        it->second->Fill(val, weight);
      }
    }
    // Notes:
    //  - this function (and specialized versions) are responsible for calling
    //     fillSingleObjectPlots for all single objects passing the single
    //     object preselection criteria
    //  - FIXME this function should take only event/ event setup (?)
    //  - FIXME responsibility to apply preselection should be elsewhere
    //          hard to fix, since we dont want to copy all objects due to
    //          performance reasons
    //  - note: implementation below working when in/out types are equal
    //          in other cases you must provide specialized version (see below)
    void getFilteredCands(TInputCandidateType*,
                          std::vector<TOutputCandidateType>& cands,
                          const edm::Event& iEvent,
                          const edm::EventSetup& iSetup,
                          const HLTConfigProvider& hltConfig,
                          const trigger::TriggerEvent& trgEvent,
                          float weight) {
      Handle<std::vector<TInputCandidateType> > hIn;
      iEvent.getByToken(m_tokens[m_input.encode()], hIn);

      if (!hIn.isValid()) {
        edm::LogError("FSQDiJetAve") << "product not found: " << m_input.encode();
        return;
      }

      for (size_t i = 0; i < hIn->size(); ++i) {
        bool preselection = m_singleObjectSelection(hIn->at(i));
        if (preselection) {
          fillSingleObjectPlots(hIn->at(i), weight);
          cands.push_back(hIn->at(i));
        }
      }
    }

    std::vector<std::string> findPathAndFilter(const HLTConfigProvider& hltConfig) {
      std::vector<std::string> ret(2, "");
      std::string filterFullName = "";
      std::string pathFullName = "";
      std::vector<std::string> filtersForThisPath;
      //int pathIndex = -1;
      int numPathMatches = 0;
      int numFilterMatches = 0;
      for (size_t i = 0; i < hltConfig.size(); ++i) {
        if (hltConfig.triggerName(i).find(m_pathPartialName) == std::string::npos)
          continue;
        pathFullName = hltConfig.triggerName(i);
        //pathIndex = i;
        ++numPathMatches;
        std::vector<std::string> moduleLabels = hltConfig.moduleLabels(i);
        for (auto& moduleLabel : moduleLabels) {
          if ("EDFilter" == hltConfig.moduleEDMType(moduleLabel)) {
            filtersForThisPath.push_back(moduleLabel);
            if (moduleLabel.find(m_filterPartialName) != std::string::npos) {
              filterFullName = moduleLabel;
              ++numFilterMatches;
            }
          }
        }
      }

      // LogWarning or LogError?
      if (numPathMatches != 1) {
        edm::LogInfo("FSQDiJetAve") << "Problem: found " << numPathMatches << " paths matching " << m_pathPartialName
                                    << std::endl;
        return ret;
      }
      ret[0] = pathFullName;
      if (numFilterMatches != 1) {
        edm::LogError("FSQDiJetAve") << "Problem: found " << numFilterMatches << " filter matching "
                                     << m_filterPartialName << " in path " << m_pathPartialName << std::endl;
        return ret;
      }
      ret[1] = filterFullName;
      return ret;
    }

    void analyze(const edm::Event& iEvent,
                 const edm::EventSetup& iSetup,
                 const HLTConfigProvider& hltConfig,
                 const trigger::TriggerEvent& trgEvent,
                 const edm::TriggerResults& triggerResults,
                 const edm::TriggerNames& triggerNames,
                 float weight) override {
      size_t found = 0;
      for (size_t i = 0; i < triggerNames.size(); ++i) {
        auto itUsedPaths = m_usedPaths.begin();
        for (; itUsedPaths != m_usedPaths.end(); ++itUsedPaths) {
          if (triggerNames.triggerName(i).find(*itUsedPaths) != std::string::npos) {
            ++found;
            break;
          }
        }

        if (found == m_usedPaths.size())
          break;
      }
      if (found != m_usedPaths.size()) {
        edm::LogInfo("FSQDiJetAve") << "One of requested paths not found, skipping event";
        return;
      }
      if (m_eventCache->configurationUpdated()) {
        m_expression->init(*m_eventCache);
      }
      if (not(*m_expression)(*m_eventCache))
        return;

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
      getFilteredCands((TInputCandidateType*)nullptr, cands, iEvent, iSetup, hltConfig, trgEvent, weight);

      if (cands.empty())
        return;

      std::vector<TOutputCandidateType> bestCombinationFromCands = getBestCombination(cands);
      if (bestCombinationFromCands.empty())
        return;

      // plot
      std::map<std::string, MonitorElement*>::iterator it, itE;
      it = m_histos.begin();
      itE = m_histos.end();
      for (; it != itE; ++it) {
        if (m_plotterType[it->first] != CombinedObjectPlotter)
          continue;
        float val = (*m_plottersCombinedObject[it->first])(bestCombinationFromCands);
        it->second->Fill(val, weight);
      }
    }

    std::vector<TOutputCandidateType> getBestCombination(std::vector<TOutputCandidateType>& cands) {
      int columnSize = cands.size();
      std::vector<int> currentCombination(m_combinedObjectDimension, 0);
      std::vector<int> bestCombination(m_combinedObjectDimension, -1);

      int maxCombinations = 1;
      int cnt = 0;
      while (cnt < m_combinedObjectDimension) {
        cnt += 1;
        maxCombinations *= columnSize;
      }

      cnt = 0;
      float bestCombinedCandVal = -1;
      while (cnt < maxCombinations) {
        cnt += 1;

        // 1. Check if current combination contains duplicates
        std::vector<int> currentCombinationCopy(currentCombination);
        std::vector<int>::iterator it;
        std::sort(currentCombinationCopy.begin(), currentCombinationCopy.end());
        it = std::unique(currentCombinationCopy.begin(), currentCombinationCopy.end());
        currentCombinationCopy.resize(std::distance(currentCombinationCopy.begin(), it));
        bool duplicatesPresent = currentCombination.size() != currentCombinationCopy.size();
        // 2. If no duplicates found -
        //          - check if current combination passes the cut
        //          - rank current combination
        if (!duplicatesPresent) {  // no duplicates, we can consider this combined object
          std::vector<TOutputCandidateType> currentCombinationFromCands;
          currentCombinationFromCands.reserve(m_combinedObjectDimension);
          for (int i = 0; i < m_combinedObjectDimension; ++i) {
            currentCombinationFromCands.push_back(cands.at(currentCombination.at(i)));
          }
          bool isOK = m_combinedObjectSelection(currentCombinationFromCands);
          if (isOK) {
            float curVal = m_combinedObjectSortFunction(currentCombinationFromCands);
            // FIXME
            if (curVal < 0) {
              edm::LogError("FSQDiJetAve")
                  << "Problem: ranking function returned negative value: " << curVal << std::endl;
            } else if (curVal > bestCombinedCandVal) {
              //std::cout << curVal << " " << bestCombinedCandVal << std::endl;
              bestCombinedCandVal = curVal;
              bestCombination = currentCombination;
            }
          }
        }
        // 3. Prepare next combination to test
        //    note to future self: less error prone method with modulo
        currentCombination.at(m_combinedObjectDimension - 1) += 1;  // increase last number
        int carry = 0;
        for (int i = m_combinedObjectDimension - 1; i >= 0;
             --i) {  // iterate over all numbers, check if we are out of range
          currentCombination.at(i) += carry;
          carry = 0;
          if (currentCombination.at(i) >= columnSize) {
            carry = 1;
            currentCombination.at(i) = 0;
          }
        }
      }  // combinations loop ends

      std::vector<TOutputCandidateType> bestCombinationFromCands;
      if (!bestCombination.empty() && bestCombination.at(0) >= 0) {
        for (int i = 0; i < m_combinedObjectDimension; ++i) {
          bestCombinationFromCands.push_back(cands.at(bestCombination.at(i)));
        }
      }
      return bestCombinationFromCands;
    }
  };
  //#############################################################################
  // Read any object inheriting from reco::Candidate. Save p4
  //
  //  problem: for reco::Candidate there is no reflex dictionary, so selector
  //  wont work
  //#############################################################################
  template <>
  void HandlerTemplate<reco::Candidate::LorentzVector, reco::Candidate::LorentzVector>::getAndStoreTokens(
      edm::ConsumesCollector&& iC) {
    edm::EDGetTokenT<View<reco::Candidate> > tok = iC.consumes<View<reco::Candidate> >(m_input);
    m_tokens[m_input.encode()] = edm::EDGetToken(tok);
  }
  template <>
  void HandlerTemplate<reco::Candidate::LorentzVector, reco::Candidate::LorentzVector>::getFilteredCands(
      reco::Candidate::LorentzVector*,  // pass a dummy pointer, makes possible to select correct getFilteredCands
      std::vector<reco::Candidate::LorentzVector>& cands,  // output collection
      const edm::Event& iEvent,
      const edm::EventSetup& iSetup,
      const HLTConfigProvider& hltConfig,
      const trigger::TriggerEvent& trgEvent,
      float weight) {
    Handle<View<reco::Candidate> > hIn;
    iEvent.getByToken(m_tokens[m_input.encode()], hIn);
    if (!hIn.isValid()) {
      edm::LogError("FSQDiJetAve") << "product not found: " << m_input.encode();
      return;
    }
    for (auto const& i : *hIn) {
      bool preselection = m_singleObjectSelection(i.p4());
      if (preselection) {
        fillSingleObjectPlots(i.p4(), weight);
        cands.push_back(i.p4());
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
  template <>
  void HandlerTemplate<reco::Track, int>::getFilteredCands(
      reco::Track*,             // pass a dummy pointer, makes possible to select correct getFilteredCands
      std::vector<int>& cands,  // output collection
      const edm::Event& iEvent,
      const edm::EventSetup& iSetup,
      const HLTConfigProvider& hltConfig,
      const trigger::TriggerEvent& trgEvent,
      float weight) {
    cands.clear();
    cands.push_back(count<reco::Track>(iEvent, m_input, m_singleObjectSelection, weight));
  }
  template <>
  void HandlerTemplate<reco::GenParticle, int>::getFilteredCands(
      reco::GenParticle*,       // pass a dummy pointer, makes possible to select correct getFilteredCands
      std::vector<int>& cands,  // output collection
      const edm::Event& iEvent,
      const edm::EventSetup& iSetup,
      const HLTConfigProvider& hltConfig,
      const trigger::TriggerEvent& trgEvent,
      float weight) {
    cands.clear();
    cands.push_back(count<reco::GenParticle>(iEvent, m_input, m_singleObjectSelection, weight));
  }
  //#############################################################################
  //
  // Count any object inheriting from reco::Track that is not to distant from
  // selected vertex. Save into std::vector<int>
  // note: this is similar to recoCand counter (code duplication is hard to
  //       avoid in this case)
  //
  //#############################################################################
  template <>
  void HandlerTemplate<reco::Track, int, BestVertexMatching>::getAndStoreTokens(edm::ConsumesCollector&& iC) {
    edm::EDGetTokenT<std::vector<reco::Track> > tok = iC.consumes<std::vector<reco::Track> >(m_input);
    m_tokens[m_input.encode()] = edm::EDGetToken(tok);

    edm::InputTag lVerticesTag = m_pset.getParameter<edm::InputTag>("vtxCollection");
    edm::EDGetTokenT<reco::VertexCollection> tok2 = iC.consumes<reco::VertexCollection>(lVerticesTag);
    m_tokens[lVerticesTag.encode()] = edm::EDGetToken(tok2);
  }

  template <>
  void HandlerTemplate<reco::Track, int, BestVertexMatching>::getFilteredCands(
      reco::Track*,             // pass a dummy pointer, makes possible to select correct getFilteredCands
      std::vector<int>& cands,  // output collection
      const edm::Event& iEvent,
      const edm::EventSetup& iSetup,
      const HLTConfigProvider& hltConfig,
      const trigger::TriggerEvent& trgEvent,
      float weight) {
    // this is not elegant, but should be thread safe
    static const edm::InputTag lVerticesTag = m_pset.getParameter<edm::InputTag>("vtxCollection");
    static const int lMinNDOF = m_pset.getParameter<int>("minNDOF");                        //7
    static const double lMaxZ = m_pset.getParameter<double>("maxZ");                        // 15
    static const double lMaxDZ = m_pset.getParameter<double>("maxDZ");                      // 0.12
    static const double lMaxDZ2dzsigma = m_pset.getParameter<double>("maxDZ2dzsigma");      // 3
    static const double lMaxDXY = m_pset.getParameter<double>("maxDXY");                    // 0.12
    static const double lMaxDXY2dxysigma = m_pset.getParameter<double>("maxDXY2dxysigma");  // 3

    cands.clear();
    cands.push_back(0);

    edm::Handle<reco::VertexCollection> vertices;
    iEvent.getByToken(m_tokens[lVerticesTag.encode()], vertices);

    math::XYZPoint vtxPoint(0.0, 0.0, 0.0);
    double vzErr = 0.0, vxErr = 0.0, vyErr = 0.0;

    // take first vertex passing the criteria
    bool vtxfound = false;
    for (const auto& vertex : *vertices) {
      if (vertex.ndof() < lMinNDOF)
        continue;
      if (fabs(vertex.z()) > lMaxZ)
        continue;

      vtxPoint = vertex.position();
      vzErr = vertex.zError();
      vxErr = vertex.xError();
      vyErr = vertex.yError();
      vtxfound = true;
      break;
    }
    if (!vtxfound)
      return;

    Handle<std::vector<reco::Track> > hIn;
    iEvent.getByToken(m_tokens[m_input.encode()], hIn);
    if (!hIn.isValid()) {
      edm::LogError("FSQDiJetAve") << "product not found: " << m_input.encode();
      return;
    }

    for (auto const& i : *hIn) {
      if (!m_singleObjectSelection(i))
        continue;

      double absdz = fabs(i.dz(vtxPoint));
      if (absdz > lMaxDZ)
        continue;  // TODO...

      //      double absdxy = fabs(-1. * i.dxy(vtxPoint));
      double absdxy = fabs(i.dxy(vtxPoint));
      if (absdxy > lMaxDXY)
        continue;

      double dzsigma2 = i.dzError() * i.dzError() + vzErr * vzErr;
      if (absdz * absdz > lMaxDZ2dzsigma * lMaxDZ2dzsigma * dzsigma2)
        continue;

      double dxysigma2 = i.dxyError() * i.dxyError() + vxErr * vyErr;
      if (absdxy * absdxy > lMaxDXY2dxysigma * lMaxDXY2dxysigma * dxysigma2)
        continue;

      cands.at(0) += 1;
    }  //loop over tracks
  }
  //#############################################################################
  //
  // Apply JEC to PFJets
  //
  //#############################################################################
  template <>
  void HandlerTemplate<reco::PFJet, reco::PFJet, ApplyJEC>::getAndStoreTokens(edm::ConsumesCollector&& iC) {
    edm::EDGetTokenT<std::vector<reco::PFJet> > tok = iC.consumes<std::vector<reco::PFJet> >(m_input);
    m_tokens[m_input.encode()] = edm::EDGetToken(tok);

    edm::InputTag jetCorTag = m_pset.getParameter<edm::InputTag>("PFJetCorLabel");
    edm::EDGetTokenT<reco::JetCorrector> jetcortoken = iC.consumes<reco::JetCorrector>(jetCorTag);
    m_tokens[jetCorTag.encode()] = edm::EDGetToken(jetcortoken);
  }

  template <>
  void HandlerTemplate<reco::PFJet, reco::PFJet, ApplyJEC>::getFilteredCands(
      reco::PFJet*,                     // pass a dummy pointer, makes possible to select correct getFilteredCands
      std::vector<reco::PFJet>& cands,  // output collection
      const edm::Event& iEvent,
      const edm::EventSetup& iSetup,
      const HLTConfigProvider& hltConfig,
      const trigger::TriggerEvent& trgEvent,
      float weight) {
    cands.clear();
    static const edm::InputTag jetCorTag = m_pset.getParameter<edm::InputTag>("PFJetCorLabel");
    edm::Handle<reco::JetCorrector> pfcorrector;
    iEvent.getByToken(m_tokens[jetCorTag.encode()], pfcorrector);

    Handle<std::vector<reco::PFJet> > hIn;
    iEvent.getByToken(m_tokens[m_input.encode()], hIn);

    if (!hIn.isValid()) {
      edm::LogError("FSQDiJetAve") << "product not found: " << m_input.encode();
      return;
    }

    for (auto const& i : *hIn) {
      double scale = pfcorrector->correction(i);
      reco::PFJet newPFJet(scale * i.p4(), i.vertex(), i.getSpecific(), i.getJetConstituents());

      bool preselection = m_singleObjectSelection(newPFJet);
      if (preselection) {
        fillSingleObjectPlots(newPFJet, weight);
        cands.push_back(newPFJet);
      }
    }
  }
  //#############################################################################
  //
  // Count any object inheriting from reco::Candidate. Save into std::vector<int>
  //  same problem as for reco::Candidate handler ()
  //
  //#############################################################################
  template <>
  void HandlerTemplate<reco::Candidate::LorentzVector, int>::getAndStoreTokens(edm::ConsumesCollector&& iC) {
    edm::EDGetTokenT<View<reco::Candidate> > tok = iC.consumes<View<reco::Candidate> >(m_input);
    m_tokens[m_input.encode()] = edm::EDGetToken(tok);
  }
  template <>
  void HandlerTemplate<reco::Candidate::LorentzVector, int>::getFilteredCands(
      reco::Candidate::LorentzVector*,  // pass a dummy pointer, makes possible to select correct getFilteredCands
      std::vector<int>& cands,          // output collection
      const edm::Event& iEvent,
      const edm::EventSetup& iSetup,
      const HLTConfigProvider& hltConfig,
      const trigger::TriggerEvent& trgEvent,
      float weight) {
    cands.clear();
    cands.push_back(0);

    Handle<View<reco::Candidate> > hIn;
    iEvent.getByToken(m_tokens[m_input.encode()], hIn);
    if (!hIn.isValid()) {
      edm::LogError("FSQDiJetAve") << "product not found: " << m_input.encode();
      return;
    }
    for (auto const& i : *hIn) {
      bool preselection = m_singleObjectSelection(i.p4());
      if (preselection) {
        fillSingleObjectPlots(i.p4(), weight);
        cands.at(0) += 1;
      }
    }
  }
  //#############################################################################
  //
  // Read and save trigger::TriggerObject from triggerEvent
  //
  //#############################################################################
  template <>
  void HandlerTemplate<trigger::TriggerObject, trigger::TriggerObject>::getFilteredCands(
      trigger::TriggerObject*,
      std::vector<trigger::TriggerObject>& cands,
      const edm::Event& iEvent,
      const edm::EventSetup& iSetup,
      const HLTConfigProvider& hltConfig,
      const trigger::TriggerEvent& trgEvent,
      float weight) {
    // 1. Find matching path. Inside matchin path find matching filter
    std::string filterFullName = findPathAndFilter(hltConfig)[1];
    if (filterFullName.empty()) {
      return;
    }

    // 2. Fetch HLT objects saved by selected filter. Save those fullfilling preselection
    //      objects are saved in cands variable
    const std::string& process = trgEvent.usedProcessName();  // broken?
    edm::InputTag hltTag(filterFullName, "", process);

    const int hltIndex = trgEvent.filterIndex(hltTag);
    if (hltIndex >= trgEvent.sizeFilters()) {
      edm::LogInfo("FSQDiJetAve") << "Cannot determine hlt index for |" << filterFullName << "|" << process;
      return;
    }

    const trigger::TriggerObjectCollection& toc(trgEvent.getObjects());
    const trigger::Keys& khlt = trgEvent.filterKeys(hltIndex);

    auto kj = khlt.begin();

    for (; kj != khlt.end(); ++kj) {
      bool preselection = m_singleObjectSelection(toc[*kj]);
      if (preselection) {
        fillSingleObjectPlots(toc[*kj], weight);
        cands.push_back(toc[*kj]);
      }
    }
  }

  typedef HandlerTemplate<trigger::TriggerObject, trigger::TriggerObject> HLTHandler;
  typedef HandlerTemplate<reco::Candidate::LorentzVector, reco::Candidate::LorentzVector>
      RecoCandidateHandler;  // in fact reco::Candidate, reco::Candidate::LorentzVector
  typedef HandlerTemplate<reco::PFJet, reco::PFJet> RecoPFJetHandler;
  typedef HandlerTemplate<reco::PFJet, reco::PFJet, ApplyJEC> RecoPFJetWithJECHandler;
  typedef HandlerTemplate<reco::Track, reco::Track> RecoTrackHandler;
  typedef HandlerTemplate<reco::Photon, reco::Photon> RecoPhotonHandler;
  typedef HandlerTemplate<reco::Muon, reco::Muon> RecoMuonHandler;
  typedef HandlerTemplate<reco::GenParticle, reco::GenParticle> RecoGenParticleHandler;
  typedef HandlerTemplate<reco::Candidate::LorentzVector, int> RecoCandidateCounter;
  typedef HandlerTemplate<reco::Track, int> RecoTrackCounter;
  typedef HandlerTemplate<reco::Track, int, BestVertexMatching> RecoTrackCounterWithVertexConstraint;
  typedef HandlerTemplate<reco::GenParticle, int> RecoGenParticleCounter;
}  // namespace FSQ
//################################################################################################
//
// Plugin functions
//
//################################################################################################
FSQDiJetAve::FSQDiJetAve(const edm::ParameterSet& iConfig)
    : m_eventCache(iConfig.getParameterSet("triggerConfiguration"), consumesCollector()) {
  m_useGenWeight = iConfig.getParameter<bool>("useGenWeight");
  if (m_useGenWeight) {
    m_genEvInfoToken = consumes<GenEventInfoProduct>(edm::InputTag("generator"));
  }

  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
  triggerSummaryToken = consumes<trigger::TriggerEvent>(triggerSummaryLabel_);
  triggerResultsToken = consumes<edm::TriggerResults>(triggerResultsLabel_);

  triggerSummaryFUToken = consumes<trigger::TriggerEvent>(
      edm::InputTag(triggerSummaryLabel_.label(), triggerSummaryLabel_.instance(), std::string("FU")));
  triggerResultsFUToken = consumes<edm::TriggerResults>(
      edm::InputTag(triggerResultsLabel_.label(), triggerResultsLabel_.instance(), std::string("FU")));

  std::vector<edm::ParameterSet> todo = iConfig.getParameter<std::vector<edm::ParameterSet> >("todo");
  for (const auto& pset : todo) {
    std::string type = pset.getParameter<std::string>("handlerType");
    if (type == "FromHLT") {
      m_handlers.push_back(std::make_shared<FSQ::HLTHandler>(pset, m_eventCache));
    } else if (type == "RecoCandidateCounter") {
      m_handlers.push_back(std::make_shared<FSQ::RecoCandidateCounter>(pset, m_eventCache));
    } else if (type == "RecoTrackCounter") {
      m_handlers.push_back(std::make_shared<FSQ::RecoTrackCounter>(pset, m_eventCache));
    } else if (type == "RecoTrackCounterWithVertexConstraint") {
      m_handlers.push_back(std::make_shared<FSQ::RecoTrackCounterWithVertexConstraint>(pset, m_eventCache));
    } else if (type == "FromRecoCandidate") {
      m_handlers.push_back(std::make_shared<FSQ::RecoCandidateHandler>(pset, m_eventCache));
    } else if (type == "RecoPFJet") {
      m_handlers.push_back(std::make_shared<FSQ::RecoPFJetHandler>(pset, m_eventCache));
    } else if (type == "RecoPFJetWithJEC") {
      m_handlers.push_back(std::make_shared<FSQ::RecoPFJetWithJECHandler>(pset, m_eventCache));
    } else if (type == "RecoTrack") {
      m_handlers.push_back(std::make_shared<FSQ::RecoTrackHandler>(pset, m_eventCache));
    } else if (type == "RecoPhoton") {
      m_handlers.push_back(std::make_shared<FSQ::RecoPhotonHandler>(pset, m_eventCache));
    } else if (type == "RecoMuon") {
      m_handlers.push_back(std::make_shared<FSQ::RecoMuonHandler>(pset, m_eventCache));
    } else if (type == "RecoGenParticleCounter") {
      m_handlers.push_back(std::make_shared<FSQ::RecoGenParticleCounter>(pset, m_eventCache));
    } else if (type == "RecoGenParticleHandler") {
      m_handlers.push_back(std::make_shared<FSQ::RecoGenParticleHandler>(pset, m_eventCache));
    } else {
      throw cms::Exception("FSQ DQM handler not know: " + type);
    }
  }
  for (auto& m_handler : m_handlers) {
    m_handler->getAndStoreTokens(consumesCollector());
  }
}

FSQDiJetAve::~FSQDiJetAve() = default;

void FSQDiJetAve::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  if (not m_eventCache.setEvent(iEvent, iSetup)) {
    edm::LogError("FSQDiJetAve") << "Could not setup the filter";
  }

  //---------- triggerResults ----------
  iEvent.getByToken(triggerResultsToken, m_triggerResults);
  if (!m_triggerResults.isValid()) {
    iEvent.getByToken(triggerResultsFUToken, m_triggerResults);
    if (!m_triggerResults.isValid()) {
      edm::LogError("FSQDiJetAve") << "TriggerResults not valid, skippng event";
      return;
    }
  }

  //---------- triggerResults ----------
  if (m_triggerResults.isValid()) {
    m_triggerNames = iEvent.triggerNames(*m_triggerResults);
  } else {
    edm::LogError("FSQDiJetAve") << "TriggerResults not found";
    return;
  }

  //---------- triggerSummary ----------
  iEvent.getByToken(triggerSummaryToken, m_trgEvent);
  if (!m_trgEvent.isValid()) {
    iEvent.getByToken(triggerSummaryFUToken, m_trgEvent);
    if (!m_trgEvent.isValid()) {
      edm::LogInfo("FSQDiJetAve") << "TriggerEvent not found, ";
      return;
    }
  }

  float weight = 1.;
  if (m_useGenWeight) {
    edm::Handle<GenEventInfoProduct> hGW;
    iEvent.getByToken(m_genEvInfoToken, hGW);
    weight = hGW->weight();
  }

  for (auto& m_handler : m_handlers) {
    m_handler->analyze(
        iEvent, iSetup, m_hltConfig, *m_trgEvent.product(), *m_triggerResults.product(), m_triggerNames, weight);
  }
}
// ------------ method called when starting to processes a run  ------------
//*
void FSQDiJetAve::dqmBeginRun(edm::Run const& run, edm::EventSetup const& c) {
  bool changed(true);
  std::string processName = triggerResultsLabel_.process();
  if (m_hltConfig.init(run, c, processName, changed)) {
    LogDebug("FSQDiJetAve") << "HLTConfigProvider failed to initialize.";
  }
}
void FSQDiJetAve::bookHistograms(DQMStore::IBooker& booker, edm::Run const& run, edm::EventSetup const& c) {
  for (auto& m_handler : m_handlers) {
    m_handler->book(booker);
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
void FSQDiJetAve::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FSQDiJetAve);
