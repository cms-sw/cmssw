#ifndef HLTriggerOffline_Higgs_HLTHiggsPlotter_H
#define HLTriggerOffline_Higgs_HLTHiggsPlotter_H

/** \class HLTHiggsPlotter
*  Generate histograms for trigger efficiencies Higgs related
*  Documentation available on the CMS TWiki:
*  https://twiki.cern.ch/twiki/bin/view/CMS/HiggsWGHLTValidate
*
*  \author  J. Duarte Campderros (based and adapted on J. Klukas,
*           M. Vander Donckt and J. Alcaraz code from the 
*           HLTriggerOffline/Muon package)
*  \author  J. Klukas, M. Vander Donckt, J. Alcaraz
*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "MatchStruct.h"

#include <vector>
#include <cstring>
#include <map>
#include <set>

struct EVTColContainer;

class HLTHiggsPlotter {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  HLTHiggsPlotter(const edm::ParameterSet &pset,
                  const std::string &hltPath,
                  //const std::string & lastFilter,
                  const std::vector<unsigned int> &objectsType,
                  const unsigned int &minCandidates,
                  const std::vector<double> &NminOneCuts);
  ~HLTHiggsPlotter();
  void beginJob();
  void beginRun(const edm::Run &, const edm::EventSetup &);
  void bookHistograms(DQMStore::IBooker &, const bool &);
  void analyze(const bool &isPassTrigger,
               const std::string &source,
               const std::vector<MatchStruct> &matches,
               const unsigned int &minCandidates);
  void analyze(const bool &isPassTrigger,
               const std::string &source,
               const std::vector<MatchStruct> &matches,
               std::map<std::string, bool> &nMinOne,
               const float &dEtaqq,
               const float &mqq,
               const float &dPhibb,
               const float &CSV1,
               const float &CSV2,
               const float &CSV3,
               const bool &passAllCuts);

  inline const std::string gethltpath() const { return _hltPath; }

private:
  void bookHist(const std::string &source, const std::string &objType, const std::string &variable, DQMStore::IBooker &);
  void fillHist(const bool &passTrigger,
                const std::string &source,
                const std::string &objType,
                const std::string &var,
                const float &value);

  std::string _hltPath;
  //std::string _lastFilter;
  std::string _hltProcessName;

  std::set<unsigned int> _objectsType;
  // Number of objects (elec,muons, ...) needed in the hlt path
  unsigned int _nObjects;

  std::vector<double> _parametersEta;
  std::vector<double> _parametersPhi;
  std::vector<double> _parametersTurnOn;

  std::map<unsigned int, double> _cutMinPt;
  std::map<unsigned int, double> _cutMaxEta;
  std::map<unsigned int, unsigned int> _cutMotherId;
  std::map<unsigned int, std::vector<double> > _cutsDr;

  // The amount of Pt plots needed for the hlt path
  unsigned int _NptPlots;

  //N-1 cut values
  std::vector<double> _NminOneCuts;

  std::map<std::string, MonitorElement *> _elements;
};
#endif
