#ifndef HLTriggerOffline_Exotica_HLTExoticaPlotter_H
#define HLTriggerOffline_Exotica_HLTExoticaPlotter_H

/** \class HLTExoticaPlotter
 *  Generate histograms for trigger efficiencies Exotica related
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/EXOTICATriggerValidation
 *
 *  \author  Thiago R. Fernandez Perez Tomei
 *           Based and adapted from:
 *           J. Duarte Campderros code from HLTriggerOffline/Higgs
 *           J. Klukas, M. Vander Donckt and J. Alcaraz code
 *           from the HLTriggerOffline/Muon package.
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <vector>
#include <cstring>
#include <map>
#include <set>

//const unsigned int kNull = (unsigned int) - 1;

class EVTColContainer;

class HLTExoticaPlotter {
public:
    HLTExoticaPlotter(const edm::ParameterSet & pset, const std::string & hltPath,
                      const std::vector<unsigned int> & objectsType);
    ~HLTExoticaPlotter();
    void beginJob();
    void beginRun(const edm::Run &, const edm::EventSetup &);
    void plotterBookHistos(DQMStore::IBooker & iBooker, const edm::Run & iRun, const edm::EventSetup & iSetup);
    void analyze(const bool & isPassTrigger, const std::string & source,
                 const std::vector<reco::LeafCandidate> & matches);

    inline const std::string gethltpath() const
    {
        return _hltPath;
    }

private:
    void bookHist(DQMStore::IBooker & iBooker, const std::string & source, const std::string & objType, const std::string & variable);
    void fillHist(const bool & passTrigger, const std::string & source,
                  const std::string & objType, const std::string & var,
                  const float & value);

    std::string _hltPath;
    std::string _hltProcessName;

    std::set<unsigned int> _objectsType;
    // Number of objects (elec,muons, ...) needed in the hlt path
    unsigned int _nObjects;

    std::vector<double> _parametersEta;
    std::vector<double> _parametersPhi;
    std::vector<double> _parametersTurnOn;

    std::map<std::string, MonitorElement *> _elements;
};
#endif
