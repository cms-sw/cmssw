#include "GsfElectronProducer.h"

#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

#include "DataFormats/EcalRecHit/interface/EcalSeverityLevel.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include <iostream>

using namespace reco;

GsfElectronProducer::GsfElectronProducer(const edm::ParameterSet& cfg, const GsfElectronAlgo::HeavyObjectCache* hoc)
    : GsfElectronBaseProducer(cfg, hoc), pfTranslatorParametersChecked_(false) {}

reco::GsfElectronCollection GsfElectronProducer::clonePreviousElectrons(edm::Event const& event) const {
  reco::GsfElectronCollection electrons;

  auto coreElectrons = event.getHandle(inputCfg_.gsfElectronCores);
  const GsfElectronCoreCollection* newCores = coreElectrons.product();

  for (auto const& oldElectron : event.get(inputCfg_.previousGsfElectrons)) {
    const GsfElectronCoreRef oldCoreRef = oldElectron.core();
    const GsfTrackRef oldElectronGsfTrackRef = oldCoreRef->gsfTrack();
    unsigned int icore;
    for (icore = 0; icore < newCores->size(); ++icore) {
      if (oldElectronGsfTrackRef == (*newCores)[icore].gsfTrack()) {
        const GsfElectronCoreRef coreRef = edm::Ref<GsfElectronCoreCollection>(coreElectrons, icore);
        electrons.emplace_back(oldElectron, coreRef);
        break;
      }
    }
  }
  return electrons;
}

void GsfElectronProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  auto electrons = clonePreviousElectrons(event);
  // don't add pflow only electrons if one so wish
  if (strategyCfg_.addPflowElectrons) {
    algo_->completeElectrons(electrons, event, setup, globalCache());
  }
  addPflowInfo(electrons, event);
  fillEvent(electrons, event);
}

void GsfElectronProducer::beginEvent(edm::Event& event, const edm::EventSetup& setup) {
  // extra configuration checks
  if (!pfTranslatorParametersChecked_) {
    pfTranslatorParametersChecked_ = true;
    edm::Handle<edm::ValueMap<float> > pfMva;
    event.getByToken(pfMVA_, pfMva);
    checkPfTranslatorParameters(edm::parameterSet(*pfMva.provenance()));
  }

  // call to base class
  return GsfElectronBaseProducer::beginEvent(event, setup);
}

void GsfElectronProducer::checkPfTranslatorParameters(edm::ParameterSet const& pset) {
  edm::ParameterSet mvaBlock = pset.getParameter<edm::ParameterSet>("MVACutBlock");
  double pfTranslatorMinMva = mvaBlock.getParameter<double>("MVACut");
  double pfTranslatorUndefined = -99.;
  if (strategyCfg_.applyPreselection && (cutsCfgPflow_.minMVA < pfTranslatorMinMva)) {
    // For pure tracker seeded electrons, if MVA is under translatorMinMva, there is no supercluster
    // of any kind available, so GsfElectronCoreProducer has already discarded the electron.
    edm::LogWarning("GsfElectronAlgo|MvaCutTooLow")
        << "Parameter minMVAPflow (" << cutsCfgPflow_.minMVA
        << ") will have no effect on purely tracker seeded electrons."
        << " It is inferior to the cut already applied by PFlow translator (" << pfTranslatorMinMva << ").";
  }
  if (strategyCfg_.applyPreselection && (cutsCfg_.minMVA < pfTranslatorMinMva)) {
    // For ecal seeded electrons, there is a cluster and GsfElectronCoreProducer has kept all electrons,
    // but when MVA is under translatorMinMva, the translator has not stored the supercluster and
    // forced the MVA value to translatorUndefined
    if (cutsCfg_.minMVA > pfTranslatorUndefined) {
      edm::LogWarning("GsfElectronAlgo|IncompletePflowInformation")
          << "Parameter minMVA  (" << cutsCfg_.minMVA << ")is inferior to the cut applied by PFlow translator ("
          << pfTranslatorMinMva << ")."
          << " Some ecal (and eventually tracker) seeded electrons may lack their MVA value and PFlow supercluster.";
    } else {
      // the MVA value has been forced to translatorUndefined, inferior minMVAPflow
      // so the cut actually applied is the PFlow one
      throw cms::Exception("GsfElectronAlgo|BadMvaCut") << "Parameter minMVA is inferior to the lowest possible value."
                                                        << " Every electron will be blessed whatever other criteria.";
    }
  }
}

// now deprecated
void GsfElectronProducer::addPflowInfo(reco::GsfElectronCollection& electrons, edm::Event const& event) const {
  //Isolation Value Maps for PF and EcalDriven electrons
  typedef std::vector<edm::Handle<edm::ValueMap<double> > > IsolationValueMaps;
  IsolationValueMaps pfIsolationValues;
  IsolationValueMaps edIsolationValues;

  //Fill in the Isolation Value Maps for PF and EcalDriven electrons
  std::vector<edm::InputTag> inputTagIsoVals;
  if (!inputCfg_.pfIsoVals.empty()) {
    inputTagIsoVals.push_back(inputCfg_.pfIsoVals.getParameter<edm::InputTag>("pfSumChargedHadronPt"));
    inputTagIsoVals.push_back(inputCfg_.pfIsoVals.getParameter<edm::InputTag>("pfSumPhotonEt"));
    inputTagIsoVals.push_back(inputCfg_.pfIsoVals.getParameter<edm::InputTag>("pfSumNeutralHadronEt"));

    pfIsolationValues.resize(inputTagIsoVals.size());

    for (size_t j = 0; j < inputTagIsoVals.size(); ++j) {
      event.getByLabel(inputTagIsoVals[j], pfIsolationValues[j]);
    }
  }

  if (!inputCfg_.edIsoVals.empty()) {
    inputTagIsoVals.clear();
    inputTagIsoVals.push_back(inputCfg_.edIsoVals.getParameter<edm::InputTag>("edSumChargedHadronPt"));
    inputTagIsoVals.push_back(inputCfg_.edIsoVals.getParameter<edm::InputTag>("edSumPhotonEt"));
    inputTagIsoVals.push_back(inputCfg_.edIsoVals.getParameter<edm::InputTag>("edSumNeutralHadronEt"));

    edIsolationValues.resize(inputTagIsoVals.size());

    for (size_t j = 0; j < inputTagIsoVals.size(); ++j) {
      event.getByLabel(inputTagIsoVals[j], edIsolationValues[j]);
    }
  }

  bool found;
  auto edElectrons = event.getHandle(inputCfg_.previousGsfElectrons);
  auto pfElectrons = event.getHandle(inputCfg_.pflowGsfElectronsTag);
  reco::GsfElectronCollection::const_iterator pfElectron, edElectron;
  unsigned int edIndex, pfIndex;

  for (auto& el : electrons) {
    // Retreive info from pflow electrons
    found = false;
    for (pfIndex = 0, pfElectron = pfElectrons->begin(); pfElectron != pfElectrons->end(); pfIndex++, pfElectron++) {
      if (pfElectron->gsfTrack() == el.gsfTrack()) {
        if (found) {
          edm::LogWarning("GsfElectronProducer") << "associated pfGsfElectron already found";
        } else {
          found = true;

          // Isolation Values
          if (!(pfIsolationValues).empty()) {
            reco::GsfElectronRef pfElectronRef(pfElectrons, pfIndex);
            reco::GsfElectron::PflowIsolationVariables isoVariables;
            isoVariables.sumChargedHadronPt = (*(pfIsolationValues)[0])[pfElectronRef];
            isoVariables.sumPhotonEt = (*(pfIsolationValues)[1])[pfElectronRef];
            isoVariables.sumNeutralHadronEt = (*(pfIsolationValues)[2])[pfElectronRef];
            el.setPfIsolationVariables(isoVariables);
          }

          // el.setPfIsolationVariables(pfElectron->pfIsolationVariables()) ;
          el.setMvaInput(pfElectron->mvaInput());
          el.setMvaOutput(pfElectron->mvaOutput());
          if (el.ecalDrivenSeed()) {
            el.setP4(GsfElectron::P4_PFLOW_COMBINATION,
                     pfElectron->p4(GsfElectron::P4_PFLOW_COMBINATION),
                     pfElectron->p4Error(GsfElectron::P4_PFLOW_COMBINATION),
                     false);
          } else {
            el.setP4(GsfElectron::P4_PFLOW_COMBINATION,
                     pfElectron->p4(GsfElectron::P4_PFLOW_COMBINATION),
                     pfElectron->p4Error(GsfElectron::P4_PFLOW_COMBINATION),
                     true);
          }
          double noCutMin = -999999999.;
          if (el.mva_e_pi() < noCutMin) {
            throw cms::Exception("GsfElectronAlgo|UnexpectedMvaValue") << "unexpected MVA value: " << el.mva_e_pi();
          }
        }
      }
    }

    // Isolation Values
    // Retreive not found info from ed electrons
    if (!(edIsolationValues).empty()) {
      edIndex = 0, edElectron = edElectrons->begin();
      while ((found == false) && (edElectron != edElectrons->end())) {
        if (edElectron->gsfTrack() == el.gsfTrack()) {
          found = true;

          // CONSTRUCTION D UNE REF dans le handle previousElectrons avec l'indice edIndex,
          // puis recuperation dans la ValueMap ED

          reco::GsfElectronRef edElectronRef(edElectrons, edIndex);
          reco::GsfElectron::PflowIsolationVariables isoVariables;
          isoVariables.sumChargedHadronPt = (*(edIsolationValues)[0])[edElectronRef];
          isoVariables.sumPhotonEt = (*(edIsolationValues)[1])[edElectronRef];
          isoVariables.sumNeutralHadronEt = (*(edIsolationValues)[2])[edElectronRef];
          el.setPfIsolationVariables(isoVariables);
        }

        edIndex++;
        edElectron++;
      }
    }

    // Preselection
    setPflowPreselectionFlag(el);
  }
}

void GsfElectronProducer::setPflowPreselectionFlag(GsfElectron& ele) const {
  ele.setPassMvaPreselection(false);

  if (ele.core()->ecalDrivenSeed()) {
    if (ele.mvaOutput().mva_e_pi >= cutsCfg_.minMVA)
      ele.setPassMvaPreselection(true);
  } else {
    if (ele.mvaOutput().mva_e_pi >= cutsCfgPflow_.minMVA)
      ele.setPassMvaPreselection(true);
  }

  if (ele.passingMvaPreselection()) {
    LogTrace("GsfElectronAlgo") << "Main mva criterion is satisfied";
  }

  ele.setPassPflowPreselection(ele.passingMvaPreselection());
}
