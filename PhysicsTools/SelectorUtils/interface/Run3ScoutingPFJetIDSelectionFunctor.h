#ifndef PhysicsTools_SelectorUtils_interface_Run3ScoutingPFJetIDSelectionFunctor_h
#define PhysicsTools_SelectorUtils_interface_Run3ScoutingPFJetIDSelectionFunctor_h

/**
  \class    Run3ScoutingPFJetIDSelectionFunctor Run3ScoutingPFJetIDSelectionFunctor.h "PhysicsTools/SelectorUtils/interface/Run3ScoutingPFJetIDSelectionFunctor.h"
  \brief    Run3ScoutingPF Jet selector for pat::Jets

  Selector functor for pat::Jets that implements quality cuts based on
  studies of noise patterns.

  Please see https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATSelectors
  for a general overview of the selectors.
*/

#ifndef __GCCXML__
#include "FWCore/Framework/interface/ConsumesCollector.h"
#endif
#include "DataFormats/Scouting/interface/Run3ScoutingPFJet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"

#include <TMath.h>

class Run3ScoutingPFJetIDSelectionFunctor {
public:  // interface
  enum Version_t { RUN3Scouting, N_VERSIONS };
  enum Quality_t { TIGHT, TIGHTLEPVETO, N_QUALITY };

  Run3ScoutingPFJetIDSelectionFunctor() {}

#ifndef __GCCXML__
  Run3ScoutingPFJetIDSelectionFunctor(edm::ParameterSet const &params, edm::ConsumesCollector &iC)
      : Run3ScoutingPFJetIDSelectionFunctor(params) {}
#endif

  Run3ScoutingPFJetIDSelectionFunctor(edm::ParameterSet const &params) {
    std::string versionStr = params.getParameter<std::string>("version");
    std::string qualityStr = params.getParameter<std::string>("quality");

    if (versionStr == "RUN3Scouting")
      version_ = RUN3Scouting;
    else {
      edm::LogWarning("BadJetIDQuality") << "JetID quality not specified -- setting default to RUN3Scouting";
      version_ =
          RUN3Scouting;  //set RUN3Scouting as default //this is extremely unsafe --> similarly it's done in PFJetIDSelectionFunctor.h
    }

    if (qualityStr == "TIGHT")
      quality_ = TIGHT;
    else if (qualityStr == "TIGHTLEPVETO")
      quality_ = TIGHTLEPVETO;
    else {
      edm::LogWarning("BadJetIDQuality") << "JetID quality not specified -- setting default to TIGHT";
      quality_ =
          TIGHT;  //set TIGHT as default //this is extremely unsafe --> similarly it's done in PFJetIDSelectionFunctor.h
    }

    initCuts();
  }

  Run3ScoutingPFJetIDSelectionFunctor(Version_t version, Quality_t quality) : version_(version), quality_(quality) {
    initCuts();
  }

  //
  // give a configuration description for derived class
  //
  static edm::ParameterSetDescription getDescription() {
    edm::ParameterSetDescription desc;

    desc.ifValue(
        edm::ParameterDescription<std::string>("version", "RUN3Scouting", true, edm::Comment("")),  //default "version"
        edm::allowedValues<std::string>("RUN3Scouting"));  //more options about "version"
    desc.ifValue(
        edm::ParameterDescription<std::string>("quality", "TIGHT", true, edm::Comment("")),  //default "quality"
        edm::allowedValues<std::string>("TIGHT", "TIGHTLEPVETO"));  //more options about "quality"
    desc.addOptional<std::vector<std::string>>("cutsToIgnore")->setComment("");

    return desc;
  }

  bool operator()(const Run3ScoutingPFJet &jet) {
    // cache some variables
    //float pt = 0.0;
    float eta = 0.0;
    //float phi = 0.0;

    double chf = 0.0;
    double nhf = 0.0;
    //double cef = 0.0;
    double nef = 0.0;
    double muf = 0.0;

    int nch = 0;
    int nconstituents = 0;
    //int nneutrals = 0;

    Run3ScoutingPFJet const *scoutingpfjet = dynamic_cast<Run3ScoutingPFJet const *>(&jet);

    if (scoutingpfjet != nullptr) {
      // CV: need to compute energy fractions in a way that works for corrected as well as for uncorrected PFJets
      //pt = scoutingpfjet->pt();
      eta = scoutingpfjet->eta();
      //phi = scoutingpfjet->phi();
      double jetEnergyUncorrected = scoutingpfjet->chargedHadronEnergy() + scoutingpfjet->neutralHadronEnergy() +
                                    scoutingpfjet->photonEnergy() + scoutingpfjet->electronEnergy() +
                                    scoutingpfjet->muonEnergy() + scoutingpfjet->HFEMEnergy();
      if (jetEnergyUncorrected > 0.) {
        chf = scoutingpfjet->chargedHadronEnergy() / jetEnergyUncorrected;
        nhf = scoutingpfjet->neutralHadronEnergy() / jetEnergyUncorrected;
        //cef= scoutingpfjet->electronEnergy() / jetEnergyUncorrected;  // for now: electron energy is 0, since HLT scouting jets by construction don't contain electrons
        nef = (scoutingpfjet->photonEnergy() + scoutingpfjet->HFEMEnergy()) / jetEnergyUncorrected;
        muf = scoutingpfjet->muonEnergy() / jetEnergyUncorrected;
      }

      nch = scoutingpfjet->chargedHadronMultiplicity() + scoutingpfjet->electronMultiplicity();
      nconstituents = scoutingpfjet->chargedHadronMultiplicity() + scoutingpfjet->electronMultiplicity() +
                      scoutingpfjet->neutralHadronMultiplicity() + scoutingpfjet->photonMultiplicity() +
                      scoutingpfjet->HFEMMultiplicity();
      //nneutrals = scoutingpfjet->neutralHadronMultiplicity() + scoutingpfjet->photonMultiplicity() + scoutingpfjet->HFEMMultiplicity();
    }

    if (version_ == RUN3Scouting) {
      if (std::abs(eta) <= 2.6) {
        if ((chf > CHF_) && (nch > NCH_) && (nef < NEF_) && (nhf < NHF_) && (nconstituents > nConstituents_))
          return true;
        if (quality_ == TIGHTLEPVETO) {
          if (muf < MUF_)
            return true;
        }
      }

      // Cuts for 2.6 <= |eta| <= 2.7
      if ((std::abs(eta) > 2.6) && (std::abs(eta) <= 2.7)) {
        if (nef < NEF_TR_)
          return true;
        if (quality_ == TIGHTLEPVETO) {
          if (muf < MUF_TR_)
            return true;
        }
      }

      // Cuts for 2.7 < |eta| <= 3.0
      if ((std::abs(eta) > 2.7) && (std::abs(eta) <= 3.0)) {
        if (nef < NEF_EC_)
          return true;
      }

      // Cuts for |eta| > 3.0
      if (std::abs(eta) > 3.0) {
        if (nef < NEF_FW_)
          return true;
      }
    }

    return false;
  }

private:  // member variables
  int nConstituents_ = 0;
  double CHF_ = 0.0;
  double NHF_ = 0.0;
  double NEF_ = 0.0;
  double MUF_ = 0.0;
  double NCH_ = 0.0;
  double NEF_TR_ = 0.0;
  double MUF_TR_ = 0.0;
  double NEF_EC_ = 0.0;
  double NEF_FW_ = 0.0;

  void initCuts() {
    if (quality_ == TIGHT) {
      if (version_ == RUN3Scouting) {
        nConstituents_ = 1;
        CHF_ = 0.01;
        NHF_ = 0.99;
        NEF_ = 0.9;
        NCH_ = 0.0;
        NEF_TR_ = 0.9;
        NEF_EC_ = 0.9;
        NEF_FW_ = 0.2;
      }
    } else if (quality_ == TIGHTLEPVETO) {
      if (version_ == RUN3Scouting) {
        CHF_ = 0.01;
        nConstituents_ = 1;
        CHF_ = 0.01;
        NHF_ = 0.99;
        NEF_ = 0.9;
        NCH_ = 0.0;
        MUF_ = 0.8;
        NEF_TR_ = 0.9;
        MUF_TR_ = 0.8;
        NEF_EC_ = 0.9;
        NEF_FW_ = 0.2;
      }
    }
  }

  Version_t version_;
  Quality_t quality_;
};

#endif
