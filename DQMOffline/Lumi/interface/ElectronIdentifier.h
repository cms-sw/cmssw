#ifndef DQMOFFLINE_LUMI_ELECTRONIDENTIFIER_H
#define DQMOFFLINE_LUMI_ELECTRONIDENTIFIER_H

#include "FWCore/Framework/interface/MakerMacros.h"   // definitions for declaring plug-in modules
#include "FWCore/Framework/interface/Frameworkfwd.h"  // declaration of EDM types
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"  // Parameters
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>  // string class
#include <TMath.h>
#include <cassert>

#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

#include "CommonTools/Egamma/interface/EffectiveAreas.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

enum EleIDCutNames {
  SIGMAIETA,
  DETAINSEED,
  DPHIIN,
  HOVERE,
  ISO,
  ONEOVERE,
  MISSINGHITS,
  CONVERSION,
};
enum EleIDWorkingPoints { VETO, MEDIUM, LOOSE, TIGHT };
enum EleIDEtaBins { BARREL, ENDCAP };
class ElectronIdentifier {
public:
  ElectronIdentifier(const edm::ParameterSet& c);
  float dEtaInSeed(const reco::GsfElectronPtr& ele);
  bool passID(const reco::GsfElectronPtr& ele,
              edm::Handle<reco::BeamSpot> beamspot,
              edm::Handle<reco::ConversionCollection> conversions);
  float isolation(const reco::GsfElectronPtr& ele);

  void setID(std::string ID);
  void setRho(double rho);

private:
  double rho_;
  int ID_;
  std::array<std::array<std::array<double, 2>, 4>, 8> cuts_;
  // Effective area constants
  EffectiveAreas _effectiveAreas;
};

#endif
