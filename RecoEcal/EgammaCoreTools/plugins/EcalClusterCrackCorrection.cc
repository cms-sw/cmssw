/** \class EcalClusterCrackCorrection
  *  Function to correct cluster for cracks in the calorimeter
  *
  *  $Id: EcalClusterCrackCorrection.h
  *  $Date:
  *  $Revision:
  *  \author Federico Ferri, CEA Saclay, November 2008
  */

#include "CondFormats/DataRecord/interface/EcalClusterCrackCorrParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterCrackCorrParameters.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "TVector2.h"

class EcalClusterCrackCorrection : public EcalClusterFunctionBaseClass {
public:
  EcalClusterCrackCorrection(const edm::ParameterSet &){};

  // get/set explicit methods for parameters
  const EcalClusterCrackCorrParameters *getParameters() const { return params_; }
  // check initialization
  void checkInit() const;

  // compute the correction
  float getValue(const reco::BasicCluster &, const EcalRecHitCollection &) const override { return 1.f; }
  float getValue(const reco::SuperCluster &, const int mode) const override;

  float getValue(const reco::CaloCluster &) const override;

  // set parameters
  void init(const edm::EventSetup &es) override;

private:
  edm::ESHandle<EcalClusterCrackCorrParameters> esParams_;
  const EcalClusterCrackCorrParameters *params_;
  const edm::EventSetup *es_;  //needed to access the ECAL geometry
};

void EcalClusterCrackCorrection::init(const edm::EventSetup &es) {
  es.get<EcalClusterCrackCorrParametersRcd>().get(esParams_);
  params_ = esParams_.product();
  es_ = &es;  //needed to access the ECAL geometry
}

void EcalClusterCrackCorrection::checkInit() const {
  if (!params_) {
    // non initialized function parameters: throw exception
    throw cms::Exception("EcalClusterCrackCorrection::checkInit()")
        << "Trying to access an uninitialized crack correction function.\n"
           "Please call `init( edm::EventSetup &)' before any use of the function.\n";
  }
}

float EcalClusterCrackCorrection::getValue(const reco::CaloCluster &seedbclus) const {
  checkInit();

  //correction factor to be returned, and to be calculated in this present function:
  double correction_factor = 1.;
  double fetacor = 1.;  //eta dependent part of the correction factor
  double fphicor = 1.;  //phi dependent part of the correction factor

  //********************************************************************************************************************//
  //These ECAL barrel module and supermodule border corrections correct a photon energy for leakage outside a 5x5 crystal cluster. They  depend on the local position in the hit crystal. The hit crystal needs to be at the border of a barrel module. The local position coordinates, called later EtaCry and PhiCry in the code, are comprised between -0.5 and 0.5 and correspond to the distance between the photon supercluster position and the center of the hit crystal, expressed in number of  crystal widthes. The correction parameters (that should be filled in CalibCalorimetry/EcalTrivialCondModules/python/EcalTrivialCondRetriever_cfi.py) were calculated using simulaion and thus take into account the effect of the magnetic field. They  only apply to unconverted photons in the barrel, but a use for non brem electrons could be considered (not tested yet). For more details, cf the CMS internal note 2009-013 by S. Tourneur and C. Seez

  //Beware: The user should make sure it only uses this correction factor for unconverted photons (or not breming electrons)

  //const reco::CaloClusterPtr & seedbclus =  superCluster.seed();

  //If not barrel, return 1:
  if (std::abs(seedbclus.eta()) > 1.4442)
    return 1.;

  edm::ESHandle<CaloGeometry> pG;
  es_->get<CaloGeometryRecord>().get(pG);

  const CaloSubdetectorGeometry *geom = pG->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);  //EcalBarrel = 1

  const math::XYZPoint &position_ = seedbclus.position();
  double Theta = -position_.theta() + 0.5 * M_PI;
  double Eta = position_.eta();
  double Phi = TVector2::Phi_mpi_pi(position_.phi());

  //Calculate expected depth of the maximum shower from energy (like in PositionCalc::Calculate_Location()):
  // The parameters X0 and T0 are hardcoded here because these values were used to calculate the corrections:
  const float X0 = 0.89;
  const float T0 = 7.4;
  double depth = X0 * (T0 + log(seedbclus.energy()));

  //search which crystal is closest to the cluster position and call it crystalseed:
  //std::vector<DetId> crystals_vector = seedbclus.getHitsByDetId();   //deprecated
  std::vector<std::pair<DetId, float> > crystals_vector = seedbclus.hitsAndFractions();
  float dphimin = 999.;
  float detamin = 999.;
  int ietaclosest = 0;
  int iphiclosest = 0;
  for (unsigned int icry = 0; icry != crystals_vector.size(); ++icry) {
    EBDetId crystal(crystals_vector[icry].first);
    auto cell = geom->getGeometry(crystal);
    GlobalPoint center_pos = cell->getPosition(depth);
    double EtaCentr = center_pos.eta();
    double PhiCentr = TVector2::Phi_mpi_pi(center_pos.phi());
    if (std::abs(EtaCentr - Eta) < detamin) {
      detamin = std::abs(EtaCentr - Eta);
      ietaclosest = crystal.ieta();
    }
    if (std::abs(TVector2::Phi_mpi_pi(PhiCentr - Phi)) < dphimin) {
      dphimin = std::abs(TVector2::Phi_mpi_pi(PhiCentr - Phi));
      iphiclosest = crystal.iphi();
    }
  }
  EBDetId crystalseed(ietaclosest, iphiclosest);

  // Get center cell position from shower depth
  auto cell = geom->getGeometry(crystalseed);
  GlobalPoint center_pos = cell->getPosition(depth);

  //if the seed crystal isn't neighbourgh of a supermodule border, don't apply the phi dependent crack corrections, but use the smaller phi dependent local containment correction instead.
  if (ietaclosest < 0)
    iphiclosest = 361 - iphiclosest;  //inversion of phi 3 degree tilt
  int iphimod20 = iphiclosest % 20;
  if (iphimod20 > 1)
    fphicor = 1.;

  else {
    double PhiCentr = TVector2::Phi_mpi_pi(center_pos.phi());
    double PhiWidth = (M_PI / 180.);
    double PhiCry = (TVector2::Phi_mpi_pi(Phi - PhiCentr)) / PhiWidth;
    if (PhiCry > 0.5)
      PhiCry = 0.5;
    if (PhiCry < -0.5)
      PhiCry = -0.5;
    //flip to take into account ECAL barrel symmetries:
    if (ietaclosest < 0)
      PhiCry *= -1.;

    //Fetching parameters of the polynomial (see  CMS IN-2009/013)
    double g[5];
    int offset = iphimod20 == 0 ? 10   //coefficients for one phi side of a SM
                                : 15;  //coefficients for the other side
    for (int k = 0; k != 5; ++k)
      g[k] = (params_->params())[k + offset];

    fphicor = 0.;
    for (int k = 0; k != 5; ++k)
      fphicor += g[k] * std::pow(PhiCry, k);
  }

  //if the seed crystal isn't neighbourgh of a module border, don't apply the eta dependent crack corrections, but use the smaller eta dependent local containment correction instead.
  int ietamod20 = ietaclosest % 20;
  if (std::abs(ietaclosest) < 25 || (std::abs(ietamod20) != 5 && std::abs(ietamod20) != 6))
    fetacor = 1.;

  else {
    double ThetaCentr = -center_pos.theta() + 0.5 * M_PI;
    double ThetaWidth = (M_PI / 180.) * std::cos(ThetaCentr);
    double EtaCry = (Theta - ThetaCentr) / ThetaWidth;
    if (EtaCry > 0.5)
      EtaCry = 0.5;
    if (EtaCry < -0.5)
      EtaCry = -0.5;
    //flip to take into account ECAL barrel symmetries:
    if (ietaclosest < 0)
      EtaCry *= -1.;

    //Fetching parameters of the polynomial (see  CMS IN-2009/013)
    double f[5];
    int offset = std::abs(ietamod20) == 5
                     ? 0   //coefficients for eta side of an intermodule gap closer to the interaction point
                     : 5;  //coefficients for the other eta side
    for (int k = 0; k != 5; ++k)
      f[k] = (params_->params())[k + offset];

    fetacor = 0.;
    for (int k = 0; k != 5; ++k)
      fetacor += f[k] * std::pow(EtaCry, k);
  }

  correction_factor = 1. / (fetacor * fphicor);
  //*********************************************************************************************************************//

  //return the correction factor. Use it to multiply the cluster energy.
  return correction_factor;
}

float EcalClusterCrackCorrection::getValue(const reco::SuperCluster &superCluster, const int mode) const {
  checkInit();

  //********************************************************************************************************************//
  //These ECAL barrel module and supermodule border corrections correct a photon energy for leakage outside a 5x5 crystal cluster. They  depend on the local position in the hit crystal. The hit crystal needs to be at the border of a barrel module. The local position coordinates, called later EtaCry and PhiCry in the code, are comprised between -0.5 and 0.5 and correspond to the distance between the photon supercluster position and the center of the hit crystal, expressed in number of  crystal widthes. The correction parameters (that should be filled in CalibCalorimetry/EcalTrivialCondModules/python/EcalTrivialCondRetriever_cfi.py) were calculated using simulaion and thus take into account the effect of the magnetic field. They  only apply to unconverted photons in the barrel, but a use for non brem electrons could be considered (not tested yet). For more details, cf the CMS internal note 2009-013 by S. Tourneur and C. Seez

  //Beware: The user should make sure it only uses this correction factor for unconverted photons (or not breming electrons)

  return getValue(*(superCluster.seed()));
}

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
DEFINE_EDM_PLUGIN(EcalClusterFunctionFactory, EcalClusterCrackCorrection, "EcalClusterCrackCorrection");
