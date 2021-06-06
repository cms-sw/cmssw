#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "RecoEcal/EgammaClusterProducers/interface/EcalBasicClusterLocalContCorrection.h"

#include "TVector2.h"

EcalBasicClusterLocalContCorrection::EcalBasicClusterLocalContCorrection(edm::ConsumesCollector &&cc)
    : paramsToken_{cc.esConsumes()}, caloGeometryToken_{cc.esConsumes()} {}

void EcalBasicClusterLocalContCorrection::init(const edm::EventSetup &es) {
  params_ = &es.getData(paramsToken_);
  caloGeometry_ = &es.getData(caloGeometryToken_);
}

void EcalBasicClusterLocalContCorrection::checkInit() const {
  if (!params_) {
    // non-initialized function parameters: throw exception
    throw cms::Exception("EcalBasicClusterLocalContCorrection::checkInit()")
        << "Trying to access an uninitialized crack correction function.\n"
           "Please call `init( edm::EventSetup &)' before any use of the function.\n";
  }
}

using namespace std;
using namespace edm;

float EcalBasicClusterLocalContCorrection::operator()(const reco::BasicCluster &basicCluster,
                                                      const EcalRecHitCollection &recHit) const {
  checkInit();

  // number of parameters needed by this parametrization
  constexpr size_t nparams = 24;

  //correction factor to be returned, and to be calculated in this present function:
  double correction_factor = 1.;
  double fetacor = 1.;  //eta dependent part of the correction factor
  double fphicor = 1.;  //phi dependent part of the correction factor

  //--------------if barrel calculate local position wrt xtal center -------------------
  const CaloSubdetectorGeometry *geom =
      caloGeometry_->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);  //EcalBarrel = 1

  const math::XYZPoint &position_ = basicCluster.position();
  double Theta = -position_.theta() + 0.5 * M_PI;
  double Eta = position_.eta();
  double Phi = TVector2::Phi_mpi_pi(position_.phi());

  //Calculate expected depth of the maximum shower from energy (like in PositionCalc::Calculate_Location()):
  // The parameters X0 and T0 are hardcoded here because these values were used to calculate the corrections:
  constexpr float X0 = 0.89;
  constexpr float T0 = 7.4;
  double depth = X0 * (T0 + log(basicCluster.energy()));

  //search which crystal is closest to the cluster position and call it crystalseed:
  //std::vector<DetId> crystals_vector = *scRef.getHitsByDetId();   //deprecated
  std::vector<std::pair<DetId, float> > crystals_vector = basicCluster.hitsAndFractions();
  float dphimin = 999.;
  float detamin = 999.;
  int ietaclosest = 0;
  int iphiclosest = 0;

  for (unsigned int icry = 0; icry != crystals_vector.size(); ++icry) {
    EBDetId crystal(crystals_vector[icry].first);
    auto cell = geom->getGeometry(crystal);  // problema qui
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

  //PHI
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

  //ETA
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

  //-------------- end calculate local position -------------

  size_t payloadsize = params_->params().size();

  if (payloadsize < nparams)
    edm::LogError("Invalid Payload") << "Parametrization requires " << nparams << " parameters but only " << payloadsize
                                     << " are found in DB. Perhaps incompatible Global Tag" << std::endl;

  if (payloadsize > nparams)
    edm::LogWarning("Size mismatch ") << "Parametrization requires " << nparams << " parameters but " << payloadsize
                                      << " are found in DB. Perhaps incompatible Global Tag" << std::endl;

  std::pair<double, double> localPosition(EtaCry, PhiCry);

  //--- local cluster coordinates
  float localEta = localPosition.first;
  float localPhi = localPosition.second;

  //--- ecal module
  int imod = getEcalModule(basicCluster.seed());

  //-- corrections parameters
  float pe[3], pp[3];
  pe[0] = (params_->params())[0 + imod * 3];
  pe[1] = (params_->params())[1 + imod * 3];
  pe[2] = (params_->params())[2 + imod * 3];
  pp[0] = (params_->params())[12 + imod * 3];
  pp[1] = (params_->params())[13 + imod * 3];
  pp[2] = (params_->params())[14 + imod * 3];

  //--- correction vs local eta
  fetacor = pe[0] + pe[1] * localEta + pe[2] * localEta * localEta;

  //--- correction vs local phi
  fphicor = pp[0] + pp[1] * localPhi + pp[2] * localPhi * localPhi;

  //if the seed crystal is neighbourgh of a supermodule border, don't apply the phi dependent  containment corrections, but use the larger crack corrections instead.
  int iphimod20 = std::abs(iphiclosest % 20);
  if (iphimod20 <= 1)
    fphicor = 1.;

  correction_factor = (1. / fetacor) * (1. / fphicor);

  //return the correction factor. Use it to multiply the cluster energy.
  return correction_factor;
}

//------------------------------------------------------------------------------------------------------
int EcalBasicClusterLocalContCorrection::getEcalModule(DetId id) const {
  int mod = 0;
  int ieta = (EBDetId(id)).ieta();

  if (fabs(ieta) <= 25)
    mod = 0;
  if (fabs(ieta) > 25 && fabs(ieta) <= 45)
    mod = 1;
  if (fabs(ieta) > 45 && fabs(ieta) <= 65)
    mod = 2;
  if (fabs(ieta) > 65 && fabs(ieta) <= 85)
    mod = 3;

  return (mod);
}
