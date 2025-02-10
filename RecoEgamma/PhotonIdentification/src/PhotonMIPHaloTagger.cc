/** \class PhotonMIPHaloTragger
 *  Determine and Set whether a haol or not
 *
 *  \author Sushil S. Chauhan, A. Nowack, M. Tripathi : (UCDavis) 
 *  \author A. Askew (FSU)
 */

#include "RecoEgamma/PhotonIdentification/interface/PhotonMIPHaloTagger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

PhotonMIPHaloTagger::PhotonMIPHaloTagger(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC)
    : EBecalCollection_{iC.consumes<EcalRecHitCollection>(
          conf.getParameter<edm::InputTag>("barrelEcalRecHitCollection"))},
      yRangeFit_{conf.getParameter<double>("YRangeFit")},
      xRangeFit_{conf.getParameter<double>("XRangeFit")},
      residualWidthEnergy_{conf.getParameter<double>("ResidualWidth")},
      haloDiscThreshold_{conf.getParameter<double>("HaloDiscThreshold")} {}

reco::Photon::MIPVariables PhotonMIPHaloTagger::mipCalculate(const reco::Photon& pho,
                                                             const edm::Event& e,
                                                             const edm::EventSetup& es) const {
  // Get EcalRecHits
  edm::Handle<EcalRecHitCollection> barrelHitHandle;
  e.getByToken(EBecalCollection_, barrelHitHandle);

  bool validEcalRecHits = barrelHitHandle.isValid();

  reco::Photon::MIPVariables mipId;
  if (validEcalRecHits) {
    mipId =
        getMipTrailFit(pho, e, es, barrelHitHandle, yRangeFit_, xRangeFit_, residualWidthEnergy_, haloDiscThreshold_);
  } else {
    edm::LogError("MIPcalculate") << "Error! Can't get the barrel hits product, hence set some default values";
    mipId.mipChi2 = -999.;
    mipId.mipTotEnergy = -999.;
    mipId.mipSlope = -999.;
    mipId.mipIntercept = -999.;
    mipId.mipNhitCone = 0;
    mipId.mipIsHalo = false;
  }

  // std::cout<<"Setting: isHalo = "<<ismipHalo_<<"    nhitcone=  "<<nhitCone_<<std::endl;
  // std::cout<<"Setting: Chi2  = "<<mipFitResults_[0]<<"  eT= "<<mipFitResults_[1]<<" slope = "<<mipFitResults_[2]<<"  Intercept ="<<mipFitResults_[3]<<std::endl;
  return mipId;
}

// Get the MIP Variable NCone, Energy etc here
reco::Photon::MIPVariables PhotonMIPHaloTagger::getMipTrailFit(const reco::Photon& photon,
                                                               const edm::Event& iEvent,
                                                               const edm::EventSetup& iSetup,
                                                               edm::Handle<EcalRecHitCollection> ecalhitsCollEB,
                                                               double inputRangeY,
                                                               double inputRangeX,
                                                               double inputResWidth,
                                                               double inputHaloDiscCut) const {
  constexpr bool printDebug = false;

  if constexpr (printDebug)
    std::cout << " inside MipFitTrail " << std::endl;
  //getting them from cfi config
  const double yrange = inputRangeY;
  const double xrange = inputRangeX;
  const double res_width = inputResWidth;         //width of the residual distribution
  const double halo_disc_cut = inputHaloDiscCut;  //cut based on  Shower,Angle and MIP

  //initilize them here
  const double m = yrange / xrange;  //slope of the lines which form the cone around the trail

  //first get the seed cell index
  int seedIEta = -999;
  int seedIPhi = -999;
  double seedE = -999.;

  //get seed propert.
  getSeedHighestE(photon, iEvent, iSetup, ecalhitsCollEB, seedIEta, seedIPhi, seedE);

  if constexpr (printDebug)
    std::cout << "Seed E =" << seedE << "  Seed ieta = " << seedIEta << "   Seed iphi =" << seedIPhi << std::endl;

  //create some vector
  std::vector<int> ieta_cell;
  std::vector<int> iphi_cell;
  std::vector<double> energy_cell;

  for (auto const& hit : *ecalhitsCollEB) {
    const EBDetId dit = hit.detid();

    const int iphicell = dit.iphi();
    int ietacell = dit.ieta();

    if (ietacell < 0)
      ietacell++;

    //Exclude all cells within +/- 5 ieta of seed cell
    if (std::abs(ietacell - seedIEta) >= 5 && hit.energy() > 0.) {
      int delt_ieta = ietacell - seedIEta;
      int delt_iphi = iphicell - seedIPhi;

      //Phi wrapping inclusion
      if (delt_iphi > 180) {
        delt_iphi = delt_iphi - 360;
      }
      if (delt_iphi < -180) {
        delt_iphi = delt_iphi + 360;
      }

      //Condition to be within the cones
      if (((delt_iphi >= (m * delt_ieta)) && (delt_iphi <= (-m * delt_ieta))) ||
          ((delt_iphi <= (m * delt_ieta)) && (delt_iphi >= (-m * delt_ieta)))) {
        ieta_cell.push_back(delt_ieta);
        iphi_cell.push_back(delt_iphi);
        energy_cell.push_back(hit.energy());
      }

    }  //within certain range of seed cell

  }  //loop over hits

  //Iterations for improvements

  int Npoints = ieta_cell.size();
  int throwaway_index = -1;
  double chi2 = 0.0;
  double eT = 0.;
  double a1 = 0.;
  double b1 = 0.;
  double hres = 99999.;

  if constexpr (printDebug)
    std::cout << " starting npoints = " << Npoints << std::endl;

  //start Iterations
  for (int it = 0; it < 200 && hres > (5.0 * res_width); it++) {  //Max iter. is 200

    //Throw away previous highest residual, if not first loop
    if (throwaway_index != -1) {
      ieta_cell.erase(ieta_cell.begin() + throwaway_index);
      iphi_cell.erase(iphi_cell.begin() + throwaway_index);
      energy_cell.erase(energy_cell.begin() + throwaway_index);
      Npoints--;
    }

    //defined some variable for iterative fitting the mip trail line
    double sx = 0.0;
    double sy = 0.0;
    double ss = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    double m_chi2 = 0.0;
    double etot_cell = 0.0;

    //Fit the line to trail
    for (int j = 0; j < Npoints; j++) {
      constexpr double wt = 1.0;
      ss += wt;
      sx += ieta_cell[j] * wt;
      sy += iphi_cell[j];
      sxx += ieta_cell[j] * ieta_cell[j] * wt;
      sxy += ieta_cell[j] * iphi_cell[j] * wt;
    }

    const double delt = ss * sxx - (sx * sx);
    a1 = ((sxx * sy) - (sx * sxy)) / delt;  // INTERCEPT
    b1 = ((ss * sxy) - (sx * sy)) / delt;   // SLOPE

    double highest_res = 0.;
    int highres_index = 0;

    for (int j = 0; j < Npoints; j++) {
      const double res = 1.0 * iphi_cell[j] - a1 - b1 * ieta_cell[j];
      const double res_sq = res * res;

      if (std::abs(res) > highest_res) {
        highest_res = std::abs(res);
        highres_index = j;
      }

      m_chi2 += res_sq;
      etot_cell += energy_cell[j];
    }

    throwaway_index = highres_index;
    hres = highest_res;

    chi2 = m_chi2 / ((Npoints - 2));
    chi2 = chi2 / (res_width * res_width);
    eT = etot_cell;

  }  //for loop for iterations

  if constexpr (printDebug)
    std::cout << "hres = " << hres << std::endl;

  //get roundness and angle for this photon candidate form EcalClusterTool
  const std::vector<float> showershapes_barrel =
      EcalClusterTools::roundnessBarrelSuperClusters(*(photon.superCluster()), (*ecalhitsCollEB.product()));

  const double roundness = showershapes_barrel[0];
  const double angle = showershapes_barrel[1];

  if constexpr (printDebug)
    std::cout << " eTot =" << eT << "     Rounness = " << roundness << "    angle  " << angle << std::endl;

  //get the halo disc variable
  const double halo_disc = eT / (roundness * angle);

  ///Now Fill the FitResults vector
  reco::Photon::MIPVariables results;
  results.mipChi2 = chi2;
  results.mipTotEnergy = eT;
  results.mipSlope = b1;
  results.mipIntercept = a1;
  results.mipNhitCone = Npoints;
  results.mipIsHalo = halo_disc > halo_disc_cut;
  //is halo?, yes if halo_disc > 70 by default
  // based on 2010 study

  if constexpr (printDebug)
    std::cout << "Endof MIP Trail: halo_dic= " << halo_disc << "   nhitcone =" << results.mipNhitCone
              << "  isHalo= " << results.mipIsHalo << std::endl;
  if constexpr (printDebug)
    std::cout << "Endof MIP Trail: Chi2  = " << chi2 << "  eT= " << eT << " slope = " << b1 << "  Intercept =" << a1
              << std::endl;

  return results;
}

//get the seed crystal index
void PhotonMIPHaloTagger::getSeedHighestE(const reco::Photon& photon,
                                          const edm::Event& iEvent,
                                          const edm::EventSetup& iSetup,
                                          edm::Handle<EcalRecHitCollection> Brechit,
                                          int& seedIeta,
                                          int& seedIphi,
                                          double& seedEnergy) const {
  constexpr bool printDebug = false;

  if constexpr (printDebug)
    std::cout << "Inside GetSeed" << std::endl;
  //Get the Seed
  double SeedE = -999.;

  //initilaze them here
  seedIeta = -999;
  seedIphi = -999;
  seedEnergy = -999.;

  std::vector<std::pair<DetId, float> > const& PhotonHit_DetIds = photon.superCluster()->hitsAndFractions();
  for (auto pr : PhotonHit_DetIds) {
    if ((pr.first).det() == DetId::Ecal && (pr.first).subdetId() == EcalBarrel) {
      EcalRecHitCollection::const_iterator thishit = Brechit->find((pr.first));

      if (thishit == Brechit->end()) {
        continue;
      }

      const EBDetId detId{pr.first};

      const double crysE = thishit->energy();

      if (crysE > SeedE) {
        SeedE = crysE;
        seedIeta = detId.ieta();
        seedIphi = (detId.iphi());
        seedEnergy = SeedE;

        if constexpr (printDebug)
          std::cout << "Current max Seed = " << SeedE << "   seedIphi = " << seedIphi << "  ieta= " << seedIeta
                    << std::endl;
      }

    }  //check if in Barrel

  }  //loop over EBrechits cells

  if constexpr (printDebug)
    std::cout << "End of  GetSeed" << std::endl;
}
