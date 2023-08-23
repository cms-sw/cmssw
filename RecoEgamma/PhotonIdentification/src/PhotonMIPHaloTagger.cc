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

void PhotonMIPHaloTagger::setup(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC) {
  EBecalCollection_ = iC.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("barrelEcalRecHitCollection"));
  EEecalCollection_ = iC.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("endcapEcalRecHitCollection"));

  yRangeFit_ = conf.getParameter<double>("YRangeFit");
  xRangeFit_ = conf.getParameter<double>("XRangeFit");
  residualWidthEnergy_ = conf.getParameter<double>("ResidualWidth");
  haloDiscThreshold_ = conf.getParameter<double>("HaloDiscThreshold");
}

void PhotonMIPHaloTagger::MIPcalculate(const reco::Photon* pho,
                                       const edm::Event& e,
                                       const edm::EventSetup& es,
                                       reco::Photon::MIPVariables& mipId) {
  //get the predefined variables
  inputRangeY = yRangeFit_;
  inputRangeX = xRangeFit_;
  inputResWidth = residualWidthEnergy_;
  inputHaloDiscCut = haloDiscThreshold_;

  //First store in local variables
  mipFitResults_.clear();

  nhitCone_ = -99;     //hit inside the cone
  ismipHalo_ = false;  // halo?

  // Get EcalRecHits
  edm::Handle<EcalRecHitCollection> barrelHitHandle;
  e.getByToken(EBecalCollection_, barrelHitHandle);

  bool validEcalRecHits = barrelHitHandle.isValid();
  if (validEcalRecHits) {
    // GetMIPTrailFit
    mipFitResults_ = GetMipTrailFit(
        pho, e, es, barrelHitHandle, inputRangeY, inputRangeX, inputResWidth, inputHaloDiscCut, nhitCone_, ismipHalo_);

    //Now set the variable in "MIPVaraible"
    mipId.mipChi2 = mipFitResults_[0];
    mipId.mipTotEnergy = mipFitResults_[1];
    mipId.mipSlope = mipFitResults_[2];
    mipId.mipIntercept = mipFitResults_[3];
    mipId.mipNhitCone = nhitCone_;
    mipId.mipIsHalo = ismipHalo_;
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
}

// Get the MIP Variable NCone, Energy etc here
std::vector<double> PhotonMIPHaloTagger::GetMipTrailFit(const reco::Photon* photon,
                                                        const edm::Event& iEvent,
                                                        const edm::EventSetup& iSetup,
                                                        edm::Handle<EcalRecHitCollection> ecalhitsCollEB,
                                                        double inputRangeY,
                                                        double inputRangeX,
                                                        double inputResWidth,
                                                        double inputHaloDiscCut,
                                                        int& NhitCone_,
                                                        bool& ismipHalo_) {
  bool debug_ = false;

  if (debug_)
    std::cout << " inside MipFitTrail " << std::endl;
  //getting them from cfi config
  double yrange = inputRangeY;
  double xrange = inputRangeX;
  double res_width = inputResWidth;         //width of the residual distribution
  double halo_disc_cut = inputHaloDiscCut;  //cut based on  Shower,Angle and MIP

  //initilize them here
  double m = 0.;
  m = yrange / xrange;  //slope of the lines which form the cone around the trail

  //first get the seed cell index
  int seedIEta = -999;
  int seedIPhi = -999;
  double seedE = -999.;

  //get seed propert.
  GetSeedHighestE(photon, iEvent, iSetup, ecalhitsCollEB, seedIEta, seedIPhi, seedE);

  if (debug_)
    std::cout << "Seed E =" << seedE << "  Seed ieta = " << seedIEta << "   Seed iphi =" << seedIPhi << std::endl;

  //to store results
  std::vector<double> FitResults_;
  FitResults_.clear();

  //create some vector and clear them
  std::vector<int> ieta_cell;
  std::vector<int> iphi_cell;
  std::vector<double> energy_cell;

  ieta_cell.clear();
  iphi_cell.clear();
  energy_cell.clear();

  int ietacell = 0;
  int iphicell = 0;
  int kArray = 0;

  int delt_ieta = 0;
  int delt_iphi = 0;

  for (EcalRecHitCollection::const_iterator it = ecalhitsCollEB->begin(); it != ecalhitsCollEB->end(); ++it) {
    EBDetId dit = it->detid();

    iphicell = dit.iphi();
    ietacell = dit.ieta();

    if (ietacell < 0)
      ietacell++;

    //Exclude all cells within +/- 5 ieta of seed cell
    if (std::abs(ietacell - seedIEta) >= 5 && it->energy() > 0.) {
      delt_ieta = ietacell - seedIEta;
      delt_iphi = iphicell - seedIPhi;

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
        energy_cell.push_back(it->energy());
        kArray++;
      }

    }  //within cerntain range of seed cell

  }  //loop voer hits

  //Iterations for imporovements

  int Npoints = 0;
  int throwaway_index = -1;
  double chi2;
  double eT = 0.;
  double hres = 99999.;

  //some tmp local variale
  double Roundness_ = 999.;
  double Angle_ = 999.;
  double halo_disc_ = 0.;

  Npoints = kArray;

  if (debug_)
    std::cout << " starting npoing = " << Npoints << std::endl;

  //defined some variable for iterative fitting the mip trail line
  double sx = 0.0;
  double sy = 0.0;
  double ss = 0.0;
  double sxx = 0.0;
  double sxy = 0.0;
  double a1 = 0.0;
  double b1 = 0.0;
  double m_chi2 = 0.0;
  double etot_cell = 0.0;

  //start Iterations
  for (int it = 0; it < 200 && hres > (5.0 * res_width); it++) {  //Max iter. is 200

    //Throw away previous highest residual, if not first loop
    if (throwaway_index != -1) {
      ieta_cell.erase(ieta_cell.begin() + throwaway_index);
      iphi_cell.erase(iphi_cell.begin() + throwaway_index);
      energy_cell.erase(energy_cell.begin() + throwaway_index);
      Npoints--;
    }

    //Lets Initialize them first for each iteration
    sx = 0.0;
    sy = 0.0;
    ss = 0.0;
    sxx = 0.0;
    sxy = 0.0;
    m_chi2 = 0.0;
    etot_cell = 0.0;

    //Fit the line to trail
    for (int j = 0; j < Npoints; j++) {
      double wt = 1.0;
      ss += wt;
      sx += ieta_cell[j] * wt;
      sy += iphi_cell[j];
      sxx += ieta_cell[j] * ieta_cell[j] * wt;
      sxy += ieta_cell[j] * iphi_cell[j] * wt;
    }

    double delt = ss * sxx - (sx * sx);
    a1 = ((sxx * sy) - (sx * sxy)) / delt;  // INTERCEPT
    b1 = ((ss * sxy) - (sx * sy)) / delt;   // SLOPE

    double highest_res = 0.;
    int highres_index = 0;

    for (int j = 0; j < Npoints; j++) {
      double res = 1.0 * iphi_cell[j] - a1 - b1 * ieta_cell[j];
      double res_sq = res * res;

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

  if (debug_)
    std::cout << "hres = " << hres << std::endl;

  //get roundness and angle for this photon candidate form EcalClusterTool
  std::vector<float> showershapes_barrel =
      EcalClusterTools::roundnessBarrelSuperClusters(*(photon->superCluster()), (*ecalhitsCollEB.product()));

  Roundness_ = showershapes_barrel[0];
  Angle_ = showershapes_barrel[1];

  if (debug_)
    std::cout << " eTot =" << eT << "     Rounness = " << Roundness_ << "    Angle_  " << Angle_ << std::endl;

  //get the halo disc variable
  halo_disc_ = eT / (Roundness_ * Angle_);

  ///Now Filll the FitResults vector
  FitResults_.push_back(chi2);  //chi2
  FitResults_.push_back(eT);    //total energy
  FitResults_.push_back(b1);    //slope
  FitResults_.push_back(a1);    //intercept
  NhitCone_ = Npoints;          //nhit in cone
  if (halo_disc_ > halo_disc_cut)
    ismipHalo_ = true;  //is halo?, yes if halo_disc > 70 by default
                        // based on 2010 study

  if (debug_)
    std::cout << "Endof MIP Trail: halo_dic= " << halo_disc_ << "   nhitcone =" << NhitCone_
              << "  isHalo= " << ismipHalo_ << std::endl;
  if (debug_)
    std::cout << "Endof MIP Trail: Chi2  = " << chi2 << "  eT= " << eT << " slope = " << b1 << "  Intercept =" << a1
              << std::endl;

  return FitResults_;
}

//get the seed crystal index
void PhotonMIPHaloTagger::GetSeedHighestE(const reco::Photon* photon,
                                          const edm::Event& iEvent,
                                          const edm::EventSetup& iSetup,
                                          edm::Handle<EcalRecHitCollection> Brechit,
                                          int& seedIeta,
                                          int& seedIphi,
                                          double& seedEnergy) {
  bool debug_ = false;

  if (debug_)
    std::cout << "Inside GetSeed" << std::endl;
  //Get the Seed
  double SeedE = -999.;

  //initilaze them here
  seedIeta = -999;
  seedIphi = -999;
  seedEnergy = -999.;

  std::vector<std::pair<DetId, float> > PhotonHit_DetIds = photon->superCluster()->hitsAndFractions();
  std::vector<std::pair<DetId, float> >::const_iterator detitr;
  for (detitr = PhotonHit_DetIds.begin(); detitr != PhotonHit_DetIds.end(); ++detitr) {
    if (((*detitr).first).det() == DetId::Ecal && ((*detitr).first).subdetId() == EcalBarrel) {
      EcalRecHitCollection::const_iterator j = Brechit->find(((*detitr).first));
      EcalRecHitCollection::const_iterator thishit;

      if (j != Brechit->end())
        thishit = j;
      if (j == Brechit->end()) {
        continue;
      }

      EBDetId detId = (EBDetId)((*detitr).first);

      double crysE = thishit->energy();

      if (crysE > SeedE) {
        SeedE = crysE;
        seedIeta = detId.ieta();
        seedIphi = (detId.iphi());
        seedEnergy = SeedE;

        if (debug_)
          std::cout << "Current max Seed = " << SeedE << "   seedIphi = " << seedIphi << "  ieta= " << seedIeta
                    << std::endl;
      }

    }  //check if in Barrel

  }  //loop over EBrechits cells

  if (debug_)
    std::cout << "End of  GetSeed" << std::endl;
}
