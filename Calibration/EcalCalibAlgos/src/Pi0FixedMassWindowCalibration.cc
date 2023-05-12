#include "Calibration/EcalCalibAlgos/interface/Pi0FixedMassWindowCalibration.h"

// System include files

// Framework

// Conditions database

#include "Calibration/Tools/interface/Pi0CalibXMLwriter.h"

// Reconstruction Classes
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
// Geometry
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

// EgammaCoreTools
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"

#include "CommonTools/Utils/interface/StringToEnumValue.h"

//const double Pi0Calibration::PDGPi0Mass =  0.1349766;

using namespace std;

//_____________________________________________________________________________

Pi0FixedMassWindowCalibration::Pi0FixedMassWindowCalibration(const edm::ParameterSet& iConfig)
    : theMaxLoops(iConfig.getUntrackedParameter<unsigned int>("maxLoops", 0)),
      ecalHitsProducer_(iConfig.getParameter<std::string>("ecalRecHitsProducer")),
      barrelHits_(iConfig.getParameter<std::string>("barrelHitCollection")),
      recHitToken_(consumes<EcalRecHitCollection>(edm::InputTag(ecalHitsProducer_, barrelHits_))),
      intercalibConstantsToken_(esConsumes()),
      geometryToken_(esConsumes()) {
  std::cout << "[Pi0FixedMassWindowCalibration] Constructor " << std::endl;
  // The verbosity level
  std::string verbosityString = iConfig.getParameter<std::string>("VerbosityLevel");
  if (verbosityString == "DEBUG")
    verbosity = IslandClusterAlgo::pDEBUG;
  else if (verbosityString == "WARNING")
    verbosity = IslandClusterAlgo::pWARNING;
  else if (verbosityString == "INFO")
    verbosity = IslandClusterAlgo::pINFO;
  else
    verbosity = IslandClusterAlgo::pERROR;

  // The names of the produced cluster collections
  barrelClusterCollection_ = iConfig.getParameter<std::string>("barrelClusterCollection");

  // Island algorithm parameters
  double barrelSeedThreshold = iConfig.getParameter<double>("IslandBarrelSeedThr");
  double endcapSeedThreshold = iConfig.getParameter<double>("IslandEndcapSeedThr");

  // Selection algorithm parameters
  selePi0PtGammaOneMin_ = iConfig.getParameter<double>("selePi0PtGammaOneMin");
  selePi0PtGammaTwoMin_ = iConfig.getParameter<double>("selePi0PtGammaTwoMin");

  selePi0DRBelt_ = iConfig.getParameter<double>("selePi0DRBelt");
  selePi0DetaBelt_ = iConfig.getParameter<double>("selePi0DetaBelt");

  selePi0PtPi0Min_ = iConfig.getParameter<double>("selePi0PtPi0Min");

  selePi0S4S9GammaOneMin_ = iConfig.getParameter<double>("selePi0S4S9GammaOneMin");
  selePi0S4S9GammaTwoMin_ = iConfig.getParameter<double>("selePi0S4S9GammaTwoMin");
  selePi0S9S25GammaOneMin_ = iConfig.getParameter<double>("selePi0S9S25GammaOneMin");
  selePi0S9S25GammaTwoMin_ = iConfig.getParameter<double>("selePi0S9S25GammaTwoMin");

  selePi0EtBeltIsoRatioMax_ = iConfig.getParameter<double>("selePi0EtBeltIsoRatioMax");

  selePi0MinvMeanFixed_ = iConfig.getParameter<double>("selePi0MinvMeanFixed");
  selePi0MinvSigmaFixed_ = iConfig.getParameter<double>("selePi0MinvSigmaFixed");

  // Parameters for the position calculation:
  edm::ParameterSet posCalcParameters = iConfig.getParameter<edm::ParameterSet>("posCalcParameters");
  posCalculator_ = PositionCalc(posCalcParameters);
  shapeAlgo_ = ClusterShapeAlgo(posCalcParameters);
  clustershapecollectionEB_ = iConfig.getParameter<std::string>("clustershapecollectionEB");

  //AssociationMap
  barrelClusterShapeAssociation_ = iConfig.getParameter<std::string>("barrelShapeAssociation");

  const std::vector<std::string> seedflagnamesEB =
      iConfig.getParameter<std::vector<std::string>>("SeedRecHitFlagToBeExcludedEB");
  const std::vector<int> seedflagsexclEB = StringToEnumValue<EcalRecHit::Flags>(seedflagnamesEB);

  const std::vector<std::string> seedflagnamesEE =
      iConfig.getParameter<std::vector<std::string>>("SeedRecHitFlagToBeExcludedEE");
  const std::vector<int> seedflagsexclEE = StringToEnumValue<EcalRecHit::Flags>(seedflagnamesEE);

  const std::vector<std::string> flagnamesEB =
      iConfig.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEB");
  const std::vector<int> flagsexclEB = StringToEnumValue<EcalRecHit::Flags>(flagnamesEB);

  const std::vector<std::string> flagnamesEE =
      iConfig.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEE");
  const std::vector<int> flagsexclEE = StringToEnumValue<EcalRecHit::Flags>(flagnamesEE);

  island_p = new IslandClusterAlgo(barrelSeedThreshold,
                                   endcapSeedThreshold,
                                   posCalculator_,
                                   seedflagsexclEB,
                                   seedflagsexclEE,
                                   flagsexclEB,
                                   flagsexclEE,
                                   verbosity);

  theParameterSet = iConfig;
}

//_____________________________________________________________________________
// Close files, etc.

Pi0FixedMassWindowCalibration::~Pi0FixedMassWindowCalibration() { delete island_p; }

void Pi0FixedMassWindowCalibration::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<unsigned int>("maxLoops", 0);
  desc.add<std::string>("ecalRecHitsProducer", "");
  desc.add<std::string>("barrelHitCollection", "");

  desc.add<std::string>("VerbosityLevel", "");
  desc.add<std::string>("barrelClusterCollection", "");

  desc.add<double>("IslandBarrelSeedThr", 0);
  desc.add<double>("IslandEndcapSeedThr", 0);

  desc.add<double>("selePi0PtGammaOneMin", 0);
  desc.add<double>("selePi0PtGammaTwoMin", 0);

  desc.add<double>("selePi0DRBelt", 0);
  desc.add<double>("selePi0DetaBelt", 0);

  desc.add<double>("selePi0PtPi0Min", 0);

  desc.add<double>("selePi0S4S9GammaOneMin", 0);
  desc.add<double>("selePi0S4S9GammaTwoMin", 0);
  desc.add<double>("selePi0S9S25GammaOneMin", 0);
  desc.add<double>("selePi0S9S25GammaTwoMin", 0);

  desc.add<double>("selePi0EtBeltIsoRatioMax", 0);

  desc.add<double>("selePi0MinvMeanFixed", 0);
  desc.add<double>("selePi0MinvSigmaFixed", 0);

  edm::ParameterSetDescription posCalcParameters;
  posCalcParameters.add<bool>("LogWeighted", true);
  posCalcParameters.add<double>("T0_barl", 7.4);
  posCalcParameters.add<double>("T0_endc", 3.1);
  posCalcParameters.add<double>("T0_endcPresh", 1.2);
  posCalcParameters.add<double>("W0", 4.2);
  posCalcParameters.add<double>("X0", 0.89);
  desc.add<edm::ParameterSetDescription>("posCalcParameters", posCalcParameters);

  desc.add<std::string>("clustershapecollectionEB", "islandBarrelShape");
  desc.add<std::string>("barrelShapeAssociation", "islandBarrelShapeAssoc");

  desc.add<std::vector<std::string>>("SeedRecHitFlagToBeExcludedEB", {});
  desc.add<std::vector<std::string>>("SeedRecHitFlagToBeExcludedEE", {});
  desc.add<std::vector<std::string>>("RecHitFlagToBeExcludedEB", {});
  desc.add<std::vector<std::string>>("RecHitFlagToBeExcludedEE", {});

  descriptions.add("Pi0FixedMassWindowCalibration", desc);
}

//_____________________________________________________________________________
// Initialize algorithm

void Pi0FixedMassWindowCalibration::beginOfJob() {
  //std::cout << "[Pi0FixedMassWindowCalibration] beginOfJob "<<std::endl;

  isfirstcall_ = true;
}

void Pi0FixedMassWindowCalibration::endOfJob() {
  std::cout << "[Pi0FixedMassWindowCalibration] endOfJob" << endl;

  // Write new calibration constants

  Pi0CalibXMLwriter barrelWriter(EcalBarrel, 99);

  std::vector<DetId>::const_iterator barrelIt = barrelCells.begin();
  for (; barrelIt != barrelCells.end(); barrelIt++) {
    EBDetId eb(*barrelIt);
    int ieta = eb.ieta();
    int iphi = eb.iphi();
    int sign = eb.zside() > 0 ? 1 : 0;
    barrelWriter.writeLine(eb, newCalibs_barl[abs(ieta) - 1][iphi - 1][sign]);
  }
}

//_____________________________________________________________________________
// Called at beginning of loop
void Pi0FixedMassWindowCalibration::startingNewLoop(unsigned int iLoop) {
  for (int sign = 0; sign < 2; sign++) {
    for (int ieta = 0; ieta < 85; ieta++) {
      for (int iphi = 0; iphi < 360; iphi++) {
        wxtals[ieta][iphi][sign] = 0.;
        mwxtals[ieta][iphi][sign] = 0.;
      }
    }
  }
  std::cout << "[Pi0FixedMassWindowCalibration] Starting loop number " << iLoop << std::endl;
}

//_____________________________________________________________________________
// Called at end of loop

edm::EDLooper::Status Pi0FixedMassWindowCalibration::endOfLoop(const edm::EventSetup& iSetup, unsigned int iLoop) {
  std::cout << "[Pi0FixedMassWindowCalibration] Ending loop " << iLoop << std::endl;

  for (int sign = 0; sign < 2; sign++) {
    for (int ieta = 0; ieta < 85; ieta++) {
      for (int iphi = 0; iphi < 360; iphi++) {
        if (wxtals[ieta][iphi][sign] == 0) {
          newCalibs_barl[ieta][iphi][sign] = oldCalibs_barl[ieta][iphi][sign];
        } else {
          newCalibs_barl[ieta][iphi][sign] =
              oldCalibs_barl[ieta][iphi][sign] * (mwxtals[ieta][iphi][sign] / wxtals[ieta][iphi][sign]);
        }
        cout << " New calibration constant: ieta iphi sign - old,mwxtals,wxtals,new: " << ieta << " " << iphi << " "
             << sign << " - " << oldCalibs_barl[ieta][iphi][sign] << " " << mwxtals[ieta][iphi][sign] << " "
             << wxtals[ieta][iphi][sign] << " " << newCalibs_barl[ieta][iphi][sign] << endl;
      }
    }
  }

  Pi0CalibXMLwriter barrelWriter(EcalBarrel, iLoop + 1);

  std::vector<DetId>::const_iterator barrelIt = barrelCells.begin();
  for (; barrelIt != barrelCells.end(); barrelIt++) {
    EBDetId eb(*barrelIt);
    int ieta = eb.ieta();
    int iphi = eb.iphi();
    int sign = eb.zside() > 0 ? 1 : 0;
    barrelWriter.writeLine(eb, newCalibs_barl[abs(ieta) - 1][iphi - 1][sign]);
    if (iphi == 1) {
      std::cout << "Calib constant for barrel crystal "
                << " (" << ieta << "," << iphi << ") changed from " << oldCalibs_barl[abs(ieta) - 1][iphi - 1][sign]
                << " to " << newCalibs_barl[abs(ieta) - 1][iphi - 1][sign] << std::endl;
    }
  }

  // replace old calibration constants with new one

  for (int sign = 0; sign < 2; sign++) {
    for (int ieta = 0; ieta < 85; ieta++) {
      for (int iphi = 0; iphi < 360; iphi++) {
        oldCalibs_barl[ieta][iphi][sign] = newCalibs_barl[ieta][iphi][sign];
        newCalibs_barl[ieta][iphi][sign] = 0;
      }
    }
  }

  if (iLoop == theMaxLoops - 1 || iLoop >= theMaxLoops)
    return kStop;
  else
    return kContinue;
}

//_____________________________________________________________________________
// Called at each event

edm::EDLooper::Status Pi0FixedMassWindowCalibration::duringLoop(const edm::Event& event, const edm::EventSetup& setup) {
  using namespace edm;
  using namespace std;

  // this chunk used to belong to beginJob(isetup). Moved here
  // with the beginJob without arguments migration

  // get the ecal geometry:
  const auto& geometry = setup.getData(geometryToken_);

  if (isfirstcall_) {
    // initialize arrays

    for (int sign = 0; sign < 2; sign++) {
      for (int ieta = 0; ieta < 85; ieta++) {
        for (int iphi = 0; iphi < 360; iphi++) {
          oldCalibs_barl[ieta][iphi][sign] = 0.;
          newCalibs_barl[ieta][iphi][sign] = 0.;
          wxtals[ieta][iphi][sign] = 0.;
          mwxtals[ieta][iphi][sign] = 0.;
        }
      }
    }

    // get initial constants out of DB

    const auto& pIcal = setup.getData(intercalibConstantsToken_);
    const auto& imap = pIcal.getMap();
    std::cout << "imap.size() = " << imap.size() << std::endl;

    // loop over all barrel crystals
    barrelCells = geometry.getValidDetIds(DetId::Ecal, EcalBarrel);
    std::vector<DetId>::const_iterator barrelIt;
    for (barrelIt = barrelCells.begin(); barrelIt != barrelCells.end(); barrelIt++) {
      EBDetId eb(*barrelIt);

      // get the initial calibration constants
      EcalIntercalibConstantMap::const_iterator itcalib = imap.find(eb.rawId());
      if (itcalib == imap.end()) {
        // FIXME -- throw error
      }
      EcalIntercalibConstant calib = (*itcalib);
      int sign = eb.zside() > 0 ? 1 : 0;
      oldCalibs_barl[abs(eb.ieta()) - 1][eb.iphi() - 1][sign] = calib;
      if (eb.iphi() == 1)
        std::cout << "Read old constant for crystal "
                  << " (" << eb.ieta() << "," << eb.iphi() << ") : " << calib << std::endl;
    }
    isfirstcall_ = false;
  }

  nevent++;

  if ((nevent < 100 && nevent % 10 == 0) || (nevent < 1000 && nevent % 100 == 0) ||
      (nevent < 10000 && nevent % 100 == 0) || (nevent < 100000 && nevent % 1000 == 0) ||
      (nevent < 10000000 && nevent % 1000 == 0))
    std::cout << "[Pi0FixedMassWindowCalibration] Events processed: " << nevent << std::endl;

  recHitsEB_map = new std::map<DetId, EcalRecHit>();

  EcalRecHitCollection* recalibEcalRecHitCollection(new EcalRecHitCollection);

  Handle<EcalRecHitCollection> pEcalRecHitBarrelCollection;
  event.getByToken(recHitToken_, pEcalRecHitBarrelCollection);
  const EcalRecHitCollection* ecalRecHitBarrelCollection = pEcalRecHitBarrelCollection.product();
  cout << " ECAL Barrel RecHits # " << ecalRecHitBarrelCollection->size() << endl;
  for (EcalRecHitCollection::const_iterator aRecHitEB = ecalRecHitBarrelCollection->begin();
       aRecHitEB != ecalRecHitBarrelCollection->end();
       aRecHitEB++) {
    EBDetId ebrhdetid = aRecHitEB->detid();
    //cout << " EBDETID: z,ieta,iphi "<<ebrhdetid.zside()<<" "<<ebrhdetid.ieta()<<" "<<ebrhdetid.iphi()<<endl;
    //cout << " EBDETID: tower_ieta,tower_iphi "<<ebrhdetid.tower_ieta()<<" "<<ebrhdetid.tower_iphi()<<endl;
    //cout << " EBDETID: iSM, ic "<<ebrhdetid.ism()<<" "<<ebrhdetid.ic()<<endl;

    int sign = ebrhdetid.zside() > 0 ? 1 : 0;
    EcalRecHit aHit(aRecHitEB->id(),
                    aRecHitEB->energy() * oldCalibs_barl[abs(ebrhdetid.ieta()) - 1][ebrhdetid.iphi() - 1][sign],
                    aRecHitEB->time());
    recalibEcalRecHitCollection->push_back(aHit);
  }

  //  cout<<" Recalib size: "<<recalibEcalRecHitCollection->size()<<endl;
  for (EcalRecHitCollection::const_iterator aRecHitEB = recalibEcalRecHitCollection->begin();
       aRecHitEB != recalibEcalRecHitCollection->end();
       aRecHitEB++) {
    //    EBDetId ebrhdetid = aRecHitEB->detid();
    //cout << " [recalibrated] EBDETID: z,ieta,iphi "<<ebrhdetid.zside()<<" "<<ebrhdetid.ieta()<<" "<<ebrhdetid.iphi()<<endl;
    //cout << " [recalibrated] EBDETID: tower_ieta,tower_iphi "<<ebrhdetid.tower_ieta()<<" "<<ebrhdetid.tower_iphi()<<endl;
    //cout << " [recalibrated] EBDETID: iSM, ic "<<ebrhdetid.ism()<<" "<<ebrhdetid.ic()<<endl;

    std::pair<DetId, EcalRecHit> map_entry(aRecHitEB->id(), *aRecHitEB);
    recHitsEB_map->insert(map_entry);
  }

  const CaloSubdetectorGeometry* geometry_p;

  std::string clustershapetag;
  geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  EcalBarrelTopology topology{geometry};

  const CaloSubdetectorGeometry* geometryES_p;
  geometryES_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

  /*
  reco::BasicClusterCollection clusters;
  clusters = island_p->makeClusters(ecalRecHitBarrelCollection,geometry_p,&topology,geometryES_p,IslandClusterAlgo::barrel);
  
  //Create associated ClusterShape objects.
  std::vector <reco::ClusterShape> ClusVec;
  for (int erg=0;erg<int(clusters.size());++erg){
    reco::ClusterShape TestShape = shapeAlgo_.Calculate(clusters[erg],ecalRecHitBarrelCollection,geometry_p,&topology);
    ClusVec.push_back(TestShape);
  }

  //Put clustershapes in event, but retain a Handle on them.
  std::unique_ptr<reco::ClusterShapeCollection> clustersshapes_p(new reco::ClusterShapeCollection);
  clustersshapes_p->assign(ClusVec.begin(), ClusVec.end());

  cout<<"[Pi0Calibration] Basic Cluster collection size: "<<clusters.size()<<endl;
  cout<<"[Pi0Calibration] Basic Cluster Shape Collection size: "<<clustersshapes_p->size()<<endl;

  int iClus=0;
  for(reco::BasicClusterCollection::const_iterator aClus = clusters.begin(); aClus != clusters.end(); aClus++) {
    cout<<" CLUSTER : #,NHits,e,et,eta,phi,e2x2,e3x3,e5x5: "<<iClus<<" "<<aClus->getHitsByDetId().size()<<" "<<aClus->energy()<<" "<<aClus->energy()*sin(aClus->position().theta())<<" "<<aClus->position().eta()<<" "<<aClus->position().phi()<<" "<<(*clustersshapes_p)[iClus].e2x2()<<" "<<(*clustersshapes_p)[iClus].e3x3()<<" "<<(*clustersshapes_p)[iClus].e5x5()<<endl; 
    iClus++;
  }
  */

  // recalibrated clusters
  reco::BasicClusterCollection clusters_recalib;
  clusters_recalib = island_p->makeClusters(
      recalibEcalRecHitCollection, geometry_p, &topology, geometryES_p, IslandClusterAlgo::barrel);

  //Create associated ClusterShape objects.
  std::vector<reco::ClusterShape> ClusVec_recalib;
  for (int erg = 0; erg < int(clusters_recalib.size()); ++erg) {
    reco::ClusterShape TestShape_recalib =
        shapeAlgo_.Calculate(clusters_recalib[erg], recalibEcalRecHitCollection, geometry_p, &topology);
    ClusVec_recalib.push_back(TestShape_recalib);
  }

  //Put clustershapes in event, but retain a Handle on them.
  std::unique_ptr<reco::ClusterShapeCollection> clustersshapes_p_recalib(new reco::ClusterShapeCollection);
  clustersshapes_p_recalib->assign(ClusVec_recalib.begin(), ClusVec_recalib.end());

  cout << "[Pi0FixedMassWindowCalibration][recalibration] Basic Cluster collection size: " << clusters_recalib.size()
       << endl;
  cout << "[Pi0FixedMassWindowCalibration][recalibration] Basic Cluster Shape Collection size: "
       << clustersshapes_p_recalib->size() << endl;

  // pizero selection

  // Get ECAL Barrel Island Basic Clusters collection
  // ECAL Barrel Island Basic Clusters
  static const int MAXBCEB = 200;
  static const int MAXBCEBRH = 200;
  int nIslandBCEB;
  float eIslandBCEB[MAXBCEB];
  float etIslandBCEB[MAXBCEB];
  float etaIslandBCEB[MAXBCEB];
  float phiIslandBCEB[MAXBCEB];
  float e2x2IslandBCEB[MAXBCEB];
  float e3x3IslandBCEB[MAXBCEB];
  float e5x5IslandBCEB[MAXBCEB];
  // indexes to the RecHits assiciated with
  // ECAL Barrel Island Basic Clusters
  int nIslandBCEBRecHits[MAXBCEB];
  //  int indexIslandBCEBRecHits[MAXBCEB][MAXBCEBRH];
  int ietaIslandBCEBRecHits[MAXBCEB][MAXBCEBRH];
  int iphiIslandBCEBRecHits[MAXBCEB][MAXBCEBRH];
  int zsideIslandBCEBRecHits[MAXBCEB][MAXBCEBRH];
  float eIslandBCEBRecHits[MAXBCEB][MAXBCEBRH];

  nIslandBCEB = 0;
  for (int i = 0; i < MAXBCEB; i++) {
    eIslandBCEB[i] = 0;
    etIslandBCEB[i] = 0;
    etaIslandBCEB[i] = 0;
    phiIslandBCEB[i] = 0;
    e2x2IslandBCEB[i] = 0;
    e3x3IslandBCEB[i] = 0;
    e5x5IslandBCEB[i] = 0;
    nIslandBCEBRecHits[i] = 0;
    for (int j = 0; j < MAXBCEBRH; j++) {
      //       indexIslandBCEBRecHits[i][j] = 0;
      ietaIslandBCEBRecHits[i][j] = 0;
      iphiIslandBCEBRecHits[i][j] = 0;
      zsideIslandBCEBRecHits[i][j] = 0;
      eIslandBCEBRecHits[i][j] = 0;
    }
  }

  int iClus_recalib = 0;
  for (reco::BasicClusterCollection::const_iterator aClus = clusters_recalib.begin(); aClus != clusters_recalib.end();
       aClus++) {
    cout << " CLUSTER [recalibration] : #,NHits,e,et,eta,phi,e2x2,e3x3,e5x5: " << iClus_recalib << " " << aClus->size()
         << " " << aClus->energy() << " " << aClus->energy() * sin(aClus->position().theta()) << " "
         << aClus->position().eta() << " " << aClus->position().phi() << " "
         << (*clustersshapes_p_recalib)[iClus_recalib].e2x2() << " "
         << (*clustersshapes_p_recalib)[iClus_recalib].e3x3() << " "
         << (*clustersshapes_p_recalib)[iClus_recalib].e5x5() << endl;

    eIslandBCEB[nIslandBCEB] = aClus->energy();
    etIslandBCEB[nIslandBCEB] = aClus->energy() * sin(aClus->position().theta());
    etaIslandBCEB[nIslandBCEB] = aClus->position().eta();
    phiIslandBCEB[nIslandBCEB] = aClus->position().phi();

    e2x2IslandBCEB[nIslandBCEB] = (*clustersshapes_p_recalib)[nIslandBCEB].e2x2();
    e3x3IslandBCEB[nIslandBCEB] = (*clustersshapes_p_recalib)[nIslandBCEB].e3x3();
    e5x5IslandBCEB[nIslandBCEB] = (*clustersshapes_p_recalib)[nIslandBCEB].e5x5();

    nIslandBCEBRecHits[nIslandBCEB] = aClus->size();

    std::vector<std::pair<DetId, float>> hits = aClus->hitsAndFractions();
    std::vector<std::pair<DetId, float>>::iterator hit;
    std::map<DetId, EcalRecHit>::iterator aHit;

    int irhcount = 0;
    for (hit = hits.begin(); hit != hits.end(); hit++) {
      // need to get hit by DetID in order to get energy
      aHit = recHitsEB_map->find((*hit).first);
      //cout << "       RecHit #: "<<irhcount<<" from Basic Cluster with Energy: "<<aHit->second.energy()<<endl;

      EBDetId sel_rh = aHit->second.detid();
      //cout << "       RecHit: z,ieta,iphi "<<sel_rh.zside()<<" "<<sel_rh.ieta()<<" "<<sel_rh.iphi()<<endl;
      //cout << "       RecHit: tower_ieta,tower_iphi "<<sel_rh.tower_ieta()<<" "<<sel_rh.tower_iphi()<<endl;
      //cout << "       RecHit: iSM, ic "<<sel_rh.ism()<<" "<<sel_rh.ic()<<endl;

      ietaIslandBCEBRecHits[nIslandBCEB][irhcount] = sel_rh.ieta();
      iphiIslandBCEBRecHits[nIslandBCEB][irhcount] = sel_rh.iphi();
      zsideIslandBCEBRecHits[nIslandBCEB][irhcount] = sel_rh.zside();
      eIslandBCEBRecHits[nIslandBCEB][irhcount] = aHit->second.energy();

      irhcount++;
    }
    nIslandBCEB++;
    iClus_recalib++;
  }

  // Selection, based on ECAL Barrel Basic Clusters

  if (nIslandBCEB > 1) {
    for (int i = 0; i < nIslandBCEB; i++) {
      for (int j = i + 1; j < nIslandBCEB; j++) {
        if (etIslandBCEB[i] > selePi0PtGammaOneMin_ && etIslandBCEB[j] > selePi0PtGammaOneMin_) {
          float theta_0 = 2. * atan(exp(-etaIslandBCEB[i]));
          float theta_1 = 2. * atan(exp(-etaIslandBCEB[j]));

          float p0x = eIslandBCEB[i] * sin(theta_0) * cos(phiIslandBCEB[i]);
          float p1x = eIslandBCEB[j] * sin(theta_1) * cos(phiIslandBCEB[j]);

          float p0y = eIslandBCEB[i] * sin(theta_0) * sin(phiIslandBCEB[i]);
          float p1y = eIslandBCEB[j] * sin(theta_1) * sin(phiIslandBCEB[j]);

          float p0z = eIslandBCEB[i] * cos(theta_0);
          float p1z = eIslandBCEB[j] * cos(theta_1);

          float pi0_px = p0x + p1x;
          float pi0_py = p0y + p1y;
          float pi0_pz = p0z + p1z;

          float pi0_ptot = sqrt(pi0_px * pi0_px + pi0_py * pi0_py + pi0_pz * pi0_pz);

          float pi0_theta = acos(pi0_pz / pi0_ptot);
          float pi0_eta = -log(tan(pi0_theta / 2));
          float pi0_phi = atan(pi0_py / pi0_px);
          //cout << " pi0_theta, pi0_eta, pi0_phi "<<pi0_theta<<" "<<pi0_eta<<" "<<pi0_phi<<endl;

          // belt isolation

          float et_belt = 0;
          for (Int_t k = 0; k < nIslandBCEB; k++) {
            if ((k != i) && (k != j)) {
              float dr_pi0_k = sqrt((etaIslandBCEB[k] - pi0_eta) * (etaIslandBCEB[k] - pi0_eta) +
                                    (phiIslandBCEB[k] - pi0_phi) * (phiIslandBCEB[k] - pi0_phi));
              float deta_pi0_k = fabs(etaIslandBCEB[k] - pi0_eta);
              if ((dr_pi0_k < selePi0DRBelt_) && (deta_pi0_k < selePi0DetaBelt_))
                et_belt = et_belt + etIslandBCEB[k];
            }
          }

          float pt_pi0 = sqrt((p0x + p1x) * (p0x + p1x) + (p0y + p1y) * (p0y + p1y));
          //float dr_pi0 = sqrt ( (etaIslandBCEB[i]-etaIslandBCEB[j])*(etaIslandBCEB[i]-etaIslandBCEB[j]) + (phiIslandBCEB[i]-phiIslandBCEB[j])*(phiIslandBCEB[i]-phiIslandBCEB[j]) );

          //cout <<" pi0 pt,dr:  "<<pt_pi0<<" "<<dr_pi0<<endl;
          if (pt_pi0 > selePi0PtPi0Min_) {
            float m_inv = sqrt((eIslandBCEB[i] + eIslandBCEB[j]) * (eIslandBCEB[i] + eIslandBCEB[j]) -
                               (p0x + p1x) * (p0x + p1x) - (p0y + p1y) * (p0y + p1y) - (p0z + p1z) * (p0z + p1z));
            cout << " pi0 (pt>2.5 GeV) m_inv = " << m_inv << endl;

            float s4s9_1 = e2x2IslandBCEB[i] / e3x3IslandBCEB[i];
            float s4s9_2 = e2x2IslandBCEB[j] / e3x3IslandBCEB[j];

            float s9s25_1 = e3x3IslandBCEB[i] / e5x5IslandBCEB[i];
            float s9s25_2 = e3x3IslandBCEB[j] / e5x5IslandBCEB[j];

            //float s9Esc_1 = e3x3IslandBCEB[i]/eIslandBCEB[i];
            //float s9Esc_2 = e3x3IslandBCEB[j]/eIslandBCEB[j];

            if (s4s9_1 > selePi0S4S9GammaOneMin_ && s4s9_2 > selePi0S4S9GammaTwoMin_ &&
                s9s25_1 > selePi0S9S25GammaOneMin_ && s9s25_2 > selePi0S9S25GammaTwoMin_ &&
                (et_belt / pt_pi0 < selePi0EtBeltIsoRatioMax_)) {
              //good pizero candidate
              if (m_inv > (selePi0MinvMeanFixed_ - 2 * selePi0MinvSigmaFixed_) &&
                  m_inv < (selePi0MinvMeanFixed_ + 2 * selePi0MinvSigmaFixed_)) {
                //fill wxtals and mwxtals weights
                cout << " Pi0 Good candidate : minv = " << m_inv << endl;
                for (int kk = 0; kk < nIslandBCEBRecHits[i]; kk++) {
                  int ieta_xtal = ietaIslandBCEBRecHits[i][kk];
                  int iphi_xtal = iphiIslandBCEBRecHits[i][kk];
                  int sign = zsideIslandBCEBRecHits[i][kk] > 0 ? 1 : 0;
                  wxtals[abs(ieta_xtal) - 1][iphi_xtal - 1][sign] =
                      wxtals[abs(ieta_xtal) - 1][iphi_xtal - 1][sign] + eIslandBCEBRecHits[i][kk] / e3x3IslandBCEB[i];
                  mwxtals[abs(ieta_xtal) - 1][iphi_xtal - 1][sign] =
                      mwxtals[abs(ieta_xtal) - 1][iphi_xtal - 1][sign] +
                      (selePi0MinvMeanFixed_ / m_inv) * (selePi0MinvMeanFixed_ / m_inv) *
                          (eIslandBCEBRecHits[i][kk] / e3x3IslandBCEB[i]);
                  cout << "[Pi0FixedMassWindowCalibration] eta, phi, sign, e, e3x3, wxtals and mwxtals: " << ieta_xtal
                       << " " << iphi_xtal << " " << sign << " " << eIslandBCEBRecHits[i][kk] << " "
                       << e3x3IslandBCEB[i] << " " << wxtals[abs(ieta_xtal) - 1][iphi_xtal - 1][sign] << " "
                       << mwxtals[abs(ieta_xtal) - 1][iphi_xtal - 1][sign] << endl;
                }

                for (int kk = 0; kk < nIslandBCEBRecHits[j]; kk++) {
                  int ieta_xtal = ietaIslandBCEBRecHits[j][kk];
                  int iphi_xtal = iphiIslandBCEBRecHits[j][kk];
                  int sign = zsideIslandBCEBRecHits[j][kk] > 0 ? 1 : 0;
                  wxtals[abs(ieta_xtal) - 1][iphi_xtal - 1][sign] =
                      wxtals[abs(ieta_xtal) - 1][iphi_xtal - 1][sign] + eIslandBCEBRecHits[j][kk] / e3x3IslandBCEB[j];
                  mwxtals[abs(ieta_xtal) - 1][iphi_xtal - 1][sign] =
                      mwxtals[abs(ieta_xtal) - 1][iphi_xtal - 1][sign] +
                      (selePi0MinvMeanFixed_ / m_inv) * (selePi0MinvMeanFixed_ / m_inv) *
                          (eIslandBCEBRecHits[j][kk] / e3x3IslandBCEB[j]);
                  cout << "[Pi0FixedMassWindowCalibration] eta, phi, sign, e, e3x3, wxtals and mwxtals: " << ieta_xtal
                       << " " << iphi_xtal << " " << sign << " " << eIslandBCEBRecHits[j][kk] << " "
                       << e3x3IslandBCEB[j] << " " << wxtals[abs(ieta_xtal) - 1][iphi_xtal - 1][sign] << " "
                       << mwxtals[abs(ieta_xtal) - 1][iphi_xtal - 1][sign] << endl;
                }
              }
            }
          }
        }

      }  // End of the "j" loop over BCEB
    }    // End of the "i" loop over BCEB

  } else {
    cout << " Not enough ECAL Barrel Basic Clusters: " << nIslandBCEB << endl;
  }

  return kContinue;
}

// ----------------------------------------------------------------------------
