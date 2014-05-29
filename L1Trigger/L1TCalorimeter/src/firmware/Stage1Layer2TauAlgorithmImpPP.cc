/// \class l1t::Stage1Layer2TauAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Kalanand Mishra - Fermilab
///


#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2TauAlgorithmImp.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/JetFinderMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"


using namespace std;
using namespace l1t;


Stage1Layer2TauAlgorithmImpPP::Stage1Layer2TauAlgorithmImpPP(CaloParamsStage1* params) : params_(params)
{
  jetLsb=params_->jetLsb();

  regionPUSType = params_->regionPUSType();
  regionPUSParams = params_->regionPUSParams();
  tauSeedThreshold= floor( params_->tauSeedThreshold()/jetLsb + 0.5); // convert GeV to HW units
  jetSeedThreshold= floor( params_->jetSeedThreshold()/jetLsb + 0.5); // convert GeV to HW units
  tauRelativeJetIsolationCut = params_->tauRelativeJetIsolationCut();

  double dswitchOffTauIso(100.); // value at which to switch of Tau iso requirement (GeV)
  do2x1Algo=false;

  switchOffTauIso= floor( dswitchOffTauIso/jetLsb + 0.5);  // convert GeV to HW units
}

Stage1Layer2TauAlgorithmImpPP::~Stage1Layer2TauAlgorithmImpPP(){};




void l1t::Stage1Layer2TauAlgorithmImpPP::processEvent(const std::vector<l1t::CaloEmCand> & EMCands,
						      const std::vector<l1t::CaloRegion> & regions,
						      const std::vector<l1t::Jet> * jets,
						      std::vector<l1t::Tau> * taus) {

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();



  //Region Correction will return uncorrected subregions if
  //regionPUSType is set to None in the config
  RegionCorrection(regions, EMCands, subRegions, regionPUSParams, regionPUSType);



  // ----- need to cluster jets in order to compute jet isolation ----
  std::vector<l1t::Jet> *unCorrJets = new std::vector<l1t::Jet>();
  slidingWindowJetFinder(jetSeedThreshold, subRegions, unCorrJets);

  std::vector<l1t::Tau> *preGtTaus = new std::vector<l1t::Tau>();


  for(CaloRegionBxCollection::const_iterator region = subRegions->begin();
      region != subRegions->end(); region++) {

    int regionEt = region->hwPt();
    if(regionEt < tauSeedThreshold) continue;

    double isolation;
    int associatedSecondRegionEt =
      AssociatedSecondRegionEt(region->hwEta(), region->hwPhi(),
			       *subRegions, isolation);

    int tauEt=regionEt;
    if(do2x1Algo && associatedSecondRegionEt>tauSeedThreshold) tauEt +=associatedSecondRegionEt;


    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > tauLorentz(0,0,0,0);

    l1t::Tau theTau(*&tauLorentz, tauEt, region->hwEta(), region->hwPhi());


    double jetIsolation = JetIsolation(tauEt, region->hwEta(), region->hwPhi(), *unCorrJets);
    // if (tauEt >200)
    //   cout << "tauET: " << tauEt << " tauETA: " << region->hwEta() << " tauPHI: " << region->hwPhi()
    // 	   << " jetIso: " << jetIsolation << " Cut: " << tauRelativeJetIsolationCut
    // 	   << " Seed Threshold: " << tauSeedThreshold << endl;

    if( tauEt >0 && (jetIsolation < tauRelativeJetIsolationCut || tauEt > switchOffTauIso || abs(jetIsolation-999.)<0.1 ))
      preGtTaus->push_back(theTau);
  }

  TauToGtScales(params_, preGtTaus, taus);

  delete subRegions;
  delete unCorrJets;
  delete preGtTaus;

  //the taus should be sorted, highest pT first.
  // do not truncate the tau list, GT converter handles that
  auto comp = [&](l1t::Tau i, l1t::Tau j)-> bool {
    return (i.hwPt() < j.hwPt() );
  };

  std::sort(taus->begin(), taus->end(), comp);
  std::reverse(taus->begin(), taus->end());
}







// Given a region at iphi/ieta, find the highest region in the surrounding
// regions. Also compute isolation.

int l1t::Stage1Layer2TauAlgorithmImpPP::AssociatedSecondRegionEt(int ieta, int iphi,
								 const std::vector<l1t::CaloRegion> & regions,
								 double& isolation) const {
  int highestNeighborEt = 0;
  isolation = 0;

  for(CaloRegionBxCollection::const_iterator region = regions.begin();
      region != regions.end(); region++) {

    int regionPhi = region->hwPhi();
    int regionEta = region->hwEta();
    unsigned int deltaPhi = iphi - regionPhi;
    if (std::abs(deltaPhi) == L1CaloRegionDetId::N_PHI-1)
      deltaPhi = -deltaPhi/std::abs(deltaPhi); //18 regions in phi

    unsigned int deltaEta = std::abs(ieta - regionEta);

    if ((deltaPhi + deltaEta) > 0 && deltaPhi < 2 && deltaEta < 2) {

      int regionEt = region->hwPt();
      isolation += regionEt;
      if (regionEt > highestNeighborEt) highestNeighborEt = regionEt;
    }
  }

  // set output
  return highestNeighborEt;
}





//  Compute jet isolation.
double l1t::Stage1Layer2TauAlgorithmImpPP::JetIsolation(int et, int ieta, int iphi,
							const std::vector<l1t::Jet> & jets) const {

  for(JetBxCollection::const_iterator jet = jets.begin();
      jet != jets.end(); jet++) {

    if (ieta==jet->hwEta() && iphi==jet->hwPhi()){

      //if (et >200)
      //  cout << "ISOL:  tauET: " << et << " jetET: " << jet->hwPt() << endl;

      double isolation = (double) (jet->hwPt() - et);
      return isolation/et;
    }
  }

  // set output
  return 999.;
}
