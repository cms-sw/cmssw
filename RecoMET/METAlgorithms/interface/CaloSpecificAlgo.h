// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      CaloSpecificAlgo
// 
/**\class CaloSpecificAlgo CaloSpecificAlgo.h RecoMET/METAlgorithms/interface/CaloSpecificAlgo.h

 Description: Adds Calorimeter specific information to MET base class

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  R. Cavanaugh (taken from F.Ratnikov, UMd)
//         Created:  June 6, 2006
// $Id: CaloSpecificAlgo.h,v 1.10 2012/06/09 21:19:30 sakuma Exp $
//
//
#ifndef METProducers_CaloMETInfo_h
#define METProducers_CaloMETInfo_h

//____________________________________________________________________________||
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/SpecificCaloMETData.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>

class CaloTower;
struct SpecificCaloMETData;

//____________________________________________________________________________||
class CaloSpecificAlgo 
{

 public:
  reco::CaloMET addInfo(edm::Handle<edm::View<reco::Candidate> > towers, CommonMETData met, bool noHF, double globalThreshold);

 private:
  typedef math::XYZTLorentzVector LorentzVector;
  typedef math::XYZPoint Point;
  void initializeSpecificCaloMETData(SpecificCaloMETData &specific);
  void update_totalEt_totalEm(double &totalEt, double& totalEm, const CaloTower* calotower, bool noHF);
  void update_MaxTowerEm_MaxTowerHad(double &MaxTowerEm, double &MaxTowerHad, const CaloTower* calotower, bool noHF);
  void update_EmEtInEB_EmEtInEE(double &EmEtInEB, double &EmEtInEE, const CaloTower* calotower);
  void update_HadEtInHB_HadEtInHE_HadEtInHO_HadEtInHF_EmEtInHF(double &HadEtInHB, double &HadEtInHE, double &HadEtInHO, double &HadEtInHF, double &EmEtInHF, const CaloTower* calotower, bool noHF);
  void update_sumEtInpHF_MExInpHF_MEyInpHF_sumEtInmHF_MExInmHF_MEyInmHF(double &sumEtInpHF, double &MExInpHF, double &MEyInpHF, double &sumEtInmHF, double &MExInmHF, double &MEyInmHF, const CaloTower* calotower);
  void remove_HF_from_MET(CommonMETData &met, double sumEtInpHF, double MExInpHF, double MEyInpHF, double sumEtInmHF, double MExInmHF, double MEyInmHF);
  void add_MET_in_HF(SpecificCaloMETData &specific, double sumEtInpHF, double MExInpHF, double MEyInpHF, double sumEtInmHF, double MExInmHF, double MEyInmHF);

  DetId find_DetId_of_HCAL_cell_in_constituent_of(const CaloTower* calotower);
  DetId find_DetId_of_ECAL_cell_in_constituent_of(const CaloTower* calotower);

};

//____________________________________________________________________________||
#endif // METProducers_CaloMETInfo_h

