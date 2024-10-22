/**
  * \file EcalEleCalibLooper.h
  * \class EcalEleCalibLooper
  * \brief ECAL TB 2006 calibration with matrix inversion technique
  * \author 
  *
*/
#ifndef __CINT__
#ifndef EcalEleCalibLooper_H
#define EcalEleCalibLooper_H
#include "Calibration/EcalCalibAlgos/interface/VEcalCalibBlock.h"
#include "FWCore/Framework/interface/EDLooper.h"
#include "FWCore/Framework/interface/Event.h"
#include "Calibration/EcalCalibAlgos/interface/VFillMap.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include <string>
#include <vector>

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "CLHEP/Matrix/GenMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"
class EcalEleCalibLooper : public edm::EDLooper {
public:
  //! ctor
  explicit EcalEleCalibLooper(const edm::ParameterSet &);
  //! dtor
  ~EcalEleCalibLooper() override;
  void beginOfJob() override;
  void endOfJob() override;
  void startingNewLoop(unsigned int) override;
  Status duringLoop(const edm::Event &, const edm::EventSetup &) override;
  Status endOfLoop(const edm::EventSetup &, unsigned int iCounter) override;

private:
  //DS to divide in Regions
  int EBRegionId(const int, const int) const;
  int EERegionId(const int, const int) const;
  //DS to define the regions for each cristal
  void EBRegionDefinition();
  void EERegionDefinition();
  //DS defines the limit for the tan of phi
  double giveLimit(int);

  //DS checks if the values of ics and ips are in EE or not
  int EEregionCheck(const int, const int) const;
  int EBregionCheck(const int eta, const int phi) const;

  //!LP Change the coordinate system
  int etaShifter(const int) const;

private:
  //! EcalBarrel Input Collection name
  const edm::InputTag m_barrelAlCa;
  //! EcalEndcap Input Collection name
  const edm::InputTag m_endcapAlCa;
  //! To take the electrons
  edm::InputTag m_ElectronLabel;

  //! reconstruction window size
  const int m_recoWindowSidex;
  const int m_recoWindowSidey;

  //! eta size of the sub-matrix
  const int m_etaWidth;  //PG sub matrix size and borders
  //! eta size of the additive border to the sub-matrix
  //    int m_etaBorder ; //FIXME
  //! phi size of the sub-matrix
  const int m_phiWidthEB;
  //! phi size of the additive border to the sub-matrix
  //    int m_phiBorderEB //FIXME;

  //! eta start of the region of interest
  const int m_etaStart;  //PG ECAL region to be calibrated
  //! eta end of the region of interest
  const int m_etaEnd;
  //! phi start of the region of interest
  const int m_phiStartEB;
  //! phi end of the region of interest
  const int m_phiEndEB;
  //!DS For the EE
  const int m_radStart;
  const int m_radEnd;
  const int m_radWidth;
  //FIXME    int m_radBorder ;
  const int m_phiStartEE;
  const int m_phiEndEE;
  const int m_phiWidthEE;

  //! maximum number of events per crystal
  const int m_maxSelectedNumPerXtal;

  //! single blocks calibrators
  std::vector<VEcalCalibBlock *> m_EcalCalibBlocks;
  //! minimum energy per crystal cut
  const double m_minEnergyPerCrystal;
  //! maximum energy per crystal cut
  const double m_maxEnergyPerCrystal;
  //! minimum coefficient accepted (RAW)
  const double m_minCoeff;
  //! maximum coefficient accepted (RAW)
  const double m_maxCoeff;
  //! to exclude the blocksolver
  const int m_usingBlockSolver;

  //!the maps of  recalib coeffs
  EcalIntercalibConstantMap m_barrelMap;
  EcalIntercalibConstantMap m_endcapMap;

  //! DS sets the number of loops to do
  const unsigned int m_loops;
  //The map Filler
  VFillMap *m_MapFiller;

  //DS number of regions in the Endcap
  inline int EEregionsNum() const;
  //DS number of regions in the barrel
  inline int EBregionsNum() const;

  std::vector<int> m_regions;

  std::vector<DetId> m_barrelCells;
  std::vector<DetId> m_endcapCells;

  std::map<int, int> m_xtalRegionId;
  std::map<int, int> m_xtalPositionInRegion;
  std::map<int, int> m_xtalNumOfHits;

  bool isfirstcall_;

  //! ED token
  const edm::EDGetTokenT<EBRecHitCollection> m_ebRecHitToken;
  const edm::EDGetTokenT<EERecHitCollection> m_eeRecHitToken;
  const edm::EDGetTokenT<reco::GsfElectronCollection> m_gsfElectronToken;
  //! ES token
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> m_geometryToken;
};
#endif
#endif
