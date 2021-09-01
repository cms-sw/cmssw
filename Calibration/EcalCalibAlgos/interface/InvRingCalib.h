/**
  * \file InvRingCalib.h
  * \class InvRingCalib
  * \brief ECAL TB 2006 calibration with matrix inversion technique
  * \author 
  *
*/
#ifndef __CINT__
#ifndef InvRingCalib_H
#define InvRingCalib_H
#include "Calibration/EcalCalibAlgos/interface/VEcalCalibBlock.h"
#include "FWCore/Framework/interface/EDLooper.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <string>
#include <vector>

#include "Calibration/EcalCalibAlgos/interface/VFillMap.h"

class InvRingCalib : public edm::EDLooper {
public:
  //! ctor
  explicit InvRingCalib(const edm::ParameterSet &);
  //! dtor
  ~InvRingCalib() override;
  void beginOfJob() override;
  void endOfJob() override;
  void startingNewLoop(unsigned int) override;
  Status duringLoop(const edm::Event &, const edm::EventSetup &) override;
  Status endOfLoop(const edm::EventSetup &, unsigned int iCounter) override;

  //end

private:
  //!The number of regions in EE
  inline int EERegionNum() const;
  //!Number of regions in EB
  int EBRegionNum() const;
  //!Defines the regions in the barrel
  void EBRegionDef();
  //!Defines the rins in the endcap
  void EERingDef(const edm::EventSetup &);
  //!Defines the regions in the endcap
  void EERegionDef();
  //!Prepares the EB regions;
  void RegPrepare();
  //!Gives back in which region you are:
  int EBRegId(const int);
  //!gives back in which region of the endcap you are.
  int EERegId(int);

  //!The class that fills the map!
  VFillMap *m_MapFiller;

private:
  //! EcalBarrel Input Collection name
  const edm::InputTag m_barrelAlCa;
  //! EcalEndcap Input Collection name
  const edm::InputTag m_endcapAlCa;
  //! To take the electrons
  const edm::InputTag m_ElectronLabel;
  //! reconstruction window size
  const int m_recoWindowSidex;
  const int m_recoWindowSidey;
  //! minimum energy per crystal cut
  const double m_minEnergyPerCrystal;
  //! maximum energy per crystal cut
  const double m_maxEnergyPerCrystal;
  //! eta start of the zone of interest
  const int m_etaStart;
  //! eta end of the zone of interest
  const int m_etaEnd;
  //! eta size of the regions
  const int m_etaWidth;
  //    std::map<int,float> m_eta;
  //! maximum number of events per Ring
  const int m_maxSelectedNumPerRing;
  //! number of events already read per Ring
  std::map<int, int> m_RingNumOfHits;
  //! single blocks calibrators
  std::vector<VEcalCalibBlock *> m_IMACalibBlocks;
  //! minimum coefficient accepted (RAW)
  const double m_minCoeff;
  //! maximum coefficient accepted (RAW)
  const double m_maxCoeff;
  //! to exclude the blocksolver
  const int m_usingBlockSolver;
  //!position of the cell, borders, coords etc...
  std::map<int, GlobalPoint> m_cellPos;
  std::map<int, int> m_cellPhi;
  //!association map between coeff and ring
  //!coeffs for the single xtals
  EcalIntercalibConstantMap m_barrelMap;
  EcalIntercalibConstantMap m_endcapMap;
  //! LP sets the number of loops to do
  unsigned int m_loops;
  //! LP define the EE region to calibrate
  const int m_startRing;
  const int m_endRing;
  //!association map between Raw detIds and Rings
  std::map<int, int> m_xtalRing;
  //!association map between  raw detIds and Region
  std::map<int, int> m_xtalRegionId;
  //!association map between raw detIds and the number of the ring inside the region
  std::map<int, int> m_RinginRegion;

  //! geometry things used all over the file
  std::vector<DetId> m_barrelCells;
  std::vector<DetId> m_endcapCells;
  //!coeffs filenames
  const std::string m_EBcoeffFile;
  const std::string m_EEcoeffFile;
  //!endcap zone to be calibrated
  const int m_EEZone;
  //!EB regions vs. eta index
  std::map<int, int> m_Reg;
  std::string m_mapFillerType;
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
