/**
  * \file InvRingCalib.h
  * \class InvRingCalib
  * \brief ECAL TB 2006 calibration with matrix inversion technique
  * $Date: 2008/01/23 11:04:54 $
  * $Revision: 1.1.2.1 $
  * \author 
  *
*/
#ifndef __CINT__
#ifndef InvRingCalib_H
#define InvRingCalib_H
#include "Calibration/Tools/interface/matrixSaver.h"
#include "Calibration/Tools/interface/InvMatrixUtils.h"
#include "Calibration/EcalCalibAlgos/interface/IMACalibBlock.h"
#include "Calibration/Tools/interface/smartSelector.h"
#include "FWCore/Framework/interface/EDLooper.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"

#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include <iostream>
#include <string>
#include <vector>
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
class InvRingCalib : public edm::EDLooper {
  public:

    //! ctor
    explicit InvRingCalib(const edm::ParameterSet&);
    //! dtor
    ~InvRingCalib();
    void beginOfJob(const edm::EventSetup&) ;
    void endOfJob();
    void startingNewLoop(unsigned int) ;
    Status duringLoop(const edm::Event&, const edm::EventSetup&) ;
    Status endOfLoop(const edm::EventSetup&,unsigned int iCounter) ;
    
 //end

  private:

  //!The number of regions in EE
  inline int EERegionNum () const ;
  //!Number of regions in EB
  inline int EBRegionNum () const ;
  //!Defines the regions in the barrel
  void EBRegionDef () ;
  //!Defines the rins in the endcap
  void EERingDef (const edm::EventSetup&);
  //!Defines the regions in the endcap
  void EERegionDef ();
  //!Gives back in which region you are:
  int EBRegId(const int) const;
  //!gives back in which region of the endcap you are.
  int EERegId ( int) ;

  typedef reco::PixelMatchGsfElectronCollection::const_iterator eleIterator;
  //! fills the barrel energy map to be sent to the CalibBlock
    void fillEBMap (EBDetId, const EcalRecHitCollection *, std::map<int, double> &, int, double &);
 //! fills the endcap energy map to be sent to the CalibBlock
    void fillEEMap (EEDetId, const EcalRecHitCollection *, std::map<int, double> &, int, double &);
//! Find the most energetic Xtals    
    DetId findMaxHit ( const std::vector<DetId> & v1,
                       const EBRecHitCollection* EBhits , 
		       const EERecHitCollection* EEhits );
  private:

    //! EcalBarrel Input Collection name
    edm::InputTag m_barrelAlCa ;
    //! EcalEndcap Input Collection name
    edm::InputTag m_endcapAlCa ;
    //! To take the electrons
    edm::InputTag m_ElectronLabel ;
    //! reconstruction window size
    int m_recoWindowSide ;
    //! minimum energy per crystal cut
    double m_minEnergyPerCrystal ;
    //! maximum energy per crystal cut
    double m_maxEnergyPerCrystal ;
    //! eta start of the zone of interest
    int m_etaStart ;   
    //! eta end of the zone of interest
    int m_etaEnd ;
    //! eta size of the regions 
    int m_etaWidth ;
    //! maximum number of events per Ring
    int m_maxSelectedNumPerRing ; 
    //! number of events already read per Ring
    std::map<int,int> m_RingNumOfHits;
    //! single blocks calibrators
    std::vector<IMACalibBlock> m_IMACalibBlocks ;
    //! minimum coefficient accepted (RAW)
    double m_minCoeff ;
    //! maximum coefficient accepted (RAW)
    double m_maxCoeff ;
    //! to exclude the blocksolver 
    int m_usingBlockSolver ;
    //!position of the cell, borders, coords etc...
    std::map<int,GlobalPoint> m_cellPos;
    std::map<int,int> m_cellPhi;
    //!association map between coeff and ring 
    std::map <int,double> m_InterRings;
    //!coeffs for the single xtals
    EcalIntercalibConstantMap m_barrelMap;
    EcalIntercalibConstantMap m_endcapMap;
    //! LP sets the number of loops to do
    int m_loops ;
    //! LP define the EE region to calibrate
    int m_startRing;
    int m_endRing;
    //!association map between Raw detIds and Rings
    std::map<int,int> m_xtalRing;
    //!association map between  raw detIds and Region
    std::map<int,int> m_xtalRegionId;
    //!association map between raw detIds and the number of the ring inside the region
    std::map<int,int> m_RinginRegion;
    //! geometry things used all over the file
    std::vector<DetId> m_barrelCells;
    std::vector<DetId> m_endcapCells;
    //!coeffs filenames
    std::string m_EBcoeffFile;
    std::string m_EEcoeffFile;
    //!endcap zone to be calibrated
    int m_EEZone;
};
#endif
#endif
