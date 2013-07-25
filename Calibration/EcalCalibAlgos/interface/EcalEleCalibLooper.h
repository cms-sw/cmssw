/**
  * \file EcalEleCalibLooper.h
  * \class EcalEleCalibLooper
  * \brief ECAL TB 2006 calibration with matrix inversion technique
  * $Date: 2010/01/18 21:31:47 $
  * $Revision: 1.5 $
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

#include "CLHEP/Matrix/GenMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"
class EcalEleCalibLooper : public edm::EDLooper {
  public:

    //! ctor
    explicit EcalEleCalibLooper(const edm::ParameterSet&);
    //! dtor
    ~EcalEleCalibLooper();
    void beginOfJob() ;
    void endOfJob();
    void startingNewLoop(unsigned int) ;
    Status duringLoop(const edm::Event&, const edm::EventSetup&) ;
    Status endOfLoop(const edm::EventSetup&,unsigned int iCounter) ;
    

  private:

  //DS to divide in Regions
  int EBRegionId (const int, const int) const;
  int EERegionId (const int, const int) const;
  //DS to define the regions for each cristal
  void EBRegionDefinition ();
  void EERegionDefinition ();
  //DS defines the limit for the tan of phi
  double giveLimit (int);

  //DS checks if the values of ics and ips are in EE or not
  int EEregionCheck (const int, const int) const;
  int EBregionCheck (const int eta,const int phi) const;

  //!LP Change the coordinate system
  int etaShifter (const int) const ;

  private:


    //! EcalBarrel Input Collection name
    edm::InputTag m_barrelAlCa ;
    //! EcalEndcap Input Collection name
    edm::InputTag m_endcapAlCa ;
  
    //! reconstruction window size
    int m_recoWindowSidex ;
    int m_recoWindowSidey ;

    //! eta size of the sub-matrix
    int m_etaWidth ;   //PG sub matrix size and borders
    //! eta size of the additive border to the sub-matrix
//    int m_etaBorder ; //FIXME
    //! phi size of the sub-matrix
    int m_phiWidthEB ;
    //! phi size of the additive border to the sub-matrix
//    int m_phiBorderEB //FIXME;
    
    //! eta start of the region of interest
    int m_etaStart ;   //PG ECAL region to be calibrated
    //! eta end of the region of interest
    int m_etaEnd ;
    //! phi start of the region of interest
    int m_phiStartEB ;
    //! phi end of the region of interest
    int m_phiEndEB ;
    //!DS For the EE
    int m_radStart ;
    int m_radEnd ;
    int m_radWidth ;
//FIXME    int m_radBorder ;
    int m_phiStartEE ;
    int m_phiEndEE ;
    int m_phiWidthEE ;

    //! maximum number of events per crystal
    int m_maxSelectedNumPerXtal ;  

    //! single blocks calibrators
    std::vector<VEcalCalibBlock *> m_EcalCalibBlocks ;
    //! minimum energy per crystal cut
    double m_minEnergyPerCrystal ;
    //! maximum energy per crystal cut
    double m_maxEnergyPerCrystal ;
    //! minimum coefficient accepted (RAW)
    double m_minCoeff ;
    //! maximum coefficient accepted (RAW)
    double m_maxCoeff ;
    //! to exclude the blocksolver 
    int m_usingBlockSolver ;

    //!the maps of  recalib coeffs
    EcalIntercalibConstantMap  m_barrelMap ;
    EcalIntercalibConstantMap  m_endcapMap ;

    //! DS sets the number of loops to do
    unsigned int m_loops ;
    //! To take the electrons
    edm::InputTag m_ElectronLabel ;
    //The map Filler
    VFillMap * m_MapFiller; 

  //DS number of regions in the Endcap
  inline int EEregionsNum () const ;
  //DS number of regions in the barrel
  inline int EBregionsNum () const ;

  std::vector<int> m_regions;
  
  std::vector<DetId> m_barrelCells;
  std::vector<DetId> m_endcapCells;

  std::map<int,int> m_xtalRegionId ;
  std::map<int,int> m_xtalPositionInRegion ;
  std::map <int,int> m_xtalNumOfHits;

  bool isfirstcall_;
};
#endif
#endif
