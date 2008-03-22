/**
  * \file EcalEleCalibLooper.h
  * \class EcalEleCalibLooper
  * \brief ECAL TB 2006 calibration with matrix inversion technique
  * $Date: 2008/01/23 11:04:54 $
  * $Revision: 1.1.2.1 $
  * \author 
  *
*/
#ifndef __CINT__
#ifndef EcalEleCalibLooper_H
#define EcalEleCalibLooper_H
#include "Calibration/EcalCalibAlgos/interface/VEcalCalibBlock.h"
#include "Calibration/Tools/interface/smartSelector.h"
#include "FWCore/Framework/interface/EDLooper.h"
#include "FWCore/Framework/interface/Event.h"

#include <iostream>
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
    void beginOfJob(const edm::EventSetup&) ;
    void endOfJob();
    void startingNewLoop(unsigned int) ;
    Status duringLoop(const edm::Event&, const edm::EventSetup&) ;
    Status endOfLoop(const edm::EventSetup&,unsigned int iCounter) ;
    

  private:

    typedef reco::GsfElectronCollection::const_iterator eleIterator;
    typedef edm::Handle<reco::BasicClusterShapeAssociationCollection>  HandleBasicCSAC; 

    DetId getMaxId(eleIterator EleIt,
                   HandleBasicCSAC  & barrelClShpHandle,
                   HandleBasicCSAC  & endcapClShpHandle);

    DetId findMaxHit (const std::vector<DetId> & v1,
    				  const EBRecHitCollection* EBhits,
    				  const EERecHitCollection* EEhits) ;

    void fillEBMap (EBDetId EBmax,
                    const EcalRecHitCollection * barrelHitsCollection,
                    std::map<int,double> & EBXtlMap,
                    int EBNumberOfRegion, double & pSubtract ) ;

    void fillEEMap (EEDetId EEmax,
                    const EcalRecHitCollection * endcapHitsCollection,
                    std::map<int,double> & EExtlMap,
                    int EENumberOfRegion, double & pSubtract ) ;


    //! write on plain text file the results
    int makeReport (std::string baseName="output") ;
    //! give the number of chi2 matrices
    int evalKaliX2Num () ;

    //! get the index of the sub-matrix
 //   int indexFinder (const int etaWorld, const int phiWorld) ;//FIXME
    //! fill a TH12F from a vector
  //  TH1F * fillTrend (std::vector<double> const & vettore, 
    //                  const int & index) ;
   //! look for void lines in the matrix   
   int findVoidLine (const CLHEP::HepMatrix & suspect) ;
   //! transfer from a CLHEP matrix to a C-like array
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
    int m_recoWindowSide ;

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
    int m_EBxtlNum[170][360] ;
    int m_EBxtlReg[170][360] ;
    int m_EExtlNum[100][100] ;
    int m_EExtlReg[100][100] ;

    //! half width on the front face of the crystal along x
//FIXME    double m_halfXBand ;
    //! half width on the front face of the crystal along y
//FIXME    double m_halfYBand ;
    //! maximum number of events per crystal
    int m_maxSelectedNumPerXtal ;  
    //! for statistical studies
//FIXME    int m_smallestFraction ;
    //! for statistical studies
//FIXME    int m_howManyFractions ;
    //! for statistical studies
//FIXME!!    smartSelector m_eventSelector ;
    //! for statistical studies
//    int halfSelecting ;
    //! for statistical studies
//    int takeOdd ;

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

    //!the map of  recalib coeffs
    std::map<int,double> m_recalibMap ;

    //! DS sets the number of loops to do
    unsigned int m_loops ;
    //! To take the electrons
    edm::InputTag m_ElectronLabel ;


  //DS numero delle regioni lungo il raggio (onion rings) (da fare divisione lungo phi)
  inline int EEregionsNum () const ;
  //DS numero delle regioni in EB
  inline int EBregionsNum () const ;

  std::vector<int> m_regions;
  
  std::vector<DetId> m_barrelCells;
  std::vector<DetId> m_endcapCells;

  std::map<int,int> m_xtalRegionId ;
  std::map<int,int> m_xtalPositionInRegion ;

  std::map <int,int> m_xtalNumOfHits;
  //  std::map<int,double> m_miscalibMap;
};
#endif
#endif
