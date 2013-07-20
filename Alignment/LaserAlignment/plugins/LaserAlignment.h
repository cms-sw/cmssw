
#ifndef LaserAlignment_LaserAlignment_H
#define LaserAlignment_LaserAlignment_H


/** \class LaserAlignment
 *  Main reconstruction module for the Laser Alignment System
 *
 *  $Date: 2013/05/25 14:31:03 $
 *  $Revision: 1.33 $
 *  \author Maarten Thomas
 *  \author Jan Olzem
 */



#include <sstream>
#include <iostream>
#include <cmath>

#include "FWCore/Framework/interface/Event.h" 
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Alignment/interface/SiStripLaserRecHit2D.h"
#include "DataFormats/Alignment/interface/TkLasBeam.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include "Alignment/LaserAlignment/interface/LASGlobalData.h"
#include "Alignment/LaserAlignment/interface/LASGlobalLoop.h"
#include "Alignment/LaserAlignment/interface/LASModuleProfile.h"
#include "Alignment/LaserAlignment/interface/LASProfileJudge.h"
#include "Alignment/LaserAlignment/interface/LASBarrelAlgorithm.h"
#include "Alignment/LaserAlignment/interface/LASAlignmentTubeAlgorithm.h"
#include "Alignment/LaserAlignment/interface/LASEndcapAlgorithm.h"
#include "Alignment/LaserAlignment/interface/LASPeakFinder.h"
#include "Alignment/LaserAlignment/interface/LASCoordinateSet.h"
#include "Alignment/LaserAlignment/interface/LASGeometryUpdater.h"
#include "Alignment/LaserAlignment/interface/LASConstants.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "TH1.h"
#include "TFile.h"
#include "TF1.h"








///
///
///
class LaserAlignment : public edm::one::EDProducer<edm::EndRunProducer>, public TObject {

 public:

  explicit LaserAlignment( edm::ParameterSet const& theConf );
  ~LaserAlignment();
  virtual void beginJob() override;
  virtual void produce( edm::Event&, edm::EventSetup const& ) override;
  virtual void endJob() override;
  virtual void endRunProduce(edm::Run&, const edm::EventSetup& ) override;

  /// for debugging & testing only, will disappear..
  void testRoutine( void );


 private:

  /// fill profiles from SiStrip(Raw)Digi container
  void fillDataProfiles( edm::Event const&, edm::EventSetup const& );

  /// fill pedestals from dbase
  void fillPedestalProfiles( edm::ESHandle<SiStripPedestals>&  );

  /// decide whether TEC or AT beams have fired
  bool isTECBeam( void );
  bool isATBeam( void );

  /// returns the nominal beam position (strips) in TOB for the profileJudge
  double getTIBTOBNominalBeamOffset( unsigned int, unsigned int, unsigned int );

  /// returns the nominal beam position (strips) in TEC (AT) for the profileJudge
  double getTEC2TECNominalBeamOffset( unsigned int, unsigned int, unsigned int );

  /// fill hard coded detIds
  void fillDetectorId( void );

  /// convert an angle in the [-pi,pi] range to the [0,2*pi] range
  double ConvertAngle( double );

  /// fills a LASGlobalData<LASCoordinateSet> with nominal module positions
  void CalculateNominalCoordinates( void );
  
  /// for debugging only, will disappear
  void DumpPosFileSet( LASGlobalData<LASCoordinateSet>& );

  /// for debugging only, will disappear
  void DumpStripFileSet( LASGlobalData<std::pair<float,float> >& );

  /// for debugging only, will disappear
  void DumpHitmaps( LASGlobalData<int>& );

  /// apply endcap correction to masked modules in TEC
  void ApplyEndcapMaskingCorrections( LASGlobalData<LASCoordinateSet>&, LASGlobalData<LASCoordinateSet>&, LASEndcapAlignmentParameterSet& );
  
  /// same for alignment tube modules
  void ApplyATMaskingCorrections( LASGlobalData<LASCoordinateSet>&, LASGlobalData<LASCoordinateSet>&, LASBarrelAlignmentParameterSet& ); 

  /// counter for the total number of events processed
  int theEvents;

  /// config switch
  bool theDoPedestalSubtraction;

  /// config switch
  bool theUseMinuitAlgorithm;

  /// config switch
  bool theApplyBeamKinkCorrections;

  /// config parameter
  double peakFinderThreshold;

  /// config switch
  bool enableJudgeZeroFilter;

  /// config parameters for the LASProfileJudge
  unsigned int judgeOverdriveThreshold;

  /// config switch
  bool updateFromInputGeometry;

  /// config switch
  bool misalignedByRefGeometry;

  /// config switch
  bool theStoreToDB;

  // digi producer
  std::vector<edm::ParameterSet> theDigiProducersList;

  /// config switch
  bool theSaveHistograms;

  /// config parameter (histograms file compression level)
  int theCompression;

  /// config parameter (histograms file output name)
  std::string theFileName;

  /// config parameters
  std::vector<unsigned int> theMaskTecModules;
  std::vector<unsigned int> theMaskAtModules;

  /// config switch
  bool theSetNominalStrips;

  // this object can judge if 
  // a LASModuleProfile is usable for position measurement
  LASProfileJudge judge;

  // colection of constants
  LASConstants theLasConstants;

  // the detector ids for all the modules
  LASGlobalData<unsigned int> detectorId;

  // the detector ids for the doubly hit modules in the TECs
  std::vector<unsigned int> tecDoubleHitDetId;

  // all the 474 profiles for the pedestals
  LASGlobalData<LASModuleProfile> pedestalProfiles;

  /// data profiles for the current event
  LASGlobalData<LASModuleProfile> currentDataProfiles;

  // summed data profiles
  LASGlobalData<LASModuleProfile> collectedDataProfiles;

  // 474 names for retrieving data from the branches
  LASGlobalData<std::string> theProfileNames;

  // number of accepted profiles for each module
  LASGlobalData<int> numberOfAcceptedProfiles;

  // hit map for the current event (int=0,1)
  // which is needed for branching into AT or TEC mode
  LASGlobalData<int> isAcceptedProfile;

  // histograms for the summed profiles;
  // these are needed for the fitting procedure (done by ROOT)
  LASGlobalData<TH1D*> summedHistograms;

  // container for nominal module positions
  LASGlobalData<LASCoordinateSet> nominalCoordinates;

  // a class for easy looping over LASGlobalData objects,
  // avoids nested for-statements
  LASGlobalLoop moduleLoop;

  /// Tree stuff
  TFile * theFile;
  TDirectory* singleModulesDir;

  /// tracker geometry;
  edm::ESHandle<GeometricDet> gD;
  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
  edm::ESHandle<Alignments> theGlobalPositionRcd;


  AlignableTracker* theAlignableTracker;

  std::string theAlignRecordName, theErrorRecordName;

  bool firstEvent_;

  const edm::ParameterSet theParameterSet;
};
#endif
