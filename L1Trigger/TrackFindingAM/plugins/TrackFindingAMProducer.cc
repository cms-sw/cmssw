/*! \class   TrackFindingAMProducer
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 *  \update by S.Viret 
 *  \date   2014, Feb 17
 *
 */

#ifndef TRACK_BUILDER_AM_H
#define TRACK_BUILDER_AM_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"

#include "L1Trigger/TrackFindingAM/interface/CMSPatternLayer.h"
#include "L1Trigger/TrackFindingAM/interface/PatternFinder.h"
#include "L1Trigger/TrackFindingAM/interface/SectorTree.h"
#include "L1Trigger/TrackFindingAM/interface/Hit.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>

#ifndef __APPLE__
BOOST_CLASS_EXPORT_IMPLEMENT(CMSPatternLayer)
#endif

class TrackFindingAMProducer : public edm::EDProducer
{
  public:
    /// Constructor
    explicit TrackFindingAMProducer( const edm::ParameterSet& iConfig );

    /// Destructor;
    ~TrackFindingAMProducer();

  private:

  /// Data members
  double                       mMagneticField;
  std::string                  nBKName;
  int                          nThresh;
  int                          nMissingHits;
  int                          nDebug;
  SectorTree                   m_st;
  PatternFinder                *m_pf;
  const StackedTrackerGeometry *theStackedTracker;
  edm::InputTag                TTStubsInputTag;
  edm::InputTag                TTClustersInputTag;
  std::string                  TTPatternOutputTag;

  /// Mandatory methods
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

  bool inPattern(int j);

}; /// Close class

/*! \brief   Implementation of methods
 */

/// Constructors
TrackFindingAMProducer::TrackFindingAMProducer( const edm::ParameterSet& iConfig )
{
  TTStubsInputTag    = iConfig.getParameter< edm::InputTag >( "TTInputStubs" );
  TTPatternOutputTag = iConfig.getParameter< std::string >( "TTPatternName" );
  nBKName            = iConfig.getParameter< std::string >("inputBankFile");
  nThresh            = iConfig.getParameter< int >("threshold");
  nMissingHits       = iConfig.getParameter< int >("nbMissingHits");
  nDebug             = iConfig.getParameter< int >("debugMode");

  std::cout << "Loading pattern bank file : " << std::endl;
  std::cout << nBKName << std::endl;

  std::ifstream ifs(nBKName.c_str());

  //boost::archive::text_iarchive ia(ifs);
  boost::iostreams::filtering_stream<boost::iostreams::input> f;
  f.push(boost::iostreams::gzip_decompressor());
  try { 
    f.push(ifs);
    boost::archive::text_iarchive ia(f);
    ia >> m_st;
  }
  catch (boost::iostreams::gzip_error& e) {
    if(e.error()==4){//file is not compressed->read it without decompression
      std::ifstream new_ifs(nBKName.c_str());
      boost::archive::text_iarchive ia(new_ifs);
      ia >> m_st;
    }
  }  

  m_pf = new PatternFinder( m_st.getSuperStripSize(), nThresh, &m_st, "", "" );

  if(nMissingHits>-1)
  {
    m_pf->useMissingHitThreshold(nMissingHits);
  }

  produces< std::vector< TTTrack< Ref_PixelDigi_ > > >( TTPatternOutputTag );
}

/// Destructor
TrackFindingAMProducer::~TrackFindingAMProducer() {}

/// Begin run
void TrackFindingAMProducer::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Get the geometry references
  edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
  iSetup.get< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  theStackedTracker = StackedTrackerGeomHandle.product();

  /// Get magnetic field
  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();
  mMagneticField = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0;
}

/// End run
void TrackFindingAMProducer::endRun( const edm::Run& run, const edm::EventSetup& iSetup ) {}

/// Implement the producer
void TrackFindingAMProducer::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  /// Prepare output
  /// The temporary collection is used to store tracks
  /// before removal of duplicates
  std::auto_ptr< std::vector< TTTrack< Ref_PixelDigi_ > > > TTTracksForOutput( new std::vector< TTTrack< Ref_PixelDigi_ > > );

  /// Get the Stubs/Cluster already stored
  edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > TTStubHandle;
  iEvent.getByLabel( TTStubsInputTag, TTStubHandle );

  /// STEP 0
  /// Prepare output

  TTTracksForOutput->clear();

  int layer  = 0;
  int ladder = 0;
  int module = 0;
  int n_active =  m_st.getAllSectors().at(0)->getNbLayers();

  /// STEP 1
  /// Loop over input stubs

  std::vector< Hit* > m_hits;
  for(unsigned int i=0;i<m_hits.size();i++) delete m_hits[i];
  m_hits.clear();

  unsigned int j = 0;
  std::map< unsigned int, edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > stubMap;

  edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >::const_iterator inputIter;
  edmNew::DetSet< TTStub< Ref_PixelDigi_ > >::const_iterator stubIter;

  for ( inputIter = TTStubHandle->begin(); inputIter != TTStubHandle->end(); ++inputIter )
  {
    for ( stubIter = inputIter->begin(); stubIter != inputIter->end(); ++stubIter )
    {
      /// Increment the counter
      j++;

      /// Make the Ref to be put in the Track
      edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > tempStubRef = makeRefTo( TTStubHandle, stubIter );

      stubMap.insert( std::make_pair( j, tempStubRef ) );

      /// Calculate average coordinates col/row for inner/outer Cluster
      /// These are already corrected for being at the center of each pixel
      MeasurementPoint mp0 = tempStubRef->getClusterRef(0)->findAverageLocalCoordinates();
      GlobalPoint posStub  = theStackedTracker->findGlobalPosition( &(*tempStubRef) );

      StackedTrackerDetId detIdStub( tempStubRef->getDetId() );

      const GeomDetUnit* det0 = theStackedTracker->idToDetUnit( detIdStub, 0 );
      const GeomDetUnit* det1 = theStackedTracker->idToDetUnit( detIdStub, 1 );

      /// Find pixel pitch and topology related information
      const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
      const PixelGeomDetUnit* pix1 = dynamic_cast< const PixelGeomDetUnit* >( det1 );
      const PixelTopology* top0    = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
      const PixelTopology* top1    = dynamic_cast< const PixelTopology* >( &(pix1->specificTopology()) );

      /// Find the z-segment
      int cols0   = top0->ncolumns();
      int cols1   = top1->ncolumns();
      int ratio   = cols0/cols1; /// This assumes the ratio is integer!
      int segment = floor( mp0.y() / ratio );

      // Here we rearrange the number in order to be compatible with the AM emulator
      if ( detIdStub.isBarrel() )
      {
        layer  = detIdStub.iLayer()+4;
        ladder = detIdStub.iPhi()-1;
        module = detIdStub.iZ()-1;
      }
      else if ( detIdStub.isEndcap() )
      {
        layer  = 10+detIdStub.iZ()+abs((int)(detIdStub.iSide())-2)*7;
        ladder = detIdStub.iRing()-1;
        module = detIdStub.iPhi()-1;

	//	std::cout << mp0.y() << " / " << cols0 << " / " << cols1 << " / " << segment << std::endl;
      }

      module = CMSPatternLayer::getModuleCode(layer,module);

      // the stub is on the third Z position on the other side of the tracker -> out of range
      if ( module < 0 )  continue;

      ladder = CMSPatternLayer::getLadderCode(layer, ladder);

      float x    = posStub.x();
      float y    = posStub.y();
      float z    = posStub.z();

      Hit* h = new Hit(layer,ladder, module, segment, mp0.x(), j, -1, 0, 0, 0, 0, x, y, z, 0, 0, 0);
      m_hits.push_back(h);
    } /// End of loop over input stubs
  } /// End of loop over DetSetVector

  /// STEP 2
  /// PAssing the superstrips into the AM chip

  std::vector< Sector* > patternsSectors = m_pf->find(m_hits); // AM PR is done here....
  if(nDebug==1)
    m_pf->displaySuperstrips(m_hits); // display the supertrips of the event

  /// STEP 3
  /// Collect the info and store the track seed stuff

  std::vector< Hit* > hits;

  std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > tempVec;

  for ( unsigned int i = 0; i < patternsSectors.size(); i++ )
  {
    std::vector< GradedPattern* > pl = patternsSectors[i]->getPatternTree()->getLDPatterns();

    if ( pl.size() == 0 ) continue; // No patterns

    int secID = patternsSectors[i]->getOfficialID();

    //    std::cout<<"Found "<<pl.size()<<" patterns in sector " << secID<<std::endl;
    //    std::cout<<"containing "<<n_active<<" layers " << secID<<std::endl;
  
    //delete the GradedPattern objects

    for ( unsigned j = 0; j < pl.size(); j++ )
    {
      hits.clear();
      hits = pl[j]->getHits();

      /// Create the Seed in the form of a Track and store it in the output
      tempVec.clear();

      for(unsigned k = 0; k < hits.size(); k++ )
        tempVec.push_back( stubMap[ hits[k]->getID() ] );


      TTTrack< Ref_PixelDigi_ > tempTrack( tempVec );
      tempTrack.setSector( secID );
      tempTrack.setWedge( n_active );
      tempTrack.setPOCA( GlobalPoint(0.,0.,0.),5);		
      TTTracksForOutput->push_back( tempTrack );

      delete pl[j];
    }

    //delete the Sectors
    delete patternsSectors[i];
  }

  /// Put in the event content
  iEvent.put( TTTracksForOutput, TTPatternOutputTag);
}
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(TrackFindingAMProducer);

#endif

