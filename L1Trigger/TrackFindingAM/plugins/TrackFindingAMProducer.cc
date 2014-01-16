/*! \class   TrackFindingAMProducer
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
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

#include "L1Trigger/TrackFindingAM/interface/CMSPatternLayer.h"
#include "L1Trigger/TrackFindingAM/interface/PatternFinder.h"
#include "L1Trigger/TrackFindingAM/interface/SectorTree.h"
#include "L1Trigger/TrackFindingAM/interface/Hit.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
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
    unsigned int                 nSectors;
    unsigned int                 nWedges;
    std::string                  nBKName;
    int                          nThresh;
    SectorTree                   m_st;
    PatternFinder                *m_pf;
    const StackedTrackerGeometry *theStackedTracker;
    edm::InputTag                TTStubsInputTag;

    /// Mandatory methods
    virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

}; /// Close class

/*! \brief   Implementation of methods
 */

/// Constructors
TrackFindingAMProducer::TrackFindingAMProducer( const edm::ParameterSet& iConfig )
{
  TTStubsInputTag = iConfig.getParameter< edm::InputTag >( "TTStubsBricks" );
  nSectors = iConfig.getParameter< int >("NumSectors");
  nWedges = iConfig.getParameter< int >("NumWedges");
  nBKName = iConfig.getParameter< std::string >("inputBankFile");
  nThresh = iConfig.getParameter< int >("threshold");

  std::cout << "Loading pattern bank file : " << std::endl;
  std::cout << nBKName << std::endl;

  std::ifstream ifs(nBKName.c_str());
  boost::archive::text_iarchive ia(ifs);

  ia >> m_st;
  m_pf = new PatternFinder( m_st.getSuperStripSize(), nThresh, &m_st, "", "" );

  produces< std::vector< TTTrack< Ref_PixelDigi_ > > >( "AML1Tracks" );
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

  /// Get the Stubs already stored away
  edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > TTStubHandle;
  iEvent.getByLabel( TTStubsInputTag, TTStubHandle );

  /// STEP 0
  /// Prepare output
  TTTracksForOutput->clear();

  int layer  = 0;
  int ladder = 0;
  int module = 0;
  int disk   = 0;
  int lad_cor= 0;

  /// STEP 1
  /// Loop over input stubs

  //  std::cout << "Start the loop on stubs to transform them into sstrips" << std::endl;

  std::vector< Hit* > m_hits;

  for(unsigned int i=0;i<m_hits.size();i++)
  {
    delete m_hits[i];
  }

  m_hits.clear();

  std::map< unsigned int, edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > stubMap;

  unsigned int j = 0;
  edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >::const_iterator inputIter;
  edmNew::DetSet< TTStub< Ref_PixelDigi_ > >::const_iterator stubIter;
  for ( inputIter = TTStubHandle->begin();
        inputIter != TTStubHandle->end();
        ++inputIter )
  {
    for ( stubIter = inputIter->begin();
          stubIter != inputIter->end();
          ++stubIter )
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
      //bool isPS = theStackedTracker->isPSModule( detIdStub );

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
        layer  = 10+detIdStub.iZ()+abs(detIdStub.iSide()-2)*7;

        if (layer>10 && layer<=17) disk=(layer-10)%8;
        if (layer>17 && layer<=24) disk=(layer-17)%8;
        if (disk>=5) lad_cor = (disk-4)%4;

        ladder = detIdStub.iRing()-1+lad_cor;
        module = detIdStub.iPhi()-1;
      }

      module = CMSPatternLayer::getModuleCode(layer,module);
      if ( module < 0 ) // the stub is on the third Z position on the other side of the tracker -> out of range
        continue;

      ladder = CMSPatternLayer::getLadderCode(layer, ladder);

      int strip  =  mp0.x();
      int tp     = -1;
      float eta  = 0;
      float phi0 = 0;
      float spt  = 0;
      float x    = posStub.x();
      float y    = posStub.y();
      float z    = posStub.z();
      float x0   = 0.;
      float y0   = 0.;
      float z0   = 0.;
      float ip   = sqrt(x0*x0+y0*y0);

      Hit* h = new Hit(layer,ladder, module, segment, strip, j, tp, spt, ip, eta, phi0, x, y, z, x0, y0, z0);
      m_hits.push_back(h);
    } /// End of loop over input stubs
  } /// End of loop over DetSetVector

  /// STEP 2
  /// PAssing the superstrips into the AM chip

  //  std::cout << "AM chip processing" << std::endl;

  std::vector< Sector* > patternsSectors = m_pf->find(m_hits); // AM PR is done here....

  /// STEP 3
  /// Collect the info and store the track seed stuff

  //  std::cout << "AM chip processing" << std::endl;

  std::vector< Hit* > hits;
  for ( unsigned int i = 0; i < patternsSectors.size(); i++ )
  {
    std::vector< GradedPattern* > pl = patternsSectors[i]->getPatternTree()->getLDPatterns();

    if ( pl.size() == 0 ) continue; // No patterns

    int secID = patternsSectors[i]->getOfficialID();

    //    std::cout<<"Found "<<pl.size()<<" patterns in sector " << secID<<std::endl;

    //delete the GradedPattern objects
    for ( unsigned j = 0; j < pl.size(); j++ )
    {
      hits.clear();
      hits = pl[j]->getHits();

      /// Create the Seed in the form of a Track and store it in the output
      std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > tempVec;

      for(unsigned k = 0; k < hits.size(); k++ )
      {
        tempVec.push_back( stubMap[ hits[k]->getID() ] );
      }

      TTTrack< Ref_PixelDigi_ > tempTrack( tempVec );
      tempTrack.setSector( secID );
      TTTracksForOutput->push_back( tempTrack );

      delete pl[j];
    }

    //delete the Sectors
    delete patternsSectors[i];
  }

  /// Put in the event content
  iEvent.put( TTTracksForOutput, "AML1Tracks" );
}

// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(TrackFindingAMProducer);

#endif

