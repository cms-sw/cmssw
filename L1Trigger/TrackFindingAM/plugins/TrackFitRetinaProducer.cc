/*! \class   TrackFitRetinaProducer
 *
 *  \author M Casarsa / L Martini (mostly cut&paste from S.Viret and G.Baulieu's TrackFitHoughProducer) 
 *  \date   2014, May 20
 *
 */

#ifndef TRACK_FITTER_AM_H
#define TRACK_FITTER_AM_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
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
#include <math.h>


//#ifndef __APPLE__
//BOOST_CLASS_EXPORT_IMPLEMENT(CMSPatternLayer)
//#endif

class TrackFitRetinaProducer : public edm::EDProducer
{
  public:
    /// Constructor
    explicit TrackFitRetinaProducer( const edm::ParameterSet& iConfig );

    /// Destructor;
    ~TrackFitRetinaProducer();

  private:
  
  /// Data members
  double                       mMagneticField;
  unsigned int                 nSectors;
  unsigned int                 nWedges;
  std::string                  nBKName;
  int                          nThresh;
  const StackedTrackerGeometry *theStackedTracker;
  edm::InputTag                TTStubsInputTag;
  edm::InputTag                TTPatternsInputTag;
  std::string                  TTTrackOutputTag;
  int                          verboseLevel_;
  bool                         fitPerTriggerTower_;
  // removeDuplicates_:
  //            0  --> no duplicate removal
  //            1  --> use deltaR
  //            2  --> use common stubs
  int                          removeDuplicates_;

  unsigned int                 icount;


  /// Mandatory methods
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

}; /// Close class

/*! \brief   Implementation of methods
 */

/// Constructors
TrackFitRetinaProducer::TrackFitRetinaProducer( const edm::ParameterSet& iConfig )
{
  TTStubsInputTag     = iConfig.getParameter< edm::InputTag >( "TTInputStubs" );
  TTPatternsInputTag  = iConfig.getParameter< edm::InputTag >( "TTInputPatterns" );
  TTTrackOutputTag    = iConfig.getParameter< std::string >( "TTTrackName" );
  verboseLevel_       = iConfig.getUntrackedParameter< int >( "verboseLevel", 0 );
  removeDuplicates_   = iConfig.getUntrackedParameter< int >( "removeDuplicates", 1 );
  fitPerTriggerTower_ = iConfig.getUntrackedParameter< bool >( "fitPerTriggerTower", false );

  produces< std::vector< TTTrack< Ref_PixelDigi_ > > >( TTTrackOutputTag );
}

/// Destructor
TrackFitRetinaProducer::~TrackFitRetinaProducer() {}

/// Begin run
void TrackFitRetinaProducer::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Initialize the event counter
  icount = 0;
  
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
void TrackFitRetinaProducer::endRun( const edm::Run& run, const edm::EventSetup& iSetup ) {}

/// Implement the producer
void TrackFitRetinaProducer::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  
  icount++;

  // Get GEN particle collection
  edm::Handle<vector<reco::GenParticle> > genPart;
  iEvent.getByLabel ("genParticles", genPart);

  /// Prepare output
  /// The temporary collection is used to store tracks
  /// before removal of duplicates
  std::auto_ptr< std::vector< TTTrack< Ref_PixelDigi_ > > > TTTracksForOutput( new std::vector< TTTrack< Ref_PixelDigi_ > > );

  /// Get the Stubs already stored away
  edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > TTStubHandle;
  edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > > TTPatternHandle;

  iEvent.getByLabel( TTStubsInputTag, TTStubHandle );
  iEvent.getByLabel( TTPatternsInputTag, TTPatternHandle );

  /// STEP 0
  /// Prepare output
  TTTracksForOutput->clear();

  int layer  = 0;
  int ladder = 0;
  int module = 0;

  /// STEP 1
  /// Loop over patterns

  //  std::cout << "Start the loop over pattern in order to recover the stubs" << std::endl;

  map<int,vector<Hit*>* > m_hitsPerSector;
  map<int,set<long>*> m_uniqueHitsPerSector;

  edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >::const_iterator inputIter;
  edmNew::DetSet< TTStub< Ref_PixelDigi_ > >::const_iterator stubIter;

  std::vector< TTTrack< Ref_PixelDigi_ > >::const_iterator iterTTTrack;


  /// Go on only if there are Patterns from PixelDigis
  if ( TTPatternHandle->size() > 0 )
  {

    // --- Printout the generated particles:
    if ( verboseLevel_ > 0 ){
      cout << "\nEvent = " << icount 
	   << " ---------------------------------------------------------------------------------" << endl;
      cout << "  Generated particles:" << endl; 
      for (std::vector <reco::GenParticle>::const_iterator thepart = genPart->begin(); 
	                                                   thepart != genPart->end(); 
	                                                 ++thepart ){
	
	// curvature and helix radius:
	double c = thepart->charge()*0.003*mMagneticField/thepart->pt();
	double R = thepart->pt()/(0.003*mMagneticField);
	  
	// helix center:
	double x0 = thepart->vx() - thepart->charge()*R*thepart->py()/thepart->pt();
	double y0 = thepart->vy() + thepart->charge()*R*thepart->px()/thepart->pt();
	  
	// transverse and longitudinal impact parameters:
	double d0 = thepart->charge()*(sqrt(x0*x0+y0*y0)-R);
	double z0 = thepart->vz() - 2./c*thepart->pz()/thepart->pt()*
	  asin(0.5*c*sqrt((thepart->vx()*thepart->vx()+thepart->vy()*thepart->vy()-d0*d0)/(1.+c*d0)));
    
	cout << "   " << std::distance(genPart->begin(),thepart)
	     << "  -  pdgId = " << thepart->pdgId()
	     << "  c = " << c
	     << "  pt = " << thepart->pt()
	     << "  d0 = " << d0
	     << "  phi = " << thepart->phi()
	     << "  eta = " << thepart->eta()
	     << "  z0 = " << z0 
	     << endl;

      } // loop over thepart  

    } // if ( verboseLevel_ > 0 )


    if ( verboseLevel_ > 1 )
      cout << "   Number of roads = " << TTPatternHandle->size() << endl;


    /// Loop over Patterns
    unsigned int tkCnt = 0;
    unsigned int j     = 0;
    unsigned int jreal = 0;

    std::map< unsigned int , edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > stubMap;
    std::map< unsigned int , edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > stubMapUsed;

    for ( iterTTTrack = TTPatternHandle->begin();
	  iterTTTrack != TTPatternHandle->end();
	  ++iterTTTrack )
    {
      edm::Ptr< TTTrack< Ref_PixelDigi_ > > tempTrackPtr( TTPatternHandle, tkCnt++ );

      /// Get everything relevant
      unsigned int seedSector = tempTrackPtr->getSector();
      //std::cout << "Pattern in sector " << seedSector << " with " 
      //		<< seedWedge << " active layers contains " 
      //		<< nStubs << " stubs" << std::endl;

      std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_  > >, TTStub< Ref_PixelDigi_  > > > trackStubs = tempTrackPtr->getStubRefs();

      //get the hits list of this sector
      map<int,vector<Hit*>*>::iterator sec_it = m_hitsPerSector.find(seedSector);
      vector<Hit*> *m_hits;
      if(sec_it==m_hitsPerSector.end())
      {
	m_hits = new vector<Hit*>();
	m_hitsPerSector[seedSector]=m_hits;
      }
      else
      {
       m_hits = sec_it->second;
      }

      //get the hits set of this sector
      map<int,set<long>*>::iterator set_it = m_uniqueHitsPerSector.find(seedSector);
      set<long> *m_hitIDs;

      if(set_it==m_uniqueHitsPerSector.end())
      {
	m_hitIDs = new set<long>();
	m_uniqueHitsPerSector[seedSector]=m_hitIDs;
      }
      else
      {
	m_hitIDs = set_it->second;
      }

      // Loop over stubs contained in the pattern to recover the info

      vector<Hit*> road_hits;

      for(unsigned int i=0;i<trackStubs.size();i++)
      {
	++j;

	edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > tempStubRef = trackStubs.at(i);

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
	  layer  = 10+detIdStub.iZ()+abs((int)(detIdStub.iSide())-2)*7;
	  ladder = detIdStub.iRing()-1;
	  module = detIdStub.iPhi()-1;
	}

	//cout << layer << " / " << ladder << " / " << module << " / " << std::endl;

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
	
	//Check if the stub is already in the list
	long hit_id = (long)layer*10000000000+(long)ladder*100000000
	  +module*100000+segment*10000+strip;

	pair<set<long>::iterator,bool> result = m_hitIDs->insert(hit_id);

	if(result.second==true) //this is a new stub -> add it to the list
	{
	  ++jreal;
	  stubMapUsed.insert( std::make_pair( jreal, tempStubRef ) );

	  if (jreal>=16384)
	  {
	    cout << "Problem!!!" << endl;
	    continue;
	  }

	  Hit* h = new Hit(layer,ladder, module, segment, strip, 
			   jreal, tp, spt, ip, eta, phi0, x, y, z, x0, y0, z0);
	  m_hits->push_back(h);

	}

	if ( !fitPerTriggerTower_ ){

	  // Find the stub index:
	  unsigned int stub_index = 0;
	  for(std::map< unsigned int , 
		edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, 
		TTStub< Ref_PixelDigi_ > > >::iterator istub  = stubMapUsed.begin();
	                                               istub != stubMapUsed.end(); 
                                                     ++istub ){

	    if ( istub->second == tempStubRef )
	      stub_index =  istub->first;

	  }

	  if ( stub_index > 0 ){
	    Hit* h1 = new Hit(layer,ladder, module, segment, strip, 
			      stub_index, tp, spt, ip, eta, phi0, x, y, z, x0, y0, z0);
	    road_hits.push_back(h1);
	  }

	}

      } /// End of loop over track stubs


      // =====================================================================================================
      //  Fit the road stubs:
      // =====================================================================================================

      if ( fitPerTriggerTower_ ) continue;

      if ( verboseLevel_ > 1 )
	cout << "   road/number of stubs = " << j << " / " << road_hits.size() << endl;

      std::vector<Track*> tracks; 
      std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > tempVec;
      RetinaTrackFitter* fitter = new RetinaTrackFitter(5);

      fitter->setSectorID(seedSector);
      fitter->setEventCounter(icount);
      fitter->setRoadID(j);
      fitter->setVerboseLevel(0);
     
      fitter->fit(road_hits);
      tracks.clear();
      tracks = fitter->getTracks();
      fitter->clean();

      // Store the tracks (no duplicate cleaning yet)
      for(unsigned int tt=0;tt<tracks.size();tt++){	

	double pt_fit = 0.003*mMagneticField/ tracks[tt]->getCurve();

	if ( verboseLevel_ > 1 ){
	  cout << "   Fitted track:  "
	       << tt << "  -  c = " << tracks[tt]->getCurve()
	       << "  pt = " << pt_fit
	       << "  d0 = " << tracks[tt]->getD0()
	       << "  phi = " << tracks[tt]->getPhi0()
	       << "  eta = " << tracks[tt]->getEta0()
	       << "  z0 = " << tracks[tt]->getZ0() 
	       << "  -   weights = " << tracks[tt]->getWxy() << " " <<  tracks[tt]->getWrz() 
	       << endl;
	}
	    
	tempVec.clear();

	vector<int> stubs = tracks[tt]->getStubs();
	for(unsigned int sti=0;sti<stubs.size();sti++){
	  //cout<<stubs[sti]<<endl;
	  tempVec.push_back( stubMapUsed[ stubs[sti] ] );
	}

	double pz = pt_fit/(tan(2.*atan(exp(-tracks[tt]->getEta0()))));

	TTTrack< Ref_PixelDigi_ > tempTrack( tempVec );
	GlobalPoint POCA(0.,0.,tracks[tt]->getZ0());
	GlobalVector mom(pt_fit*cos(tracks[tt]->getPhi0()),
			 pt_fit*sin(tracks[tt]->getPhi0()),
			 pz);

	//	std::cout << tracks[tt]->getZ0() << " / " << POCA.z() << std::endl;


	// kludge: We save the maximum weights in the chi2 variable
	tempTrack.setChi2(tracks[tt]->getWxy(), 5);
	tempTrack.setChi2(tracks[tt]->getWrz(), 4);

	tempTrack.setRInv(tracks[tt]->getCurve(), 5);

	tempTrack.setSector( seedSector );
	tempTrack.setWedge( -1 );
	tempTrack.setMomentum( mom , 5);
	tempTrack.setPOCA( POCA , 5);
	//	std::cout << tracks[tt]->getZ0() << " / " << POCA.z() << " / " << tempTrack.getPOCA().z() << std::endl;
	TTTracksForOutput->push_back( tempTrack );
	    
	delete tracks[tt];
      }
    
      // --- Clean-up memory:
      delete(fitter);

      // Clean-up the road stub vector:
      for(std::vector<Hit*>::iterator ihit=road_hits.begin(); ihit!=road_hits.end(); ++ihit)
	delete *ihit;

    } // End of loop over patterns


    //free the map of sets
    for(map<int,set<long>*>::iterator set_it=m_uniqueHitsPerSector.begin();set_it!=m_uniqueHitsPerSector.end();set_it++)
      delete set_it->second;//delete the set*


    // =====================================================================================================
    //  Fit the all trigger tower stubs:
    // =====================================================================================================

    if ( fitPerTriggerTower_ ) {

      RetinaTrackFitter* fitter = new RetinaTrackFitter(5);
      vector<Track*> tracks; 
      std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > tempVec;

      // Loop over the different sectors
      for(map<int,vector<Hit*>*>::iterator sec_it=m_hitsPerSector.begin();sec_it!=m_hitsPerSector.end();sec_it++)
	{
	  //cout<<"Sector "<<sec_it->first<<endl;
	  fitter->setSectorID(sec_it->first);
	  fitter->setEventCounter(icount);
	  fitter->setVerboseLevel(0);

	  // Do the fit
	  fitter->fit(*(sec_it->second));
	  tracks.clear();
	  tracks = fitter->getTracks();
	  fitter->clean();


	  // Store the tracks (no duplicate cleaning yet)
	  for(unsigned int tt=0;tt<tracks.size();tt++)
	    {	

	      tempVec.clear();

	      double pt_fit = 0.003*mMagneticField/ tracks[tt]->getCurve();

	      if ( verboseLevel_ > 1 ){
		cout << "   Fitted track:  "
		     << tt << "  -  c = " << tracks[tt]->getCurve()
		     << "  pt = " << pt_fit
		     << "  d0 = " << tracks[tt]->getD0()
		     << "  phi = " << tracks[tt]->getPhi0()
		     << "  eta = " << tracks[tt]->getEta0()
		     << "  z0 = " << tracks[tt]->getZ0() 
		     << "  -   weights = " << tracks[tt]->getWxy() << " " <<  tracks[tt]->getWrz() 
		     << endl;
	      }

	      vector<int> stubs = tracks[tt]->getStubs();
	      for(unsigned int sti=0;sti<stubs.size();sti++){
		//cout<<stubs[sti]<<endl;
		tempVec.push_back( stubMapUsed[ stubs[sti] ] );
	      }

	      double pz = pt_fit/(tan(2.*atan(exp(-tracks[tt]->getEta0()))));
	      
	      TTTrack< Ref_PixelDigi_ > tempTrack( tempVec );
	      GlobalPoint POCA(0.,0.,tracks[tt]->getZ0());
	      GlobalVector mom(pt_fit*cos(tracks[tt]->getPhi0()),
			       pt_fit*sin(tracks[tt]->getPhi0()),
			       pz);

	      //	std::cout << tracks[tt]->getZ0() << " / " << POCA.z() << std::endl;


	      // kludge: We save the maximum weights in the chi2 variable
	      tempTrack.setChi2(tracks[tt]->getWxy(), 5);
	      tempTrack.setChi2(tracks[tt]->getWrz(), 4);

	      tempTrack.setRInv(tracks[tt]->getCurve(), 5);

	      tempTrack.setSector( sec_it->first );
	      tempTrack.setWedge( -1 );
	      tempTrack.setMomentum( mom , 5);
	      tempTrack.setPOCA( POCA , 5);
	      //	std::cout << tracks[tt]->getZ0() << " / " << POCA.z() << " / " << tempTrack.getPOCA().z() << std::endl;
	      TTTracksForOutput->push_back( tempTrack );
	    
	      delete tracks[tt];
	    }

	}
    
      delete(fitter);
    
    } // if ( fitPerTriggerTower_ )


    // =====================================================================================================
    //  Remove duplicate tracks:
    // =====================================================================================================
    if ( TTTracksForOutput->size()>1 && removeDuplicates_>0 ){

      
      for ( std::vector< TTTrack< Ref_PixelDigi_ > >::iterator itrk  = TTTracksForOutput->begin();
	                                                       itrk != TTTracksForOutput->end(); 
	                                                     ++itrk ){

	for ( std::vector< TTTrack< Ref_PixelDigi_ > >::iterator jtrk  = itrk+1;
	                                                         jtrk != TTTracksForOutput->end(); 
	                                                              ){ 

	  // Method I: identify duplicates cutting on deltaR
	  if ( removeDuplicates_ == 1 ) {

	    double delta_phi =  itrk->getMomentum(5).phi() - jtrk->getMomentum(5).phi();
	    if (fabs(delta_phi) > 3.14159265358979312)
	      delta_phi = 6.28318530717958623 - fabs(delta_phi);
	    double delta_eta =  itrk->getMomentum(5).eta() - jtrk->getMomentum(5).eta();

	    double delta_R = sqrt(delta_phi*delta_phi+delta_eta*delta_eta); 

	    if ( delta_R < 0.05 ){ 

	      double weight_itrk = itrk->getChi2(5) + itrk->getChi2(4);
	      double weight_jtrk = jtrk->getChi2(5) + jtrk->getChi2(4);

	      // Remove the duplicate with smaller weight:
	      if ( weight_itrk < weight_jtrk ){
		itrk = TTTracksForOutput->erase(itrk);
		--itrk;
		break;
	      } 
	      else { 
		jtrk = TTTracksForOutput->erase(jtrk);
		continue;
	      }

	    }

	  }
	  // Method II: identify duplicates checking whether the two tracks 
	  //            have more than one stub in common:
	  else if ( removeDuplicates_ == 2 ){ 

	    if ( itrk->isTheSameAs(*jtrk) ) {

	      double weight_itrk = itrk->getChi2(5) + itrk->getChi2(4);
	      double weight_jtrk = jtrk->getChi2(5) + jtrk->getChi2(4);

	      // Remove the duplicate with smaller weight:
	      if ( weight_itrk < weight_jtrk ){
		itrk = TTTracksForOutput->erase(itrk);
		--itrk;
		break;
	      } 
	      else { 
		jtrk = TTTracksForOutput->erase(jtrk);
		continue;
	      }

	    }

	  }

	  ++jtrk;

	} // loop over jtrk

      } // loop over itrk

    } // if ( TTTracksForOutput->size()>1 && removeDuplicates_>0 )

  
    
    // =====================================================================================================
    //  Printout the fitted tracks:
    // =====================================================================================================
    if ( TTTracksForOutput->size()>0 && verboseLevel_>0 ){
      
      if ( removeDuplicates_==0 )
	cout << "  Fitted tracks (no duplicate removal):" << endl;
      else
	cout << "  Fitted tracks (after duplicate removal):" << endl;
      
      for ( std::vector< TTTrack< Ref_PixelDigi_ > >::iterator itrk  = TTTracksForOutput->begin();
	                                                             itrk != TTTracksForOutput->end(); 
                                                                   ++itrk ){
	cout << "   "
	     << std::distance(TTTracksForOutput->begin(),itrk)
	     << "  -  c = " << itrk->getRInv(5)
	     << "  pt = " << itrk->getMomentum(5).perp()
	  //<< "  d0 = 0 " 
	     << "  phi = " << itrk->getMomentum(5).phi()
	     << "  eta = " << itrk->getMomentum(5).eta()
	     << "  z0 = "  << itrk->getPOCA(5).z() 
	     << endl;

      }
    
    } // if  ( TTTracksForOutput->size()>0 && verboseLevel_>0 )


    // Clean up the stubs map:
    for(map<int,vector<Hit*>*>::iterator sec_it  = m_hitsPerSector.begin();
	                                 sec_it != m_hitsPerSector.end();
	                               ++sec_it ){

      for(unsigned int i=0;i<sec_it->second->size();i++)
	delete sec_it->second->at(i);//delete the Hit object
      
      delete sec_it->second;//delete the vector*

    }


  } // if  ( TTPatternHandle->size() > 0 )


  /// Put in the event content
  iEvent.put( TTTracksForOutput, TTTrackOutputTag);

}

// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(TrackFitRetinaProducer);

#endif

