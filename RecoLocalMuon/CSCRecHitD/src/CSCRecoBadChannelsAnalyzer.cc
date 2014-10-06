/// Stand-alone test of bad strip channels testing via CSCRecoConditions
/// Tim Cox - 03-Oct-2014

#include <string>
#include <iostream>
#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibMuon/CSCCalibration/interface/CSCIndexerBase.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerRecord.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperBase.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperRecord.h"

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>

#include <RecoLocalMuon/CSCRecHitD/src/CSCRecoConditions.h>


  class CSCRecoBadChannelsAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  CSCRecoBadChannelsAnalyzer(edm::ParameterSet const& ps ) 
      : dashedLineWidth_( 80 ), 
        dashedLine_( std::string(dashedLineWidth_, '-') ), 
	myName_( "CSCRecoBadChannelsAnalyzer" ),
	readBadChannels_( ps.getParameter<bool>("readBadChannels") ),
        recoConditions_( new CSCRecoConditions( ps ) ) {
    }

    virtual ~CSCRecoBadChannelsAnalyzer() { delete recoConditions_; }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

    /// did we request reading bad channel info from db?
    bool readBadChannels() const { return readBadChannels_; }
    const std::string& myName() { return myName_;}

  private:

    const int dashedLineWidth_;
    const std::string dashedLine_;
    const std::string myName_;

    bool readBadChannels_; 
    CSCRecoConditions* recoConditions_;
  };
  
  void CSCRecoBadChannelsAnalyzer::analyze(const edm::Event& ev, const edm::EventSetup& evsetup )
  {
    using namespace edm::eventsetup;

    std::cout << myName() << "::analyze running..." << std::endl;
    std::cout << "start " << dashedLine_ << std::endl;

    std::cout << "RUN# " << ev.id().run() << std::endl;
    std::cout << "EVENT# " << ev.id().event() << std::endl;

    edm::ESHandle<CSCIndexerBase> theIndexer;
    evsetup.get<CSCIndexerRecord>().get(theIndexer);

    std::cout << myName() << "::analyze sees indexer " << theIndexer->name()  << " in Event Setup" << std::endl;

    edm::ESHandle<CSCChannelMapperBase> theMapper;
    evsetup.get<CSCChannelMapperRecord>().get(theMapper);

    std::cout << myName() << "::analyze sees mapper " << theMapper->name()  << " in Event Setup" << std::endl;

    edm::ESHandle<CSCGeometry> theGeometry;
    evsetup.get<MuonGeometryRecord>().get( theGeometry );     

    std::cout << " Geometry node for CSCGeom is  " << &(*theGeometry) << std::endl;   
    std::cout << " There are " << theGeometry->dets().size() << " dets" << std::endl;
    std::cout << " There are " << theGeometry->detTypes().size() << " types" << "\n" << std::endl;

    // INITIALIZE CSCConditions
    recoConditions_->initializeEvent(evsetup);

    // HERE NEED TO ITERATE OVER ALL CSCDetId

    std::cout << myName() << ": Begin iteration over geometry..." << std::endl;

    const CSCGeometry::LayerContainer& vecOfLayers = theGeometry->layers();
    std::cout << "There are " << vecOfLayers.size() << " layers" << std::endl;

    std::cout << dashedLine_ << std::endl;

    int ibadchannels = 0; // COUNT OF BAD STRIP CHANNELS
    int ibadlayers = 0 ; //COUNT OF LAYERS WITH BAD STRIP CHANNELS

    for( auto it = vecOfLayers.begin(); it != vecOfLayers.end(); ++it ){

      const CSCLayer* layer = *it;

      if( layer ){
        CSCDetId id = layer->id();
        int nstrips = layer->geometry()->numberOfStrips();
        std::cout << "Layer " << id << " has " << nstrips << " strips" << std::endl;

	// GET BAD CHANNELS FOR THIS LAYER

        recoConditions_->fillBadChannelWords( id );

	// SEARCH FOR BAD STRIP CHANNELS IN THIS LAYER - GEOMETRIC STRIP INPUT!!

	bool layerhasbadchannels = false;
	for ( short is = 1; is<=nstrips; ++is ) {
	  if ( recoConditions_->badStrip( id, is, nstrips ) ) {
              ++ibadchannels;
	      layerhasbadchannels = true;
	      std::cout << id << " strip " << is << " is bad" << std::endl;
	    }            
	}

	for ( short is = 1; is<=nstrips; ++is ) {
	  if ( recoConditions_->nearBadStrip( id, is, nstrips ) ) {
	    std::cout << id << " strip " << is << " is a neighbor of a bad strip" << std::endl;
	    }            
	}

	if (layerhasbadchannels) ++ibadlayers;

      }
      else {
	std::cout << "WEIRD ERROR: a null CSCLayer* was seen" << std::endl;
      }
    }

    std::cout << "No. of layers with bad strip channels = " << ibadlayers << std::endl;
    std::cout << "No. of bad strip channels seen = " << ibadchannels << std::endl;

    std::cout << dashedLine_ << " end" << std::endl;
  }

  DEFINE_FWK_MODULE(CSCRecoBadChannelsAnalyzer);


