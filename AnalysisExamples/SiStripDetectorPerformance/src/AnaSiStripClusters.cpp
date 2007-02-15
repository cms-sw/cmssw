// Author : Samvel Khalatyan (samvel at fnal dot gov)
// Created: 01/04/07
// Licence: GPL
#include <iostream>

#include <TFile.h>
#include <TTree.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

/*
#include "Tutorial/Digis/interface/GetSiStripDigis.h"
#include "Tutorial/Clusters/interface/GetSiStripClusters.h"
#include "Tutorial/Clusters/interface/GetSiStripClusterEta.h"
*/

#include "AnalysisExamples/SiStripDetectorPerformance/interface/AnaSiStripClusters.h"

//   Constructor is the perfect place for getting variables values from 
// Configuration files.
// @argument
//    1  ParameterSet which is basically configurations manager
// @return
//    None
AnaSiStripClusters::AnaSiStripClusters( const edm::ParameterSet &roPARAMETER_SET)
  // Initialize internal variables with default values
  : oOutputFileName_( 
      roPARAMETER_SET.getUntrackedParameter<std::string>( "oOutputFileName")),
    oLblSiStripCluster_(
      roPARAMETER_SET.getUntrackedParameter<std::string>( 
	      "oLabelSiStripCluster")),
    oProdInstNameCluster_(
      roPARAMETER_SET.getUntrackedParameter<std::string>( 
        "oProdInstNameCluster")),
    oLblSiStripDigi_(
      roPARAMETER_SET.getUntrackedParameter<std::string>( 
	      "oLabelSiStripDigi")),
    oProdInstNameDigi_(
      roPARAMETER_SET.getUntrackedParameter<std::string>( "oProdInstNameDigi")),
    bMTCCMode_(
      roPARAMETER_SET.getUntrackedParameter<bool>( 
	      "bMTCCMode")),
    poOutputFile_( 0),
    poClusterTree_( 0) 
{}

AnaSiStripClusters::~AnaSiStripClusters() {}

// ----------------------------------------------------------------------------
//	PRIVATES
// ----------------------------------------------------------------------------
//    beginJob is called only once right before beginning any analyzes and is 
// the perfect place for opening any output files like root ones prepare class
// instance, initialize objects, extract magnetic field from EventSetup and 
// so on.
// @argument
//    1	  EventSetup
// @return
//    None
void AnaSiStripClusters::beginJob( const edm::EventSetup &roEVENT_SETUP) {
  // Open output ROOT file with leafs and histograms
  poOutputFile_ = new TFile( oOutputFileName_.c_str(), "RECREATE");

  // Create TTree and associate internal variables with leafs
  poClusterTree_ = new TTree( "ClusterTree", "Clusters Basic Values Tree");
  poClusterTree_->Branch( "nClusterModule", 
			  &( oCluster_.nModule), "nClusterModule/I");
  poClusterTree_->Branch( "nClusterSubdet", 
			  &( oCluster_.nSubdet), "nClusterSubdet/I");
  poClusterTree_->Branch( "nClusterLayer", 
			  &( oCluster_.nLayer), "nClusterLayer/I");
  poClusterTree_->Branch( "nClusterPos", 
			  &( oCluster_.nPosition), "nClusterPos/I");
  poClusterTree_->Branch( "nClusterWidth", 
			  &( oCluster_.nWidth), "nClusterWidth/I");
  poClusterTree_->Branch( "dClusterBarCen", 
			  &( oCluster_.dBaryCenter), "dClusterBarCen/F");
  poClusterTree_->Branch( "dClusterEta", 
			  &( oCluster_.dEta), "dClusterEta/F");
  poClusterTree_->Branch( "dClusterEtaTutorial", 
			  &( oCluster_.dEtaTutorial), "dClusterEtaTutorial/F");
  poClusterTree_->Branch( "nClusterCharge", 
			  &( oCluster_.nCharge), "nClusterCharge/I");

  double dVar = 5;
  std::cout << "dVar: " << dVar
            << "\t(uint16_t): " << ( uint16_t) dVar
	    << "\tstatic_cast<uint16_t>: " << static_cast<uint16_t>( dVar)
	    << std::endl;
  dVar = -5;
  std::cout << "dVar: " << dVar
            << "\t(uint16_t): " << ( uint16_t) dVar
	    << "\tstatic_cast<uint16_t>: " << static_cast<uint16_t>( dVar)
	    << std::endl;
}

//    analyze is called during processing Event's only once for each of them.
// @argument
//    1	  EventSetup
//    2	  Event object holding data containers
// @return
//    None
void AnaSiStripClusters::analyze( const edm::Event &roEVENT,
				                          const edm::EventSetup &roEVENT_SETUP) {

  // Extract all SiStripCluster's
  typedef edm::DetSetVector<SiStripCluster> DSVSiStripClusters;

  edm::Handle<DSVSiStripClusters> oDSVSiStripClusters;
  roEVENT.getByLabel( oLblSiStripCluster_.c_str(),
                      oProdInstNameCluster_.c_str(),
                      oDSVSiStripClusters);
  /*
  extra::getSiStripClusters( oDSVSiStripClusters,
			     roEVENT,
			     oLblSiStripCluster_);
  */

  // Extract all SiStripDigi's
  typedef edm::DetSetVector<SiStripDigi> DSVSiStripDigis;

  edm::Handle<DSVSiStripDigis> oDSVSiStripDigis;
  roEVENT.getByLabel( oLblSiStripDigi_.c_str(),
                      oProdInstNameDigi_.c_str(),
                      oDSVSiStripDigis);
  /*
  extra::getSiStripDigis( oDSVSiStripDigis,
			  roEVENT,
			  oLblSiStripDigi_,
			  oProdInstNameDigi_);
  */

  struct {
    // By default cTIB and cTOB are initialized with 0
    char cTIB;
    char cTOB;
  } oMTCCLayerCorr;

  if( bMTCCMode_) {
    oMTCCLayerCorr.cTIB = 1;
    oMTCCLayerCorr.cTOB = 2;
  }

  // Loop over Cluster's collection: keys are DetId's
  for( DSVSiStripClusters::const_iterator oDSVIter = 
	       oDSVSiStripClusters->begin();
       oDSVIter != oDSVSiStripClusters->end();
       ++oDSVIter) {

    // Get key that is DetId
    DetId oDetId( oDSVIter->id);

    // Get vector of Clusters that belong to a given DetId
    const std::vector<SiStripCluster> &roVCLUSTERS = oDSVIter->data;

    // Loop over Clusters in given DetId
    for( std::vector<SiStripCluster>::const_iterator oVIter = 
	         roVCLUSTERS.begin();
	       oVIter != roVCLUSTERS.end();
	       ++oVIter) {

      switch( oDetId.subdetId()) {
        case StripSubdetector::TIB:
          {
            TIBDetId oTIBDetId( oDetId.rawId());
            oCluster_.nLayer = oTIBDetId.layer() + oMTCCLayerCorr.cTIB;
            break;
          }
        case StripSubdetector::TOB:
          {
            TOBDetId oTOBDetId( oDetId.rawId());
            oCluster_.nLayer = oTOBDetId.layer() + oMTCCLayerCorr.cTOB;
            break;
          }
        default:
          continue;
      }

      // Fill leafs
      oCluster_.nModule	    = oDetId.rawId();
      oCluster_.nSubdet     = oDetId.subdetId();
      oCluster_.nPosition   = oVIter->firstStrip();
      oCluster_.nWidth	    = oVIter->amplitudes().size();
      oCluster_.dBaryCenter = oVIter->barycenter();

      const std::vector<uint16_t> &roCLUSTER_AMPLITUDES = oVIter->amplitudes();
      oCluster_.nCharge = 0;
      for( std::vector<uint16_t>::const_iterator oITER = roCLUSTER_AMPLITUDES.begin();
           oITER != roCLUSTER_AMPLITUDES.end();
	         ++oITER) {

        oCluster_.nCharge += *oITER;
      }

      oCluster_.dEta = 
        getClusterEta( roCLUSTER_AMPLITUDES,
                       oVIter->firstStrip(),
                       oDSVSiStripDigis->operator[]( 
                         oDetId.rawId()).data);

      oCluster_.dEtaTutorial = -1;

      // Fill Tree with leafs combination
      poClusterTree_->Fill();
    } // End loop over Clusters in specific DetId
  } // End loop over Global Clusters collection
}

//    endJob is called after all analyzes finished and usually used to fit any
// filled histograms, save output files, close any opened files, delete created
// objects, etc.
// @return
//    None
void AnaSiStripClusters::endJob() {
  // Write to disk and close output file
  poOutputFile_->Write();
  poOutputFile_->Close( "R");

  // Clean up memory: objects are not needed any more
  // delete poClusterTree_; // Damn ROOT: it removes TTree upon TFile::Close()
			    // call. There is no way programmer may control it
			    // in old fashion by deleting objects explicitly or
			    // using SmartPointers :(
  delete poOutputFile_;
}

// ClusterEta = SignalL / ( SignalL + SignalR)
// where:
//   SignalL and SignalR are two strips with the highest amplitudes. 
//   SignalL - is the strip with the smaller strip number
//   SignalR - with the highest strip number accordingly
// @arguments
//   roSTRIP_AMPLITUDES	 vector of strips ADC counts in cluster
//   rnFIRST_STRIP	 cluster first strip shift whithin module
//   roDIGIS		 vector of digis within current module
// @return
//   int  ClusterEta or -99 on error
double 
  AnaSiStripClusters::getClusterEta( const std::vector<uint16_t> &roSTRIP_AMPLITUDES,
				     const int			  &rnFIRST_STRIP,
				     const DigisVector		  &roDIGIS) const {

  /*
  std::string oOutStr( "ClusterPos: ");
  oOutStr += rnFIRST_STRIP;
  oOutStr += '\n';

  for( DigisVector::const_iterator oITER = 
  */

  // Given value is used to separate non-physical values
  // [Example: cluster with empty amplitudes vector]
  double dClusterEta = -99;

  // Cluster eta calculation
  int anMaxSignal[2][2];

  // Null array before using it
  for( int i = 0; 2 > i; ++i) {
    for( int j = 0; 2 > j; ++j) {
      anMaxSignal[i][j] = 0;
    }
  }

  struct {
    double operator()( const int &rnLEFT,
                       const int &rnRIGHT) {

      return ( 1.0 * rnLEFT) / ( rnLEFT + rnRIGHT);
    }
  } calcEta;
	
  // Find two strips with highest amplitudes
  // i is a stip number within module
  for( int i = 0, nSize = roSTRIP_AMPLITUDES.size(); nSize > i; ++i) {
    int nCurCharge = roSTRIP_AMPLITUDES[i];

    if( nCurCharge > anMaxSignal[1][1]) {
      anMaxSignal[0][0] = anMaxSignal[1][0];
      anMaxSignal[0][1] = anMaxSignal[1][1];
      // Convert to global strip number within module
      anMaxSignal[1][0] = i + rnFIRST_STRIP; 
      anMaxSignal[1][1] = nCurCharge;
    } else if( nCurCharge > anMaxSignal[0][1]) {
      // Convert to global strip number within module
      anMaxSignal[0][0] = i + rnFIRST_STRIP;
      anMaxSignal[0][1] = nCurCharge;
    }
  }
  
  if( ( anMaxSignal[1][1] + anMaxSignal[0][1]) != 0) {
    if( anMaxSignal[0][0] > anMaxSignal[1][0]) {
      // anMaxSignal[1] is Left one

      dClusterEta = calcEta( anMaxSignal[1][1], anMaxSignal[0][1]);
      /*
      dClusterEta = ( 1.0 * anMaxSignal[1][1]) / ( anMaxSignal[1][1] + 
						   anMaxSignal[0][1]);
      */
    } else if( 0 == anMaxSignal[0][0] && 
	       0 == anMaxSignal[0][1]) {

      // One Strip cluster: check for Digis
      DigisVector::const_iterator oITER( roDIGIS.begin());
      for( ;
	   oITER != roDIGIS.end() && oITER->strip() != anMaxSignal[1][0];
	   ++oITER) {}

      // Check if Digi for given cluster strip was found
      if( oITER != roDIGIS.end()) {

	// Check if previous neighbouring strip exists
	DigisVector::const_iterator oITER_PREV( roDIGIS.end());
	if( oITER != roDIGIS.begin() &&
	    ( oITER->strip() - 1) == ( oITER - 1)->strip()) {
	  // There is previous strip specified :)
	  oITER_PREV = oITER - 1;
	}

	// Check if next neighbouring strip exists
	DigisVector::const_iterator oITER_NEXT( roDIGIS.end());
	if( oITER != roDIGIS.end() &&
	    oITER != ( roDIGIS.end() - 1) &&
	    ( oITER->strip() + 1) == ( oITER + 1)->strip()) {
	  // There is previous strip specified :)
	  oITER_NEXT = oITER + 1;
	}

	if( oITER_PREV != oITER_NEXT) {
	  if( oITER_PREV != roDIGIS.end() && oITER_NEXT != roDIGIS.end()) {
	    // Both Digis are specified
	    // Now Pick the one with max amplitude
	    if( oITER_PREV->adc() > oITER_NEXT->adc()) {
              dClusterEta = calcEta( oITER_PREV->adc(), anMaxSignal[1][1]);
	      /*
	      dClusterEta = ( 1.0 * oITER_PREV->adc()) / ( oITER_PREV->adc() + 
							   anMaxSignal[1][1]);
	      */
	    } else {
              dClusterEta = calcEta( anMaxSignal[1][1], oITER_NEXT->adc());
	      /*
	      dClusterEta = ( 1.0 * anMaxSignal[1][1]) / ( oITER_NEXT->adc() + 
							   anMaxSignal[1][1]);
	      */
	    }
	  } else if( oITER_PREV != roDIGIS.end()) {
	    // only Prev digi is specified
            dClusterEta = calcEta( oITER_PREV->adc(), anMaxSignal[1][1]);
	    /*
	    dClusterEta = ( 1.0 * oITER_PREV->adc()) / ( oITER_PREV->adc() + 
							 anMaxSignal[1][1]);
	    */
	  } else {
	    // only Next digi is specified
            dClusterEta = calcEta( anMaxSignal[1][1], oITER_NEXT->adc());
	    /*
	    dClusterEta = ( 1.0 * anMaxSignal[1][1]) / ( oITER_NEXT->adc() + 
							 anMaxSignal[1][1]);
	    */
	  }
	} else {
	  // PREV and NEXT iterators point to the end of DIGIs vector. 
	  // Consequently it is assumed there are no neighbouring digis at all
	  // for given cluster. It is obvious why ClusterEta should be Zero.
	  // [Hint: take a look at the case [0][0] < [1][0] ]
	  dClusterEta = 0;
	} // End check if any neighbouring digi is specified
      } else {
	// Digi for given Clusters strip was not found
	dClusterEta = 0;
      } // end check if Digi for given cluster strip was found
    } else {
      // anMaxSignal[0] is Left one
      dClusterEta = calcEta( anMaxSignal[0][1], anMaxSignal[1][1]);
      /*
      dClusterEta = ( 1.0 * anMaxSignal[0][1]) / ( anMaxSignal[1][1] + 
						   anMaxSignal[0][1]);
      */
    }
  } 

  return dClusterEta;
}
