// system include files
#include <vector>

#include <TRandom.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

// #include "Tutorial/Digis/interface/GetSiStripDigisFwd.h"
// #include "Tutorial/Digis/interface/GetSiStripDigis.h"

#include "AnalysisExamples/SiStripDetectorPerformance/interface/MTCCAmplifyDigis.h"

MTCCAmplifyDigis::MTCCAmplifyDigis( const edm::ParameterSet &roPARAMETER_SET):
  oSiStripDigisLabel_( 
    roPARAMETER_SET.getUntrackedParameter<std::string>( 
      "oSiStripDigisLabel")),
  oSiStripDigisProdInstName_( 
    roPARAMETER_SET.getUntrackedParameter<std::string>( 
      "oSiStripDigisProdInstName")),
  oNewSiStripDigisLabel_( 
    roPARAMETER_SET.getUntrackedParameter<std::string>( 
      "oNewSiStripDigisLabel"))
{
  // Declare what collection will given producer create
  produces<edm::DetSetVector<SiStripDigi> >();

  // Exract DIGI's amply sigmas
  const edm::ParameterSet oPSDIGI_AMPLIFY_SIGMAS =
    roPARAMETER_SET.getUntrackedParameter<edm::ParameterSet>(
      "oDigiAmplifySigmas");

  oDigiAmplifySigma_.dTIB = 
    oPSDIGI_AMPLIFY_SIGMAS.getUntrackedParameter<double>( "dTIB");
  oDigiAmplifySigma_.dTOB = 
    oPSDIGI_AMPLIFY_SIGMAS.getUntrackedParameter<double>( "dTOB");

  // Extract DIGI's scale factors for diffeferent layers
  const edm::ParameterSet oPSDIGI_SCALE_FACTORS = 
    roPARAMETER_SET.getUntrackedParameter<edm::ParameterSet>( 
      "oDigiScaleFactors");

  // TIB
  {
    const edm::ParameterSet oPS_SUBDET =
      oPSDIGI_SCALE_FACTORS.getUntrackedParameter<edm::ParameterSet>(
	"oTIB");
    oDigiScaleFactor_.oTIB.dL1 = 
      oPS_SUBDET.getUntrackedParameter<double>( "dL1");
    oDigiScaleFactor_.oTIB.dL2 = 
      oPS_SUBDET.getUntrackedParameter<double>( "dL2");
  }

  // TOB
  {
    const edm::ParameterSet oPS_SUBDET =
      oPSDIGI_SCALE_FACTORS.getUntrackedParameter<edm::ParameterSet>(
	"oTOB");
    oDigiScaleFactor_.oTOB.dL1 = 
      oPS_SUBDET.getUntrackedParameter<double>( "dL1");
    oDigiScaleFactor_.oTOB.dL2 = 
      oPS_SUBDET.getUntrackedParameter<double>( "dL2");
  }

  LogDebug( "MTCCAmplifyDigis::MTCCAmplifyDigis")
    << "Scale Factors" << "\n"
    << "TIB" << "\n"
    << "\t\tL1: " << oDigiScaleFactor_.oTIB.dL1 << "\n"
    << "\t\tL2: " << oDigiScaleFactor_.oTIB.dL2 << "\n"
    << "TOB" << "\n"
    << "\t\tL1: " << oDigiScaleFactor_.oTOB.dL1 << "\n"
    << "\t\tL2: " << oDigiScaleFactor_.oTOB.dL2;
}


MTCCAmplifyDigis::~MTCCAmplifyDigis() {}

void MTCCAmplifyDigis::produce( edm::Event &roEvent, 
                                const edm::EventSetup &roEVENT_SETUP) {
  // Extract all SiStripDigi's
  typedef edm::DetSetVector<SiStripDigi> DSVSiStripDigis;

  edm::Handle<DSVSiStripDigis> oDSVSiStripDigis;
  roEvent.getByLabel( oSiStripDigisLabel_.c_str(),
                      oSiStripDigisProdInstName_.c_str(),
                      oDSVSiStripDigis);
  /*
  extra::getSiStripDigis( oDSVSiStripDigis,
			  roEvent,
			  oSiStripDigisLabel_,
			  oSiStripDigisProdInstName_);
  */

  // Declare inline function
  struct {
    SiStripDigi operator()( const SiStripDigi &roDIGI,
                            const double      &rdSIGMA,
			                      const double      &rdSCALE) {
      static TRandom oTRandom;

      // 1. Apply GAUSS to digi ADC
      // 2. Scale result
      double dNewAdc = rdSCALE *
			                 oTRandom.Gaus( roDIGI.adc(),
					                            rdSIGMA);

      if( 0 >= dNewAdc) {
        dNewAdc = 0;
      }

      return SiStripDigi( roDIGI.strip(),
                          static_cast<uint16_t>( dNewAdc));
    }
  } amplifyDigi;

  // Create new empty collection for Amplified Digis
  std::vector<edm::DetSet<SiStripDigi> > oVDSAmplifiedDigis;

  // Amplify Original Digis
  // Loop over Digi's collection: keys are DetId's
  for( DSVSiStripDigis::const_iterator oDSVIter = 
         oDSVSiStripDigis->begin();
       oDSVIter != oDSVSiStripDigis->end();
       ++oDSVIter) {

    typedef std::vector<SiStripDigi> DigisVector;

    // Get vector of Digis that belong to a given DetId
    const DigisVector &roVDIGIS = oDSVIter->data;

    // Create new collection where Amplified Digis for given module will be
    // stored
    edm::DetSet<SiStripDigi> oNewDSDigis( oDSVIter->id);

    // 1. Loop over and perform: Digis -> NewDigis 
    for( DigisVector::const_iterator oDIGI_ITER = roVDIGIS.begin();
	 oDIGI_ITER != roVDIGIS.end();
	 ++oDIGI_ITER) {

      // Determine which scale factor to be used: different for different
      // layers
      double dDigiScaleFactor;
      double dDigiAmplifySigma;
      DetId oDetId( oDSVIter->id);

      switch( oDetId.subdetId()) {
	case StripSubdetector::TIB:
	  {
	    TIBDetId oTIBDetId( oDSVIter->id);
	    
	    // Possible layers: 1 and 2
	    switch( oTIBDetId.layer()) {
	      case 1:
	        dDigiScaleFactor = oDigiScaleFactor_.oTIB.dL1;
	        break;
	      case 2:
	        dDigiScaleFactor = oDigiScaleFactor_.oTIB.dL2;
	        break;
	      default:
	        // Hmm, some other layer: skip analysis
		continue;
	    }

            dDigiAmplifySigma = oDigiAmplifySigma_.dTIB;
	    break;
	  }
	case StripSubdetector::TOB:
	  {
	    TOBDetId oTOBDetId( oDSVIter->id);

	    // Possible layers: 1 and 2
	    switch( oTOBDetId.layer()) {
	      case 1:
	        dDigiScaleFactor = oDigiScaleFactor_.oTOB.dL1;
	        break;
	      case 2:
	        dDigiScaleFactor = oDigiScaleFactor_.oTOB.dL2;
	        break;
	      default:
	        // Hmm, some other layer: skip analysis
		continue;
	    }

            dDigiAmplifySigma = oDigiAmplifySigma_.dTOB;
	    break;
	  }
	default:
	  // Unwanted SubDet: skip analysis
	  continue;
      }

      // Amplify digi
      oNewDSDigis.push_back( amplifyDigi( *oDIGI_ITER,
                                           dDigiAmplifySigma,
					   dDigiScaleFactor));
    }

    oVDSAmplifiedDigis.push_back( oNewDSDigis);
  }

  // Create Amplified Digis collection
  std::auto_ptr<DSVSiStripDigis > 
    oAmplifiedDigis( new DSVSiStripDigis( oVDSAmplifiedDigis));

  // Write Amplified Digis to Event
  roEvent.put( oAmplifiedDigis);
}
