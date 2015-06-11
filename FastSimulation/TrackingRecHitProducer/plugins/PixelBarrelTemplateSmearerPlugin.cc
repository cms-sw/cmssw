
/** PixelBarrelTemplateSmearerPlugin.cc
 * ---------------------------------------------------------------------
 * Description:  see PixelBarrelTemplateSmearerPlugin.h
 * Authors:  R. Ranieri (CERN), M. Galanti
 * History: Oct 11, 2006 -  initial version
 * 
 * New Pixel Resolution Parameterization 
 * Introduce SiPixelTemplate Object to Assign Pixel Errors
 * by G. Hu
 * ---------------------------------------------------------------------
 */

// SiPixel Gaussian Smearing
#include "FastSimulation/TrackingRecHitProducer/interface/PixelTemplateSmearerBase.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

// Geometry
//#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
//#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"

// Famos
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/Utilities/interface/SimpleHistogramGenerator.h"

// STL

// ROOT
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
//#include <TAxis.h>

//#define FAMOS_DEBUG

using namespace std;

class PixelBarrelTemplateSmearerPlugin : public PixelTemplateSmearerBase
{
public:
  explicit PixelBarrelTemplateSmearerPlugin( const std::string& name,
					     const edm::ParameterSet& config,
					     edm::ConsumesCollector& consumesCollector
					     );
  virtual ~PixelBarrelTemplateSmearerPlugin();
  
private:
  void initializeBarrel();
};



//------------------------------------------------------------------------------
//  Constructor. In this case, only one is possible, since it needs to
//  match the base class in order to be made by the factory.
//
// &&& Need to change the signature of the constructor...
// &&& Who is calling this one?  What is the pixelPart ?
//------------------------------------------------------------------------------
PixelBarrelTemplateSmearerPlugin::PixelBarrelTemplateSmearerPlugin(
  const std::string& name,
  const edm::ParameterSet& config,
  edm::ConsumesCollector& consumesCollector 
  ) :
  PixelTemplateSmearerBase(name,config,consumesCollector)
{
  setPixelPart(GeomDetEnumerators::PixelBarrel);
  std::cout << "PixelBarrelTemplateSmearerPlugin" << std::endl;

  isForward = false;
  thePixelResolutionFileName1 = pset_.getParameter<string>( "NewPixelBarrelResolutionFile1" );
  thePixelResolutionFile1 = new 
    TFile( edm::FileInPath( thePixelResolutionFileName1 ).fullPath().c_str()  ,"READ");
  thePixelResolutionFileName2 =  pset_.getParameter<string>( "NewPixelBarrelResolutionFile2" );
  thePixelResolutionFile2 = new
    TFile( edm::FileInPath( thePixelResolutionFileName2 ).fullPath().c_str()  ,"READ");
  //ALICE: loading new pixel barrel resolution file as defined in ../python/SiTrackerGaussianSmearingRecHitConverter_cfi.py 
  thePixelResolutionFileName3 = pset_.getParameter<string>( "NewPixelBarrelResolutionFile3" );
  thePixelResolutionFile3 = new 
    TFile( edm::FileInPath( thePixelResolutionFileName3 ).fullPath().c_str()  ,"READ");
  probfileName = pset_.getParameter<string>( "probfilebarrel" );
  probfile = new
    TFile( edm::FileInPath( probfileName ).fullPath().c_str()  ,"READ");
  initializeBarrel();
  tempId = pset_.getParameter<int> ( "templateIdBarrel" );
  if( ! SiPixelTemplate::pushfile(tempId, thePixelTemp_) )
    throw cms::Exception("PixelBarrelTemplateSmearerPlugin:")
      <<"SiPixel Barrel Template Not Loaded Correctly!"<<endl;

#ifdef  FAMOS_DEBUG
  cout<<"The Barrel map size is "<<theXHistos.size()<<" and "<<theYHistos.size()<<endl;
#endif

  std::cout << "end P PixelBarrelTemplateSmearerPlugin"<< std::endl;
}



//------------------------------------------------------------------------------
//  Destructor.  Empty, because the base class clears all histograms.
//------------------------------------------------------------------------------
PixelBarrelTemplateSmearerPlugin::~PixelBarrelTemplateSmearerPlugin()
{
}



//-----------------------------------------------------------------------------
//   Initialize Pixel Barrel info.
//-----------------------------------------------------------------------------
void PixelBarrelTemplateSmearerPlugin::initializeBarrel()
{
  std::cout << "P initializeBarrel"<< std::endl;
  //Hard coded at the moment, can easily be changed to be configurable
  rescotAlpha_binMin = -0.2;
  rescotAlpha_binWidth = 0.08 ;
  rescotAlpha_binN = 5;
  rescotBeta_binMin = -5.5;
  rescotBeta_binWidth = 1.0;
  rescotBeta_binN = 11;
  resqbin_binMin = 0;
  resqbin_binWidth = 1;
  resqbin_binN = 4;

  // Initialize the barrel histos once and for all, and prepare the random generation
  for ( unsigned cotalphaHistBin=1; cotalphaHistBin<=rescotAlpha_binN; ++cotalphaHistBin ) 
     for ( unsigned cotbetaHistBin=1; cotbetaHistBin<=rescotBeta_binN; ++cotbetaHistBin )  {
         unsigned int singleBigPixelHistN = 1 * 100000
                                        + cotalphaHistBin * 100
                                        + cotbetaHistBin ;
	 theXHistos[singleBigPixelHistN] = new SimpleHistogramGenerator(
										      (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hx%u" , singleBigPixelHistN ) ) ); 
	 theYHistos[singleBigPixelHistN] = new SimpleHistogramGenerator(
									              (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hy%u" , singleBigPixelHistN ) ) );
	 //	 	 unsigned int singlePixelHistN = 1 * 100000 + 1 * 1000
	 //                               + cotalphaHistBin * 100
	 //                              + cotbetaHistBin ;
	 // theXHistos[singlePixelHistN] = new SimpleHistogramGenerator(
	 //							     (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hx%u" , singlePixelHistN ) ) );
	 // theYHistos[singlePixelHistN] = new SimpleHistogramGenerator(
	 //							     (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hy%u" , singlePixelHistN ) ) );
	 //ALICE: getting new histogram
	 unsigned int singlePixelHistN = 1 * 10000 
	                               + cotbetaHistBin * 10
	                               + cotalphaHistBin ;
	 theXHistos[singlePixelHistN] = new SimpleHistogramGenerator(
	 							                   (TH1F*) thePixelResolutionFile3->Get(  Form( "hx%u" , singlePixelHistN ) ) );
	 theYHistos[singlePixelHistN] = new SimpleHistogramGenerator(
	 							                   (TH1F*) thePixelResolutionFile3->Get(  Form( "hy%u" , singlePixelHistN ) ) );
	 for( unsigned qbinBin=1;  qbinBin<=resqbin_binN; ++qbinBin )  {
		       unsigned int edgePixelHistN = cotalphaHistBin * 1000
		                        +  cotbetaHistBin * 10
                                        +  qbinBin;
             theXHistos[edgePixelHistN] = new SimpleHistogramGenerator(
								                     (TH1F*) thePixelResolutionFile2->Get(  Form( "DQMData/clustBPIX/hx0%u" ,edgePixelHistN ) ) );
	     theYHistos[edgePixelHistN] = new SimpleHistogramGenerator(
								                    (TH1F*) thePixelResolutionFile2->Get(  Form( "DQMData/clustBPIX/hy0%u" ,edgePixelHistN ) ) );
	     unsigned int multiPixelBigHistN = 1 * 1000000 + 1 * 100000
	                                    + cotalphaHistBin * 1000
                                           + cotbetaHistBin * 10
                                           + qbinBin;
             theXHistos[multiPixelBigHistN] = new SimpleHistogramGenerator(
									                 (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hx%u" ,multiPixelBigHistN ) ) );
	      theYHistos[multiPixelBigHistN] = new SimpleHistogramGenerator(
									                 (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hy%u" ,multiPixelBigHistN ) ) );
	      //   unsigned int multiPixelHistN = 1 * 1000000 + 1 * 100000 + 1 * 10000
	      //	                            + cotalphaHistBin * 1000
	      //	                            + cotbetaHistBin * 10
	      //	                            + qbinBin;
	      // theXHistos[multiPixelHistN] = new SimpleHistogramGenerator(
	      //								 (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hx%u" , multiPixelHistN ) ) );
	      //theYHistos[multiPixelHistN] = new SimpleHistogramGenerator(
	      //								 (TH1F*) thePixelResolutionFile1->Get(  Form( "DQMData/clustBPIX/hy%u" , multiPixelHistN ) ) );
	      //ALICE: getting new histogram
	           unsigned int multiPixelHistN = 1 * 100000 + 1 * 10000
	                                 + cotbetaHistBin * 100
	                                 + cotalphaHistBin * 10
	                                 + qbinBin;
	        theXHistos[multiPixelHistN] = new SimpleHistogramGenerator(
	      								             (TH1F*) thePixelResolutionFile3->Get(  Form( "hx%u" , multiPixelHistN ) ) );
	       theYHistos[multiPixelHistN] = new SimpleHistogramGenerator(
	      								             (TH1F*) thePixelResolutionFile3->Get(  Form( "hy%u" , multiPixelHistN ) ) );
	   	 } //end for qbinBin
     }//end for cotalphaHistBin, cotbetaHistBin
  std::cout << "endP initializeBarrel"<< std::endl;
}




//-----------------------------------------------------------------------------
//   Declare this class to be a plugin, creatable by TrackingRecHitAlgorithmFactory
//-----------------------------------------------------------------------------
DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    PixelBarrelTemplateSmearerPlugin,
    "PixelBarrelTemplateSmearerPlugin"
);
