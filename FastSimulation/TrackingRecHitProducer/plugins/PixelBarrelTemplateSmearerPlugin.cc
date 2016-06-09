
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


using namespace std;

class PixelBarrelTemplateSmearerPlugin:
    public PixelTemplateSmearerBase
{
    public:
        explicit PixelBarrelTemplateSmearerPlugin(
            const std::string& name,
            const edm::ParameterSet& config,
            edm::ConsumesCollector& consumesCollector
        );
        virtual ~PixelBarrelTemplateSmearerPlugin();

    private:
        void initializeBarrel();
};


PixelBarrelTemplateSmearerPlugin::PixelBarrelTemplateSmearerPlugin(
    const std::string& name,
    const edm::ParameterSet& config,
    edm::ConsumesCollector& consumesCollector
):
    PixelTemplateSmearerBase(name,config,consumesCollector)
{
    setPixelPart(GeomDetEnumerators::PixelBarrel);

    isForward = false;
    thePixelResolutionFileName1 = config.getParameter<string>( "NewPixelBarrelResolutionFile1" );
    thePixelResolutionFile1 = new TFile( edm::FileInPath( thePixelResolutionFileName1 ).fullPath().c_str()  ,"READ");
    thePixelResolutionFileName2 =  config.getParameter<string>( "NewPixelBarrelResolutionFile2" );
    thePixelResolutionFile2 = new TFile( edm::FileInPath( thePixelResolutionFileName2 ).fullPath().c_str()  ,"READ");
    thePixelResolutionFileName3 = config.getParameter<string>( "NewPixelBarrelResolutionFile3" );
    thePixelResolutionFile3 = new TFile( edm::FileInPath( thePixelResolutionFileName3 ).fullPath().c_str()  ,"READ");
    
    probfileName = config.getParameter<string>( "probfilebarrel" );
    probfile = new TFile( edm::FileInPath( probfileName ).fullPath().c_str()  ,"READ");
    thePixelResolutionMergedXFileName = config.getParameter<string>( "pixelresxmergedbarrel" );
    thePixelResolutionMergedXFile = new TFile( edm::FileInPath( thePixelResolutionMergedXFileName ).fullPath().c_str()  ,"READ");
    thePixelResolutionMergedYFileName = config.getParameter<string>( "pixelresymergedbarrel" );
    thePixelResolutionMergedYFile = new TFile( edm::FileInPath( thePixelResolutionMergedYFileName ).fullPath().c_str()  ,"READ");
    initializeBarrel();
    tempId = config.getParameter<int> ( "templateIdBarrel" );
    
    if (!SiPixelTemplate::pushfile(tempId, thePixelTemp_))
    {
        throw cms::Exception("PixelBarrelTemplateSmearerPlugin:")<<"SiPixel Barrel Template Not Loaded Correctly!"<<endl;
    }

}


PixelBarrelTemplateSmearerPlugin::~PixelBarrelTemplateSmearerPlugin()
{
}


void PixelBarrelTemplateSmearerPlugin::initializeBarrel()
{
    rescotAlpha_binMin = -0.2;
    rescotAlpha_binWidth = 0.08 ;
    rescotAlpha_binN = 5;
    rescotBeta_binMin = -5.5;
    rescotBeta_binWidth = 1.0;
    rescotBeta_binN = 11;
    resqbin_binMin = 0;
    resqbin_binWidth = 1;
    resqbin_binN = 4;

    for (unsigned cotalphaHistBin=1; cotalphaHistBin<=rescotAlpha_binN; ++cotalphaHistBin)
    {
        for (unsigned cotbetaHistBin=1; cotbetaHistBin<=rescotBeta_binN; ++cotbetaHistBin)
        {
            unsigned int singleBigPixelHistN = 1*100000 + cotalphaHistBin*100 + cotbetaHistBin;
            theXHistos[singleBigPixelHistN] = new SimpleHistogramGenerator((TH1F*)thePixelResolutionFile1->Get(Form("DQMData/clustBPIX/hx%u",singleBigPixelHistN)));
            theYHistos[singleBigPixelHistN] = new SimpleHistogramGenerator((TH1F*)thePixelResolutionFile1->Get(Form("DQMData/clustBPIX/hy%u",singleBigPixelHistN)));

            unsigned int singlePixelHistN = 1*10000 + cotbetaHistBin*10 + cotalphaHistBin;
            theXHistos[singlePixelHistN] = new SimpleHistogramGenerator((TH1F*)thePixelResolutionFile3->Get(Form("hx%u",singlePixelHistN)));
            theYHistos[singlePixelHistN] = new SimpleHistogramGenerator((TH1F*)thePixelResolutionFile3->Get(Form("hy%u",singlePixelHistN)));
            
            for(unsigned qbinBin=1; qbinBin<=resqbin_binN; ++qbinBin )
            {
                unsigned int edgePixelHistN = cotalphaHistBin*1000 + cotbetaHistBin*10 + qbinBin;
                theXHistos[edgePixelHistN] = new SimpleHistogramGenerator((TH1F*)thePixelResolutionFile2->Get(Form("DQMData/clustBPIX/hx0%u",edgePixelHistN)));
                theYHistos[edgePixelHistN] = new SimpleHistogramGenerator((TH1F*)thePixelResolutionFile2->Get(Form("DQMData/clustBPIX/hy0%u",edgePixelHistN)));
                
                unsigned int multiPixelBigHistN = 1*1000000 + 1*100000 + cotalphaHistBin*1000 + cotbetaHistBin * 10 + qbinBin;
                theXHistos[multiPixelBigHistN] = new SimpleHistogramGenerator((TH1F*)thePixelResolutionFile1->Get(Form("DQMData/clustBPIX/hx%u",multiPixelBigHistN)));
                theYHistos[multiPixelBigHistN] = new SimpleHistogramGenerator((TH1F*)thePixelResolutionFile1->Get(Form("DQMData/clustBPIX/hy%u",multiPixelBigHistN)));
                
                unsigned int multiPixelHistN = 1*100000 + 1*10000 + cotbetaHistBin*100 + cotalphaHistBin*10 + qbinBin;
                theXHistos[multiPixelHistN] = new SimpleHistogramGenerator((TH1F*)thePixelResolutionFile3->Get(Form("hx%u",multiPixelHistN)));
                theYHistos[multiPixelHistN] = new SimpleHistogramGenerator((TH1F*)thePixelResolutionFile3->Get(Form("hy%u",multiPixelHistN)));
            }
        }
    }
}


DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    PixelBarrelTemplateSmearerPlugin,
    "PixelBarrelTemplateSmearerPlugin"
);
