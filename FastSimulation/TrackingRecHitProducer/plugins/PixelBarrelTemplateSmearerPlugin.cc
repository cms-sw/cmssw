
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
    isForward = false;

    theBigPixelResolutionFileName = config.getParameter<string>( "BigPixelBarrelResolutionFile" );
    theBigPixelResolutionFile = std::make_unique<TFile>( edm::FileInPath( theBigPixelResolutionFileName ).fullPath().c_str()  ,"READ");
    theEdgePixelResolutionFileName =  config.getParameter<string>( "EdgePixelBarrelResolutionFile" );
    theEdgePixelResolutionFile = std::make_unique<TFile>( edm::FileInPath( theEdgePixelResolutionFileName ).fullPath().c_str()  ,"READ");
    theRegularPixelResolutionFileName = config.getParameter<string>( "RegularPixelBarrelResolutionFile" );
    theRegularPixelResolutionFile = std::make_unique<TFile>( edm::FileInPath( theRegularPixelResolutionFileName ).fullPath().c_str()  ,"READ");
    
    theMergingProbabilityFileName = config.getParameter<string>( "MergingProbabilityBarrelFile" );
    theMergingProbabilityFile = std::make_unique<TFile>( edm::FileInPath( theMergingProbabilityFileName ).fullPath().c_str()  ,"READ");
    theMergedPixelResolutionXFileName = config.getParameter<string>( "MergedPixelBarrelResolutionXFile" );
    theMergedPixelResolutionXFile = std::make_unique<TFile>( edm::FileInPath( theMergedPixelResolutionXFileName ).fullPath().c_str()  ,"READ");
    theMergedPixelResolutionYFileName = config.getParameter<string>( "MergedPixelBarrelResolutionYFile" );
    theMergedPixelResolutionYFile = std::make_unique<TFile>( edm::FileInPath( theMergedPixelResolutionYFileName ).fullPath().c_str()  ,"READ");

    initializeBarrel();
    
    if (!SiPixelTemplate::pushfile(templateId, thePixelTemp_))
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
            theXHistos[singleBigPixelHistN] = new SimpleHistogramGenerator((TH1F*)theBigPixelResolutionFile->Get(Form("DQMData/clustBPIX/hx%u",singleBigPixelHistN)));
            theYHistos[singleBigPixelHistN] = new SimpleHistogramGenerator((TH1F*)theBigPixelResolutionFile->Get(Form("DQMData/clustBPIX/hy%u",singleBigPixelHistN)));

            unsigned int singlePixelHistN = 1*10000 + cotbetaHistBin*10 + cotalphaHistBin;
            theXHistos[singlePixelHistN] = new SimpleHistogramGenerator((TH1F*)theRegularPixelResolutionFile->Get(Form("hx%u",singlePixelHistN)));
            theYHistos[singlePixelHistN] = new SimpleHistogramGenerator((TH1F*)theRegularPixelResolutionFile->Get(Form("hy%u",singlePixelHistN)));
            
            for(unsigned qbinBin=1; qbinBin<=resqbin_binN; ++qbinBin )
            {
                unsigned int edgePixelHistN = cotalphaHistBin*1000 + cotbetaHistBin*10 + qbinBin;
                theXHistos[edgePixelHistN] = new SimpleHistogramGenerator((TH1F*)theEdgePixelResolutionFile->Get(Form("DQMData/clustBPIX/hx0%u",edgePixelHistN)));
                theYHistos[edgePixelHistN] = new SimpleHistogramGenerator((TH1F*)theEdgePixelResolutionFile->Get(Form("DQMData/clustBPIX/hy0%u",edgePixelHistN)));
                
                unsigned int multiPixelBigHistN = 1*1000000 + 1*100000 + cotalphaHistBin*1000 + cotbetaHistBin * 10 + qbinBin;
                theXHistos[multiPixelBigHistN] = new SimpleHistogramGenerator((TH1F*)theBigPixelResolutionFile->Get(Form("DQMData/clustBPIX/hx%u",multiPixelBigHistN)));
                theYHistos[multiPixelBigHistN] = new SimpleHistogramGenerator((TH1F*)theBigPixelResolutionFile->Get(Form("DQMData/clustBPIX/hy%u",multiPixelBigHistN)));
                
                unsigned int multiPixelHistN = 1*100000 + 1*10000 + cotbetaHistBin*100 + cotalphaHistBin*10 + qbinBin;
                theXHistos[multiPixelHistN] = new SimpleHistogramGenerator((TH1F*)theRegularPixelResolutionFile->Get(Form("hx%u",multiPixelHistN)));
                theYHistos[multiPixelHistN] = new SimpleHistogramGenerator((TH1F*)theRegularPixelResolutionFile->Get(Form("hy%u",multiPixelHistN)));
            }
        }
    }
}


DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    PixelBarrelTemplateSmearerPlugin,
    "PixelBarrelTemplateSmearerPlugin"
);
