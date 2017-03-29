######################################################################
######################################################################
intoNTuplesTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("ValidationIntoNTuples")

# global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo." 

process.load("Configuration.Geometry.GeometryRecoDB_cff")

process.load("CondCore.CondDB.CondDB_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo', 
        'cout')
) 

.oO[condLoad]Oo.

process.source = cms.Source("EmptySource",
    firstRun=cms.untracked.uint32(.oO[runGeomComp]Oo.)
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.dump = cms.EDAnalyzer("TrackerGeometryIntoNtuples",
    outputFile = cms.untracked.string('.oO[alignmentName]Oo.ROOTGeometry.root'),
    outputTreename = cms.untracked.string('alignTree')
)

process.p = cms.Path(process.dump)  
"""


######################################################################
######################################################################
compareTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("validation")

# global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo." 

process.load("Configuration.Geometry.GeometryRecoDB_cff")

process.load("CondCore.CondDB.CondDB_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo', 
        'cout')
)

process.source = cms.Source("EmptySource",
    firstRun=cms.untracked.uint32(.oO[runGeomComp]Oo.)
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.siStripQualityESProducer.ListOfRecordToMerge=cms.VPSet(
	cms.PSet(record = cms.string('SiStripDetCablingRcd'),
            tag = cms.string('')), 
        cms.PSet(record = cms.string('RunInfoRcd'),
            tag = cms.string('')), 
        cms.PSet(record = cms.string('SiStripBadChannelRcd'),
            tag = cms.string('')), 
        cms.PSet(record = cms.string('SiStripBadFiberRcd'),
            tag = cms.string('')), 
        cms.PSet(record = cms.string('SiStripBadModuleRcd'),
            tag = cms.string('')), 
        cms.PSet(record = cms.string('SiStripBadStripRcd'),
            tag = cms.string(''))
)

process.load("DQM.SiStripCommon.TkHistoMap_cfi")

process.DQMStore=cms.Service("DQMStore")

process.load("DQMServices.Core.DQMStore_cfg") 

  # configuration of the Tracker Geometry Comparison Tool
  # Tracker Geometry Comparison
process.load("Alignment.OfflineValidation.TrackerGeometryCompare_cfi")
  # the input "IDEAL" is special indicating to use the ideal geometry of the release

process.TrackerGeometryCompare.inputROOTFile1 = '.oO[comparedGeometry]Oo.'
process.TrackerGeometryCompare.inputROOTFile2 = '.oO[referenceGeometry]Oo.'
process.TrackerGeometryCompare.moduleList = '.oO[moduleListBase]Oo.'
process.TrackerGeometryCompare.outputFile = ".oO[name]Oo..Comparison_common.oO[common]Oo..root"

process.load("CommonTools.UtilAlgos.TFileService_cfi")  
process.TFileService.fileName = cms.string("TkSurfDeform_.oO[name]Oo..Comparison_common.oO[common]Oo..root") 

process.TrackerGeometryCompare.levels = [ .oO[levels]Oo. ]

  ##FIXME!!!!!!!!!
  ##replace TrackerGeometryCompare.writeToDB = .oO[dbOutput]Oo.
  ##removed: dbOutputService

process.p = cms.Path(process.TrackerGeometryCompare)
"""
  

######################################################################
######################################################################
dbOutputTemplate= """
//_________________________ db Output ____________________________
        # setup for writing out to DB
        include "CondCore/DBCommon/CondDBSetup.cfi"
#       include "CondCore/DBCommon/data/CondDBCommon.cfi"

    service = PoolDBOutputService {
        using CondDBSetup
        VPSet toPut = {
            { string record = "TrackerAlignmentRcd"  string tag = ".oO[tag]Oo." },
            { string record = "TrackerAlignmentErrorExtendedRcd"  string tag = ".oO[errortag]Oo." }
        }
                # string connect = "sqlite_file:.oO[workdir]Oo./.oO[name]Oo.Common.oO[common]Oo..db"
                string connect = "sqlite_file:.oO[name]Oo.Common.oO[common]Oo..db"
                # untracked string catalog = "file:alignments.xml"
        untracked string timetype = "runnumber"
    }
"""

######################################################################
######################################################################
visualizationTrackerTemplate= """
#include ".oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/scripts/visualizationTracker.C"
void TkAl3DVisualization_.oO[common]Oo._.oO[name]Oo.(){
            //------------------------------ONLY NEEDED INPUTS-------------------------------//
//------Tree Read In--------
    TString inputFileName = ".oO[outputFile]Oo.";
    //output file name
    string outputFileName = ".oO[common]Oo._.oO[name]Oo..Visualization";
    //title
    string line1 = ".oO[alignmentTitle]Oo.";
    string line2 = "vs. .oO[referenceTitle]Oo.";
    //set subdetectors to see
    int subdetector1 = .oO[3DSubdetector1]Oo.;
    int subdetector2 = .oO[3DSubdetector2]Oo.;
    //translation scale factor
    int sclftr = .oO[3DTranslationalScaleFactor]Oo.;
    //rotation scale factor
    int sclfrt = 1;
    //module size scale factor
    float sclfmodulesizex = 1;
    float sclfmodulesizey = 1;
    float sclfmodulesizez = 1;
    //beam pipe radius
    float piperadius = 2.25;
    //beam pipe xy coordinates
    float pipexcoord = 0;
    float pipeycoord = 0;
    //beam line xy coordinates
    float linexcoord = 0;
    float lineycoord = 0;
//------------------------------End of ONLY NEEDED INPUTS-------------------------------//
    cout << "running visualizer" << endl;
    runVisualizer(inputFileName,
                    outputFileName,
                    line1,
                    line2,
                    subdetector1,
                    subdetector2,
                    sclftr,
                    sclfrt,
                    sclfmodulesizex,
                    sclfmodulesizey,
                    sclfmodulesizez,
                    piperadius,
                    pipexcoord,
                    pipeycoord,
                    linexcoord,
                    lineycoord );
}
"""
