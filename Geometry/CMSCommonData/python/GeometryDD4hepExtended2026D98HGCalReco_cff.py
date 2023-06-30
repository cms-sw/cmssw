import FWCore.ParameterSet.Config as cms

from Configuration.Geometry.GeometryDD4hep_cff import *
DDDetectorESProducer.confGeomXMLFiles = cms.FileInPath("Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2026D98HGCal.xml")
from Geometry.HGCalCommonData.hgcalParametersInitialization_cfi import *
from Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi import *
from Geometry.CaloEventSetup.HGCalTopology_cfi import *
from Geometry.HGCalGeometry.HGCalGeometryESProducer_cfi import *
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.CaloGeometryBuilder_cfi import *
CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring("HGCalEESensitive",
                                "HGCalHESiliconSensitive",
                                "HGCalHEScintillatorSensitive"
    )
)
