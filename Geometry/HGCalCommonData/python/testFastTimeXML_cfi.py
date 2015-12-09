import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
         'Geometry/TrackerCommonData/data/trackermaterial.xml',
         'Geometry/CMSCommonData/data/rotations.xml',
         'Geometry/CMSCommonData/data/extend/cmsextent.xml',
         'Geometry/CMSCommonData/data/cms.xml',
         'Geometry/CMSCommonData/data/cmsMother.xml',
         'Geometry/CMSCommonData/data/eta3/etaMax.xml',
         'Geometry/CMSCommonData/data/PhaseII/caloBase.xml',
         'Geometry/CMSCommonData/data/cmsCalo.xml',
         'Geometry/EcalCommonData/data/PhaseII/eregalgo.xml',
         'Geometry/EcalCommonData/data/PhaseII/escon.xml',
         'Geometry/EcalCommonData/data/PhaseII/esalgo.xml',
         'Geometry/EcalCommonData/data/ectkcable.xml',
         'Geometry/EcalCommonData/data/ebcon.xml',
         'Geometry/EcalCommonData/data/eecon.xml',
         'Geometry/HGCalCommonData/data/NoTaper/shashlik.xml',
         'Geometry/HGCalCommonData/data/NoTaper/shashliksupermodule.xml',
         'Geometry/HGCalCommonData/data/NoTaper/shashlikmodule.xml',
         'Geometry/HGCalCommonData/data/NoTaper/shashlikConstEta3.xml',
         'Geometry/HGCalCommonData/data/fastTiming.xml',
         'Geometry/HGCalCommonData/data/v2/fastTimingElement.xml',
         'Geometry/HGCalCommonData/data/fastTimingConst.xml',
         'Geometry/HGCalSimData/data/fasttimesens.xml',
         'Geometry/HGCalSimData/data/fasttimeProdCuts.xml',
         'Geometry/HGCalSimData/data/shashliksens.xml',
         'Geometry/HGCalSimData/data/shashlikProdCuts.xml'),
    rootNodeName = cms.string('cms:OCMS')
)

 
