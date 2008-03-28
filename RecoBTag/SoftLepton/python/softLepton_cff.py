import FWCore.ParameterSet.Config as cms

from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoBTau.JetTagComputer.jetTagRecord_cfi import *
from RecoBTag.SoftLepton.btagSoftElectrons_cfi import *
from RecoBTag.SoftLepton.softElectronTagInfos_cfi import *
from RecoBTag.SoftLepton.softElectronES_cfi import *
from RecoBTag.SoftLepton.softElectronBJetTags_cfi import *
from RecoBTag.SoftLepton.softMuonTagInfos_cfi import *
from RecoBTag.SoftLepton.softMuonES_cfi import *
from RecoBTag.SoftLepton.softMuonBJetTags_cfi import *
from RecoBTag.SoftLepton.softMuonNoIPES_cfi import *
from RecoBTag.SoftLepton.softMuonNoIPBJetTags_cfi import *

