import FWCore.ParameterSet.Config as cms

#--- Allow for multiple calls to the database ---#
from CondCore.DBCommon.CondDBSetup_cfi import *
#--- Geometry Setup ---#
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.RPCGeometry.rpcGeometry_cfi import *
import copy
from EventFilter.CSCRawToDigi.cscUnpacker_cfi import *
#--- CSC ---#
muonCSCDigis = copy.deepcopy(muonCSCDigis)
import copy
from EventFilter.DTRawToDigi.dtunpacker_cfi import *
#--- DT ---#
muonDTDigis = copy.deepcopy(muonDTDigis)
import copy
from EventFilter.RPCRawToDigi.rpcUnpacker_cfi import *
#--- RPC ---#
muonRPCDigis = copy.deepcopy(rpcunpacker)
MuonRawToDigi = cms.Sequence(muonCSCDigis+muonDTDigis+muonRPCDigis)
muonCSCDigis.InputObjects = 'rawDataCollector'
muonCSCDigis.UseExaminer = False
muonDTDigis.fedColl = 'rawDataCollector'
muonRPCDigis.InputLabel = 'rawDataCollector'

