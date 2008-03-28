import FWCore.ParameterSet.Config as cms

#
# include geometry service
#
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
#
# include ECAl and HCAL geometry services
#
from Geometry.CaloEventSetup.CaloGeometry_cff import *
#
# calo tower constituents map builder
#
#es_module = CaloTowerConstituentsMapBuilder {
#  untracked string MapFile="Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz"
#}
#
# calo topology builder
#
#include "Geometry/CaloEventSetup/data/CaloTopology.cfi"
#es_module = HcalTopologyIdealEP {}
from FastSimulation.L1CaloTriggerProducer.fastl1calosim_cfi import *

