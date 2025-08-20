import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.siStripRegionConnectivity_cfi import siStripRegionConnectivity as _siStripRegionConnectivity
SiStripRegionConnectivity = _siStripRegionConnectivity.clone(
    EtaDivisions = 20,
    PhiDivisions = 20,
    EtaMax = 2.5
)


