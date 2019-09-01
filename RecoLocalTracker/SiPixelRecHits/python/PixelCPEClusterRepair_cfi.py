import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits._templates2_default_cfi import _templates2_default
templates2 = _templates2_default.clone()

# This customization will be removed once we get the templates for phase2 pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(templates2,
  LoadTemplatesFromDB = False,
  DoLorentz = False,
)

