import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits._templates_default_cfi import _templates_default
templates = _templates_default.clone()
templates.DoLorentz = True

# This customization will be removed once we get the templates for phase2 pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(templates,
  LoadTemplatesFromDB = False,
  DoLorentz = False,
)

