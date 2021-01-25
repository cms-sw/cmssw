from .adapt_to_new_backend import *
dqmitems={}

def sistripAPVgainLayout(i, p, *rows): i["AlCaReco/GainValidation/" + p] = rows

sistripAPVgainLayout(dqmitems, "00 - Fit Error Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/MPVError",
    'description': "Error distribution of Landau fit made on every APV ",
    'draw': { 'withref': "no" }
  },
  { 'path': "AlCaReco/SiStripGainsHarvesting/MPVErrorVsN",
    'description': "Error distribution of Landau fit as a function of the number of entries ",
    'draw': { 'withref': "no" }
  }],
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/MPVErrorVsEta",
    'description': "Error distribution of Landau fit as a function of the module eta position ",
    'draw': { 'withref': "no" }
  },
  { 'path': "AlCaReco/SiStripGainsHarvesting/MPVErrorVsPhi",
    'description': "Error distribution of Landau fit as a function of the module phi position ",
    'draw': { 'withref': "no" }
  }])

sistripAPVgainLayout(dqmitems, "01 - TIB Gain Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/DiffWRTPrevGainTIB",
    'description': "Ratio among the new gain factor and the old gain factor ",
    'draw': { 'withref': "no" }
  },
  { 'path': "AlCaReco/SiStripGainsHarvesting/GainVsPrevGainTIB",
    'description': "Correlation among the new gain factor and the old gain factor ",
    'draw': { 'withref': "no" }
  }],
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/MPVvsEtaTIB",
    'description': "MPV of the Landau fit as a function of the eta module position ",
    'draw': { 'withref': "no" }
  },
  { 'path': "AlCaReco/SiStripGainsHarvesting/MPVvsPhiTIB",
    'description': "MPV of the Landau fit as a function of the phi module position ",
    'draw': { 'withref': "no" }
  }])

sistripAPVgainLayout(dqmitems, "02 - TOB Gain Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/DiffWRTPrevGainTOB",
    'description': "Ratio among the new gain factor and the old gain factor ",
    'draw': { 'withref': "no" }
  },
  { 'path': "AlCaReco/SiStripGainsHarvesting/GainVsPrevGainTOB",
    'description': "Correlation among the new gain factor and the old gain factor ",
    'draw': { 'withref': "no" }
  }],
  [{ 'path': "AlCaReco/SiStripGainsHarvesting/MPVvsEtaTOB",
    'description': "MPV of the Landau fit as a function of the eta module position ",
    'draw': { 'withref': "no" }
  },
  { 'path': "AlCaReco/SiStripGainsHarvesting/MPVvsPhiTOB",
    'description': "MPV of the Landau fit as a function of the phi module position ",
    'draw': { 'withref': "no" }
  }])

sistripAPVgainLayout(dqmitems, "03 - TID Gain Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/DiffWRTPrevGainTID",
    'description': "Ratio among the new gain factor and the old gain factor ",
    'draw': { 'withref': "no" }
  },
  { 'path': "AlCaReco/SiStripGainsHarvesting/GainVsPrevGainTID",
    'description': "Correlation among the new gain factor and the old gain factor ",
    'draw': { 'withref': "no" }
  }],
  [{ 'path': "AlCaReco/SiStripGainsHarvesting/MPVvsEtaTID",
    'description': "MPV of the Landau fit as a function of the eta module position ",
    'draw': { 'withref': "no" }
  },
  { 'path': "AlCaReco/SiStripGainsHarvesting/MPVvsPhiTID",
    'description': "MPV of the Landau fit as a function of the phi module position ",
    'draw': { 'withref': "no" }
  }])

sistripAPVgainLayout(dqmitems, "04 - TEC Gain Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/DiffWRTPrevGainTEC",
    'description': "Ratio among the new gain factor and the old gain factor ",
    'draw': { 'withref': "no" }
  },
  { 'path': "AlCaReco/SiStripGainsHarvesting/GainVsPrevGainTEC",
    'description': "Correlation among the new gain factor and the old gain factor ",
    'draw': { 'withref': "no" }
  }],
  [{ 'path': "AlCaReco/SiStripGainsHarvesting/MPVvsEtaTEC",
    'description': "MPV of the Landau fit as a function of the eta module position ",
    'draw': { 'withref': "no" }
  },
  { 'path': "AlCaReco/SiStripGainsHarvesting/MPVvsPhiTEC",
    'description': "MPV of the Landau fit as a function of the phi module position ",
    'draw': { 'withref': "no" }
  }])

sistripAPVgainLayout(dqmitems, "05 - Missing APV Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/NoMPVfit",
    'description': "Position of non-calibrated APV ",
    'draw': { 'withref': "no" }
  },
  { 'path': "AlCaReco/SiStripGainsHarvesting/NoMPVmasked",
    'description': "Position of masked APV ",
    'draw': { 'withref': "no" },
  }])

sistripAPVgainLayout(dqmitems, "06 - TIB performance Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/TIB__newG2",
    'description': "MPV after calibration ",
    'draw': { 'withref': "no" },
    'overlays': ["AlCaReco/SiStripGainsAAG/TIB_AagBunch"]
  }])

sistripAPVgainLayout(dqmitems, "07 - TIB layer 1 performance Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/TIB_layer_1__newG2",
    'description': "MPV after calibration ",
    'draw': { 'withref': "no" },
    'overlays': ["AlCaReco/SiStripGainsAAG/TIB_layer_1_AagBunch"]
  }])

sistripAPVgainLayout(dqmitems, "08 - TOB performance Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/TOB__newG2",
    'description': "MPV after calibration ",
    'draw': { 'withref': "no" },
    'overlays': ["AlCaReco/SiStripGainsAAG/TOB_AagBunch"]
  }])

sistripAPVgainLayout(dqmitems, "09 - TOB layer 1 performance Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/TOB_layer_1__newG2",
    'description': "MPV after calibration ",
    'draw': { 'withref': "no" },
    'overlays': ["AlCaReco/SiStripGainsAAG/TOB_layer_1_AagBunch"]
  }])

sistripAPVgainLayout(dqmitems, "10 - TID minus performance Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/TIDminus__newG2",
    'description': "MPV after calibration ",
    'draw': { 'withref': "no" },
    'overlays': ["AlCaReco/SiStripGainsAAG/TIDminus_AagBunch"]
  }])

sistripAPVgainLayout(dqmitems, "11 - TID plus performance Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/TIDplus__newG2",
    'description': "MPV after calibration ",
    'draw': { 'withref': "no" },
    'overlays': ["AlCaReco/SiStripGainsAAG/TIDplus_AagBunch"]
  }])

sistripAPVgainLayout(dqmitems, "12 - TEC minus performance Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/TECminus__newG2",
    'description': "MPV after calibration ",
    'draw': { 'withref': "no" },
    'overlays': ["AlCaReco/SiStripGainsAAG/TECminus_AagBunch"]
  }])

sistripAPVgainLayout(dqmitems, "13 - TEC plus performance Summary",
 [{ 'path': "AlCaReco/SiStripGainsHarvesting/TECplus__newG2",
    'description': "MPV after calibration ",
    'draw': { 'withref': "no" },
    'overlays': ["AlCaReco/SiStripGainsAAG/TECplus_AagBunch"]
  }])


apply_dqm_items_to_new_back_end(dqmitems, __file__)
