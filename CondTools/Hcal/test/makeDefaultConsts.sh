destination='sqlite:hcal_hardcoded_conditions.db'

hcalCalibrationsCopy pedestals -input defaults -output $destination -outputrun 999999 -outputtag hardcoded_pedestals
hcalCalibrationsCopy pwidth -input defaults -output $destination -outputrun 999999 -outputtag hardcoded_pedestal_widths
hcalCalibrationsCopy gains -input defaults -output $destination -outputrun 999999 -outputtag hardcoded_gains
hcalCalibrationsCopy gwidth -input defaults -output $destination -outputrun 999999 -outputtag hardcoded_gain_widths
hcalCalibrationsCopy qie -input defaults -output $destination -outputrun 999999 -outputtag hardcoded_qie
hcalCalibrationsCopy emap -input defaults -output $destination -outputrun 999999 -outputtag hardcoded_emap
