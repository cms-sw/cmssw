
destination='sqlite:hcal_hardcoded_conditions.db'
#destination='oracle://devdb10/CMS_COND_HCAL'

hcalCalibrationsCopy pedestals -input defaults -output $destination -outputrun 999999 -outputtag hardcoded_pedestals
hcalCalibrationsCopy pwidths -input defaults -output $destination -outputrun 999999 -outputtag hardcoded_pedestal_widths
hcalCalibrationsCopy gains -input defaults -output $destination -outputrun 999999 -outputtag hardcoded_gains
hcalCalibrationsCopy gwidths -input defaults -output $destination -outputrun 999999 -outputtag hardcoded_gain_widths
hcalCalibrationsCopy qie -input defaults -output $destination -outputrun 999999 -outputtag hardcoded_qie
hcalCalibrationsCopy emap -input defaults -output $destination -outputrun 999999 -outputtag hardcoded_emap
