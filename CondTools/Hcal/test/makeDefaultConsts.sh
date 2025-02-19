
destination='sqlite:hcal_hardcoded_conditions.db'
#destination='oracle://devdb10/CMS_COND_HCAL'

hcalCalibrationsCopy pedestals -input defaults -output oracle://devdb10/CMS_COND_HCAL -outputrun 999999 -outputtag hardcoded_pedestals
hcalCalibrationsCopy pwidths -input defaults -output oracle://devdb10/CMS_COND_HCAL -outputrun 999999 -outputtag hardcoded_pedestal_widths
hcalCalibrationsCopy gains -input defaults -output oracle://devdb10/CMS_COND_HCAL -outputrun 999999 -outputtag hardcoded_gains
hcalCalibrationsCopy gwidths -input defaults -output oracle://devdb10/CMS_COND_HCAL -outputrun 999999 -outputtag hardcoded_gain_widths
hcalCalibrationsCopy qie -input defaults -output oracle://devdb10/CMS_COND_HCAL -outputrun 999999 -outputtag hardcoded_qie
hcalCalibrationsCopy emap -input defaults -output oracle://devdb10/CMS_COND_HCAL -outputrun 999999 -outputtag hardcoded_emap

hcalCalibrationsCopy pedestals -output oracle://devdb10/CMS_COND_HCAL -outputtag MICHAL_02_PEDESTALS -outputrun 999999 -inputrun 1 -input CMS_HCL_PRTTYPE_HCAL_READER/HCAL_Reader_88@cms_hcl -inputtag MICHAL_02 -verbose
hcalCalibrationsCopy pwidths -output oracle://devdb10/CMS_COND_HCAL -outputtag MICHAL_02_PEDESTAL_WIDTHS -outputrun 999999 -inputrun 1 -input CMS_HCL_PRTTYPE_HCAL_READER/HCAL_Reader_88@cms_hcl -inputtag MICHAL_02 -verbose
