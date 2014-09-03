def ecalcaliblayout(i, p, *rows): i["EcalCalibration/Layouts/" + p] = DQMItem(layout=rows)
def ecalcaliblclayout(i, p, *rows): i["EcalCalibration/Layouts/00 Light Checker/" + p] = DQMItem(layout=rows)

# Quick Collections
ecalcaliblayout(dqmitems, "00 Laser Sequence Validation",
                [{ 'path': "EcalCalibration/Laser/EcalLaser sequence validation", 'description': "EcalLaser: time, FED number, and status of the laser sequence. Legend: green = good; yellow = warning; red = bad" }])

# Light Checker Layout
ecalcaliblclayout(dqmitems, "00 Laser Sequence Validation",
                [{ 'path': "EcalCalibration/Laser/EcalLaser sequence validation", 'description': "EcalLaser: time, FED number, and status of the laser sequence. Legend: green = good; yellow = warning; red = bad" }])

ecalcaliblclayout(dqmitems, "01 Laser Amplitude Trend",
                [{ 'path': "EcalCalibration/Laser/EcalLaser L1 (blue) amplitude trend", 'description': "Amplitude of the blue laser measured at the source" }],
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L4 (red) amplitude trend", 'description': "Amplitude of the IR laser measured at the source" }])

ecalcaliblclayout(dqmitems, "02 Laser Amplitude RMS Trend",
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L1 (blue) amplitude RMS trend", 'description': "RMS of the amplitude of the blue laser measured at the source"}],
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L4 (red) amplitude RMS trend", 'description': "RMS of the amplitude of the IR laser measured at the source"}])

ecalcaliblclayout(dqmitems, "03 Laser Amplitude Jitter Trend",
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L1 (blue) jitter trend", 'description': "Jitter of the blue laser measured at the source"}],
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L4 (red) jitter trend", 'description': "Jitter of the IR laser measured at the source"}])

ecalcaliblclayout(dqmitems, "04 Laser FWHM Trend",
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L1 (blue) FWHM trend", 'description': "FWHM of the blue laser pulse measured at the source"}],
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L4 (red) FWHM trend", 'description': "FWHM of the IR laser pulse measured at the source"}])

ecalcaliblclayout(dqmitems, "05 Laser Timing Trend",
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L1 (blue) timing trend", 'description': "Timing of the blue laser measured at the source"}],
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L4 (red) timing trend", 'description': "Timing of the IR laser measured at the source"}])

ecalcaliblclayout(dqmitems, "06 Laser Pre-pulse Amplitude Trend",
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L1 (blue) prepulse amplitude trend", 'description': "Amplitude of the pre-pulse of the blue laser measured at the source"}],
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L4 (red) prepulse amplitude trend", 'description': "Amplitude of the pre-pulse of the IR laser measured at the source"}])

ecalcaliblclayout(dqmitems, "07 Laser Pre-pulse Width Trend",
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L1 (blue) prepulse width trend", 'description': "Width of the pre-pulse of the blue laser measured at the source"}],
                  [{ 'path': "EcalCalibration/Laser/EcalLaser L4 (red) prepulse width trend", 'description': "Width of the pre-pulse of the IR laser measured at the source"}])

ecalcaliblclayout(dqmitems, "08 Laser GPIB Action Duration",
                  [{ 'path': "EcalCalibration/Laser/EcalLaser region move duration", 'description': "" }],
                  [{ 'path': "EcalCalibration/Laser/EcalLaser attenuator change duration", 'description': "" }],
                  [{ 'path': "EcalCalibration/Laser/EcalLaser color change duration", 'description': "" }])

ecalcaliblclayout(dqmitems, "09 Laser Amplitude Map Barrel",
                  [{ 'path': "EcalBarrel/EBLaserTask/Laser1/EBLT amplitude map L1", 'description': "Amplitude of the blue laser measured at the detector"}],
                  [{ 'path': "EcalBarrel/EBLaserTask/Laser4/EBLT amplitude map L4", 'description': "Amplitude of the IR laser measured at the detector"}])

ecalcaliblclayout(dqmitems, "10 Laser Amplitude Map Endcap",
                  [{ 'path': "EcalEndcap/EELaserTask/Laser1/EELT amplitude map L1 EE -", 'description': "Amplitude of the blue laser measured at the detector EE -"},
                   { 'path': "EcalEndcap/EELaserTask/Laser1/EELT amplitude map L1 EE +", 'description': "Amplitude of the blue laser measured at the detector EE +" }],
                  [{ 'path': "EcalEndcap/EELaserTask/Laser4/EELT amplitude map L4 EE -", 'description': "Amplitude of the IR laser measured at the detector EE -"},
                   { 'path': "EcalEndcap/EELaserTask/Laser4/EELT amplitude map L4 EE +", 'description': "Amplitude of the IR laser measured at the detector EE +" }])
