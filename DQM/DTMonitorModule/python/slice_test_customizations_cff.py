from __future__ import print_function
import FWCore.ParameterSet.Config as cms                                                

def customise_for_slice_test(process): 

    print("[customise_for_slice_test]: cloning unpacker and DTDigiTask + customising AB7 sequence")

    # This is commented out as the AB7 unpacker is not in CMSSW
    # at present, the following lines need to be uncommented in the P5 setup

    # from EventFilter.DTRawToDigi.dtab7unpacker_cfi import dtAB7unpacker
    # process.dtAB7Unpacker = dtAB7unpacker.clone()

    # Here using the uROS unpacker as proxy, the following lines
    # need to be commented out in the setup running @ P5

    from EventFilter.DTRawToDigi.dturosunpacker_cfi import dturosunpacker
    process.dtAB7Unpacker = dturosunpacker.clone()
    
    from DQM.DTMonitorModule.dtDigiTask_cfi import dtDigiMonitor
    process.dtAB7DigiMonitor = dtDigiMonitor.clone()

    process.dtAB7DigiMonitor.dtDigiLabel = cms.InputTag("dtAB7Unpacker")
    process.dtAB7DigiMonitor.sliceTestMode = True

    process.dtAB7DigiMonitor.performPerWireT0Calibration = False

    if hasattr(process,"dtDQMTask") :
        print("[customise_for_slice_test]: extending dtDQMTask sequence to include AB7 monitoring")
        process.dtDQMTask.replace(process.dtDigiMonitor, process.dtDigiMonitor +\
                                                         process.dtAB7Unpacker +\
                                                         process.dtAB7DigiMonitor) 

    return process
