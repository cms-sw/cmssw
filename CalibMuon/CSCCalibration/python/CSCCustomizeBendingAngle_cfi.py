def set_6bit_gemcsc_bending_LUTs(process):
  process.CSCL1TPLookupTableEP.esDiffToSlopeME11bFiles = [
    "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/GEMCSC_SlopeAmendment_6bit_NoCOSI_ME1b_even_layer1.txt",
    "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/GEMCSC_SlopeAmendment_6bit_NoCOSI_ME1b_odd_layer1.txt",
    "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/GEMCSC_SlopeAmendment_6bit_NoCOSI_ME1b_even_layer2.txt",
    "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/GEMCSC_SlopeAmendment_6bit_NoCOSI_ME1b_odd_layer2.txt",
  ]
  process.CSCL1TPLookupTableEP.esDiffToSlopeME11aFiles = [
    "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/GEMCSC_SlopeAmendment_6bit_NoCOSI_ME1a_even_layer1.txt",
    "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/GEMCSC_SlopeAmendment_6bit_NoCOSI_ME1a_odd_layer1.txt",
    "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/GEMCSC_SlopeAmendment_6bit_NoCOSI_ME1a_even_layer2.txt",
    "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/GEMCSC_SlopeAmendment_6bit_NoCOSI_ME1a_odd_layer2.txt",
  ]
  return process