L1TCaloLayer1RawToDigi contains CMSSW EDProducer which decodes
Layer-1 calorimeter trigger data obtained from its three FEDs.

The producer makes ECAL and HCAL TPG objects.

These should be identical to the ones read out from ECAL TCCs
and HCAL uHTRs.  However, any link issues could result in 
discrepancies.  
