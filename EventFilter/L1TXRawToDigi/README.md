L1TXRawToDigi contains CMSSW EDProducers for L1T modules not using the 
common unpacker framework in L1TRawToDigi.

L1TCaloLayer1RawToDigi unpacks Layer-1 calorimeter trigger data obtained from its three FEDs.

The producer makes ECAL and HCAL TPG objects.

These should be identical to the ones read out from ECAL TCCs
and HCAL uHTRs.  However, any link issues could result in 
discrepancies.  



L1TTriwMuxRawToDigi unpacks the TwinMux in the Muon system.
