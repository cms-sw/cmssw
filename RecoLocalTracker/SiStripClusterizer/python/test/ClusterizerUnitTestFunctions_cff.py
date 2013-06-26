import FWCore.ParameterSet.Config as cms    

def ClusterizerTest(label, params, detunitlist) :
    return  cms.PSet(
        Label = cms.string(label),
        ClusterizerParameters = params,
        Tests = cms.VPSet() + detunitlist
        )

def DetUnit(label, digis, clusters, invalid=False) :
    return cms.PSet(
        Label = cms.string(label),
        Digis = cms.VPSet() + digis,
        Clusters = cms.VPSet() + clusters,
        InvalidCharge = cms.bool(invalid))
    
def digi(strip, adc, noise, gain, quality) :
    return cms.PSet( Strip = cms.uint32(strip),
                     ADC   = cms.uint32(adc),
                     Noise = cms.double(noise),
                     Gain  = cms.double(gain),
                     Quality = cms.bool(quality) )

def cluster(firststrip, amplitudes) :
    return cms.PSet( FirstStrip = cms.uint32(firststrip),
                     Amplitudes = cms.vuint32() + amplitudes )

noise1 = 1;
gain1  = 1;
good  = True;
bad   = False;
Invalid = True;
