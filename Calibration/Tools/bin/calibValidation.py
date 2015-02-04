from ROOT import *
gSystem.Load("libFWCoreFWLite.so")
AutoLibraryLoader.enable()
gSystem.Load("libCalibrationTools.so")
ic = IC()

recovery_ring = {}

for l in open("dati_pred.dat"):
        v = l.split()
        recovery_ring[int(float(v[1]))] = float(v[2])

rings = DRings()
rings.setEERings("eerings.dat");

EcalBarrel = 1
EcalEndcap = 2

for id in ic.ids():
        if id.subdetId() == EcalEndcap:
                eeid = EEDetId(id)
                #print eeid.ix(), eeid.iy(), eeid.zside(), rings.ieta(id), recovery_ring[rings.ieta(id)]
                print eeid.ix(), eeid.iy(), eeid.zside(), recovery_ring[rings.ieta(id)]
