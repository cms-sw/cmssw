
import FWCore.ParameterSet.Config as cms


from L1Trigger.L1TMuonBarrel.simKBmtfDigis_cfi import bmtfKalmanTrackingSettings as settings

eLoss = settings.eLoss[0]
alpha = settings.aPhiB[0]
alpha2 = settings.aPhiBNLO[0]



dxy=[]
deltaK=[]


for addr in range(0,2048):
    Knew=addr*2-int(2*addr/(1+eLoss*addr))
    deltaK.append(str(abs(Knew)))

    d = int(alpha*addr/(1+alpha2*addr))
    dxy.append(str(abs(d)))
        




print 'ap_ufixed<12,12> eLossVertex[2048] = {'+','.join(deltaK)+'};'
print 'ap_ufixed<12,12> dxyVertex[2048] = {'+','.join(dxy)+'};'



