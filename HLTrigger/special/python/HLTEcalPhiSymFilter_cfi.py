import FWCore.ParameterSet.Config as cms

#
# Configure HLT filter and alcareco stream for  
# Ecal PhiSymmetry
#
# Author: Stefano Argiro
# $Id: HLTEcalPhiSymFilter_cfi.py,v 1.4 2009/10/06 08:27:48 argiro Exp $

# 
# eCut_barrel : threshold to accept rechits
# eCut_barrel_high : threshold to accept rechits if the channel in 'not OK' 
# statusThreshold  : threshold to channel status to mark it as 'not OK'
# useRecoFlag      : use recoFlag() from rechit or ChannelStatus from
#                    to mark bad channels with statusThreshold

alCaPhiSymStream = cms.EDFilter("HLTEcalPhiSymFilter",
    endcapHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    eCut_barrel = cms.double(150.0),
    eCut_endcap = cms.double(750.0),
    eCut_barrel_high = cms.double(999999.00),
    eCut_endcap_high = cms.double(999999.00),
    statusThreshold = cms.uint32(3),
    useRecoFlag = cms.bool(False),                            
    phiSymBarrelHitCollection = cms.string('phiSymEcalRecHitsEB'),
    barrelHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    phiSymEndcapHitCollection = cms.string('phiSymEcalRecHitsEE')                               
)


