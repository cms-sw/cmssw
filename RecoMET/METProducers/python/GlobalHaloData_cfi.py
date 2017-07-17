import FWCore.ParameterSet.Config as cms
# File: GlobalHaloData_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module to build GlobalHaloData Object and put into the event
# Date: Oct. 15, 2009

GlobalHaloData = cms.EDProducer("GlobalHaloDataProducer",                         
                                # Higher Level Reco
                                metLabel = cms.InputTag("caloMet"),
                                calotowerLabel = cms.InputTag("towerMaker"),
                                CSCSegmentLabel = cms.InputTag("cscSegments"),
                                CSCRecHitLabel = cms.InputTag("csc2DRecHits"),
                                MuonLabel = cms.InputTag("muons"),
                                EcalMinMatchingRadiusParam = cms.double(110.),
                                EcalMaxMatchingRadiusParam  = cms.double(330.),
                                
                                HcalMinMatchingRadiusParam = cms.double(110.),
                                HcalMaxMatchingRadiusParam = cms.double(490.),

                                CSCHaloDataLabel = cms.InputTag("CSCHaloData"),
                                EcalHaloDataLabel = cms.InputTag("EcalHaloData"),
                                HcalHaloDataLabel = cms.InputTag("HcalHaloData"),

                                CaloTowerEtThresholdParam = cms.double(0.3),

                                #Parameters for CSC-calo matching
                                MaxSegmentTheta = cms.double(0.7),

                                rh_et_threshforcscmatching_eb = cms.double(10.),
                                rcalominrsegm_lowthresh_eb = cms.double(-30.),
                                rcalominrsegm_highthresh_eb = cms.double(15.),
                                dtcalosegm_thresh_eb = cms.double(15.),
                                dphicalosegm_thresh_eb = cms.double(0.04),
                                
                                rh_et_threshforcscmatching_ee = cms.double(10.),
                                rcalominrsegm_lowthresh_ee = cms.double(-30.),
                                rcalominrsegm_highthresh_ee = cms.double(30.),
                                dtcalosegm_thresh_ee = cms.double(15.),
                                dphicalosegm_thresh_ee = cms.double(0.04),
                                
                                rh_et_threshforcscmatching_hb = cms.double(20.),
                                rcalominrsegm_lowthresh_hb = cms.double(-100.),
                                rcalominrsegm_highthresh_hb = cms.double(20.),
                                dtcalosegm_thresh_hb = cms.double(15.),
                                dphicalosegm_thresh_hb = cms.double(0.15),
                                
                                rh_et_threshforcscmatching_he = cms.double(20.),
                                rcalominrsegm_lowthresh_he = cms.double(-30.),
                                rcalominrsegm_highthresh_he = cms.double(30.),
                                dtcalosegm_thresh_he = cms.double(15.),
                                dphicalosegm_thresh_he = cms.double(0.1),
                                IsHLT = cms.bool(False)
                                
                                                                
                                )


