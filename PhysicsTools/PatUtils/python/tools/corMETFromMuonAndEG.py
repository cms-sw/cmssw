import FWCore.ParameterSet.Config as cms


def corMETFromMuonAndEG(process,
                        pfCandCollection,
                        electronCollection,
                        photonCollection,
                        corElectronCollection,
                        corPhotonCollection,
                        muCorrection,
                        eGCorrection,
                        allMETEGCorrected,
                        runOnMiniAOD,
                        eGPfCandMatching=False,
                        corMetName="slimmedMETsCorMuAndEG",
                        muSelection="",
                        muonCollection="",
                        eGPFix="",
                        postfix=""
                        ):

    #Muon first =======================================
    if muCorrection:
        from PhysicsTools.PatUtils.tools.muonRecoMitigation import muonRecoMitigation
        from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD, runMetCorAndUncForMiniAODProduction
    
        muonRecoMitigation(process,
                           pfCandCollection=pfCandCollection,
                           runOnMiniAOD=runOnMiniAOD,
                           muonCollection=muonCollection,
                           selection=muSelection,
                           cleaningScheme="duplicated",
                           postfix=postfix)
        
        if runOnMiniAOD:
            runMetCorAndUncFromMiniAOD(process,
                                       pfCandColl="cleanMuonsPFCandidates"+postfix,
                                       recoMetFromPFCs=True,
                                       postfix="MuClean"+postfix
                                       )
        else:
            runMetCorAndUncForMiniAODProduction(process,
                                                pfCandColl="cleanMuonsPFCandidates"+postfix,
                                                recoMetFromPFCs=True,
                                                postfix="MuClean"+postfix
                                                )

    #EGamma simultaneously ====================================
    if eGCorrection:
        from PhysicsTools.PatUtils.tools.eGammaCorrection import eGammaCorrection

        #no better idea for now, duplicating the full std METs
        #if we do not correct for the muons
        pFix="MuClean"+postfix if muCorrection else eGPFix #postfix
        metCollections=["patPFMetT1"+pFix]
        if allMETEGCorrected:
            metCollections.extend([
                    "patPFMetRaw"+pFix,
                    "patPFMetT0pcT1"+pFix,
                    #"patPFMetT0pcT1T2"+pFix,
                    "patPFMetT1Smear"+pFix,
                    "patPFMetT1SmearTxy"+pFix,
                    "patPFMetT0pcT1SmearTxy"+pFix,
                    #"patPFMetT1T2"+pFix,
                    "patPFMetT0pcT1Txy"+pFix,
                    "patPFMetT1Txy"+pFix,
                    "patPFMetTxy"+pFix])
            if not muCorrection:
                variations=["Up","Down"]
                for var in variations:
                    metCollections.extend([
                            "patPFMetT1JetEn"+var+pFix,
                            "patPFMetT1JetRes"+var+pFix,
                            "patPFMetT1SmearJetRes"+var+pFix,
                            "patPFMetT1ElectronEn"+var+pFix,
                            "patPFMetT1PhotonEn"+var+pFix,
                            "patPFMetT1MuonEn"+var+pFix,
                            "patPFMetT1TauEn"+var+pFix,
                            "patPFMetT1UnclusteredEn"+var+pFix,
                            ])
            
        eGPfCandCollection= pfCandCollection if not muCorrection else "cleanMuonsPFCandidates"+postfix
        eGammaCorrection(process, 
                         electronCollection=electronCollection,
                         photonCollection=photonCollection,
                         corElectronCollection=corElectronCollection,
                         corPhotonCollection=corPhotonCollection,
                         metCollections=metCollections,
                         pfCandMatching=eGPfCandMatching,
                         pfCandidateCollection=eGPfCandCollection,
                         corMetName=corMetName,
                         postfix=postfix
                         )



    
