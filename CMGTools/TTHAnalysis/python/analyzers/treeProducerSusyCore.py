from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer  import * 

susyCore_globalVariables = [
            NTupleVariable("rho",  lambda ev: ev.rho, float, help="kt6PFJets rho"),
            NTupleVariable("nVert",  lambda ev: len(ev.goodVertices), int, help="Number of good vertices"), 

#            NTupleVariable("nJet25", lambda ev: len(ev.cleanJets), int, help="Number of jets with pt > 25"),
#            NTupleVariable("nBJetLoose25", lambda ev: len(ev.bjetsLoose), int, help="Number of jets with pt > 25 passing CSV loose"),
#            NTupleVariable("nBJetMedium25", lambda ev: len(ev.bjetsMedium), int, help="Number of jets with pt > 25 passing CSV medium"),
#            NTupleVariable("nBJetTight25", lambda ev: sum([j.btagWP("CSVv2IVFT") for j in ev.bjetsMedium]), int, help="Number of jets with pt > 25 passing CSV tight"),

            NTupleVariable("nJet25", lambda ev: sum([j.pt() > 25 for j in ev.cleanJets]), int, help="Number of jets with pt > 25, |eta|<2.4"),
            NTupleVariable("nJet25a", lambda ev: sum([j.pt() > 25 for j in ev.cleanJetsAll]), int, help="Number of jets with pt > 25, |eta|<4.7"),
            NTupleVariable("nBJetLoose25", lambda ev: sum([j.btagWP("CSVv2IVFL") for j in ev.cleanJets if j.pt() > 25]), int, help="Number of jets with pt > 25 passing CSV loose"),
            NTupleVariable("nBJetMedium25", lambda ev: sum([j.btagWP("CSVv2IVFM") for j in ev.bjetsMedium if j.pt() > 25]), int, help="Number of jets with pt > 25 passing CSV medium"),
            NTupleVariable("nBJetTight25", lambda ev: sum([j.btagWP("CSVv2IVFT") for j in ev.bjetsMedium if j.pt() > 25]), int, help="Number of jets with pt > 25 passing CSV tight"),

            NTupleVariable("nJet30", lambda ev: sum([j.pt() > 30 for j in ev.cleanJets]), int, help="Number of jets with pt > 30, |eta|<2.4"),
            NTupleVariable("nJet30a", lambda ev: sum([j.pt() > 30 for j in ev.cleanJetsAll]), int, help="Number of jets with pt > 30, |eta|<4.7"),
            NTupleVariable("nBJetLoose30", lambda ev: sum([j.btagWP("CSVv2IVFL") for j in ev.cleanJets if j.pt() > 30]), int, help="Number of jets with pt > 30 passing CSV loose"),
            NTupleVariable("nBJetMedium30", lambda ev: sum([j.btagWP("CSVv2IVFM") for j in ev.bjetsMedium if j.pt() > 30]), int, help="Number of jets with pt > 30 passing CSV medium"),
            NTupleVariable("nBJetTight30", lambda ev: sum([j.btagWP("CSVv2IVFT") for j in ev.bjetsMedium if j.pt() > 30]), int, help="Number of jets with pt > 30 passing CSV tight"),

            NTupleVariable("nJet40", lambda ev: sum([j.pt() > 40 for j in ev.cleanJets]), int, help="Number of jets with pt > 40, |eta|<2.4"),
            NTupleVariable("nJet40a", lambda ev: sum([j.pt() > 40 for j in ev.cleanJetsAll]), int, help="Number of jets with pt > 40, |eta|<4.7"),
            NTupleVariable("nBJetLoose40", lambda ev: sum([j.btagWP("CSVv2IVFL") for j in ev.cleanJets if j.pt() > 40]), int, help="Number of jets with pt > 40 passing CSV loose"),
            NTupleVariable("nBJetMedium40", lambda ev: sum([j.btagWP("CSVv2IVFM") for j in ev.bjetsMedium if j.pt() > 40]), int, help="Number of jets with pt > 40 passing CSV medium"),
            NTupleVariable("nBJetTight40", lambda ev: sum([j.btagWP("CSVv2IVFT") for j in ev.bjetsMedium if j.pt() > 40]), int, help="Number of jets with pt > 40 passing CSV tight"),

            ##--------------------------------------------------
            NTupleVariable("nLepGood20", lambda ev: sum([l.pt() > 20 for l in ev.selectedLeptons]), int, help="Number of leptons with pt > 20"),
            NTupleVariable("nLepGood15", lambda ev: sum([l.pt() > 15 for l in ev.selectedLeptons]), int, help="Number of leptons with pt > 15"),
            NTupleVariable("nLepGood10", lambda ev: sum([l.pt() > 10 for l in ev.selectedLeptons]), int, help="Number of leptons with pt > 10"),
            ##--------------------------------------------------
            #NTupleVariable("GenHeaviestQCDFlavour", lambda ev : ev.heaviestQCDFlavour, int, mcOnly=True, help="pdgId of heaviest parton in the event (after shower)"),
            #NTupleVariable("LepEff_1lep", lambda ev : ev.LepEff_1lep, mcOnly=True, help="Lepton preselection SF (1 lep)"),
            #NTupleVariable("LepEff_2lep", lambda ev : ev.LepEff_2lep, mcOnly=True, help="Lepton preselection SF (2 lep)"),
            ##------------------------------------------------
            NTupleVariable("GenSusyMScan1", lambda ev : ev.genSusyMScan1, int, mcOnly=True, help="Susy mass 1 in scan"),
            NTupleVariable("GenSusyMScan2", lambda ev : ev.genSusyMScan2, int, mcOnly=True, help="Susy mass 2 in scan"),
            NTupleVariable("GenSusyMScan3", lambda ev : ev.genSusyMScan3, int, mcOnly=True, help="Susy mass 3 in scan"),
            NTupleVariable("GenSusyMScan4", lambda ev : ev.genSusyMScan4, int, mcOnly=True, help="Susy mass 4 in scan"),
            NTupleVariable("GenSusyMGluino", lambda ev : ev.genSusyMGluino, int, mcOnly=True, help="Susy Gluino mass"),
            NTupleVariable("GenSusyMGravitino", lambda ev : ev.genSusyMGravitino, int, mcOnly=True, help="Susy Gravitino mass"),
            NTupleVariable("GenSusyMStop", lambda ev : ev.genSusyMStop, int, mcOnly=True, help="Susy Stop mass"),
            NTupleVariable("GenSusyMSbottom", lambda ev : ev.genSusyMSbottom, int, mcOnly=True, help="Susy Sbottom mass"),
            NTupleVariable("GenSusyMStop2", lambda ev : ev.genSusyMStop2, int, mcOnly=True, help="Susy Stop2 mass"),
            NTupleVariable("GenSusyMSbottom2", lambda ev : ev.genSusyMSbottom2, int, mcOnly=True, help="Susy Sbottom2 mass"),
            NTupleVariable("GenSusyMSquark", lambda ev : ev.genSusyMSquark, int, mcOnly=True, help="Susy Squark mass"),
            NTupleVariable("GenSusyMNeutralino", lambda ev : ev.genSusyMNeutralino, int, mcOnly=True, help="Susy Neutralino mass"),
            NTupleVariable("GenSusyMNeutralino2", lambda ev : ev.genSusyMNeutralino2, int, mcOnly=True, help="Susy Neutralino2 mass"),
            NTupleVariable("GenSusyMNeutralino3", lambda ev : ev.genSusyMNeutralino3, int, mcOnly=True, help="Susy Neutralino3 mass"),
            NTupleVariable("GenSusyMNeutralino4", lambda ev : ev.genSusyMNeutralino4, int, mcOnly=True, help="Susy Neutralino4 mass"),
            NTupleVariable("GenSusyMChargino", lambda ev : ev.genSusyMChargino, int, mcOnly=True, help="Susy Chargino mass"),
            NTupleVariable("GenSusyMChargino2", lambda ev : ev.genSusyMChargino2, int, mcOnly=True, help="Susy Chargino2 mass"),
]

susyCore_globalObjects = {
            "met" : NTupleObject("met", metType, help="PF E_{T}^{miss}, after type 1 corrections"),
            #"metNoPU" : NTupleObject("metNoPU", fourVectorType, help="PF noPU E_{T}^{miss}"),
}

susyCore_collections = {
            "genleps"         : NTupleCollection("genLep",     genParticleWithLinksType, 10, help="Generated leptons (e/mu) from W/Z decays"),                                                                                                
            "gentauleps"      : NTupleCollection("genLepFromTau", genParticleWithLinksType, 10, help="Generated leptons (e/mu) from decays of taus from W/Z/h decays"),                                                                       
            "gentaus"         : NTupleCollection("genTau",     genParticleWithLinksType, 10, help="Generated leptons (tau) from W/Z decays"),                            
            "generatorSummary" : NTupleCollection("GenPart", genParticleWithLinksType, 100 , help="Hard scattering particles, with ancestry and links"),
}
