import FWCore.ParameterSet.Config as cms
import DPGAnalysis.Skims.HCALHighEnergyCombinedPath_cff

process = cms.Process("SKIM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load('Configuration/EventContent/EventContentCosmics_cff')

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/FE0A36FF-11ED-DE11-A8EE-002618943849.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/F89D50B7-07ED-DE11-B079-0026189437ED.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/F0E44CBD-07ED-DE11-A0F7-002618943914.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/DAB587B8-07ED-DE11-8369-00304867C1BC.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/C8E03DBC-07ED-DE11-88ED-00248C0BE005.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/C4C791B7-07ED-DE11-B8EF-0026189438B3.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/C4C5EFBA-07ED-DE11-8D22-002354EF3BE0.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/B8F9A6B8-07ED-DE11-BD5E-0026189438BC.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/B4F9CDB9-0BED-DE11-997A-002618943935.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/92A65397-09ED-DE11-B2C9-003048678B38.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/86BB98BE-07ED-DE11-A1EC-002618943940.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/8464EBB6-0BED-DE11-A1C6-002618943900.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/82517594-09ED-DE11-B00D-00261894391C.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/76BEEDB8-0BED-DE11-B401-002618943922.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/647E26BE-0BED-DE11-B131-00261894387D.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/621529B7-07ED-DE11-9E00-002618943954.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/5C6F4B6B-05ED-DE11-8D3F-002618943922.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/5C1EE4B7-07ED-DE11-9B8F-00304867BF18.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/5A1324BD-0BED-DE11-BD8A-003048678AC8.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/4EDAEB96-09ED-DE11-9194-0026189438F4.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/4ECC5DB8-0BED-DE11-B96D-00261894387E.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/404A2E98-09ED-DE11-B7CF-002618943821.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/3C641992-09ED-DE11-A1E8-00248C0BE005.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/2854A618-10ED-DE11-8FA0-003048D15E24.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/22786B91-09ED-DE11-8316-00304867BED8.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/1E18E995-09ED-DE11-9684-002618943870.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/1C938F94-09ED-DE11-B09B-003048679080.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/1C8C25B7-07ED-DE11-BF59-002618943940.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/1886CDB9-07ED-DE11-B316-002354EF3BDA.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/0A898597-09ED-DE11-91AD-00261894386E.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/0419D454-05ED-DE11-A526-00248C55CC9D.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0001/C26AC2FD-FEEC-DE11-A9DD-002618943849.root'
    ),
                            secondaryFileNames = cms.untracked.vstring(
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/FEF5C2C0-1AE7-DE11-B906-000423D99896.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/EC9EE32B-34E7-DE11-970D-0030486780B8.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/EC37B3A3-30E7-DE11-AB2D-001617C3B76A.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/E872544D-36E7-DE11-BE0A-003048D375AA.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/E2FB2D46-36E7-DE11-A97E-003048D37580.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/BE8CC520-39E7-DE11-994F-000423D98634.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/B00793A8-30E7-DE11-B9BE-000423D6C8E6.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/A69CB0C0-1AE7-DE11-8DC0-0030486780B8.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/9AA3357D-2EE7-DE11-942D-001D09F254CE.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/907F8FBB-32E7-DE11-8209-001D09F2932B.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/8ABE4753-31E7-DE11-B191-000423D6C8E6.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/8A549B69-22E7-DE11-89B8-000423D99F1E.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/80E25D4D-1BE7-DE11-BE3F-001D09F2910A.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/7261AF77-33E7-DE11-BE70-000423D944FC.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/7095DA3C-2FE7-DE11-9CA3-000423D99B3E.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/5E1B037E-2EE7-DE11-81B8-001D09F28F1B.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/5C753EFB-36E7-DE11-B039-0019DB29C614.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/58DADC98-35E7-DE11-B66A-003048D2C020.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/56F009B0-37E7-DE11-ADBC-001617C3B6FE.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/5075C87B-29E7-DE11-A023-003048D2C1C4.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/4E72E311-32E7-DE11-B671-001617E30D40.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/4C94EA59-3BE7-DE11-AA88-001D09F25208.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/4ACCF1BB-32E7-DE11-A859-001D09F25208.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/488CAD78-38E7-DE11-BDFE-001617E30D40.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/4844FED2-39E7-DE11-A379-000423D99A8E.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/44CD25B0-37E7-DE11-B5E6-001617E30D52.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/4417F7CB-2DE7-DE11-AF3C-001D09F252DA.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/40AE5ADD-34E7-DE11-A3FD-003048678098.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/405E4654-3BE7-DE11-9022-000423D99614.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/3AA88AF5-2FE7-DE11-A46D-001D09F231C9.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/3A249889-3AE7-DE11-9131-001617C3B5E4.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/34E6C1F4-2FE7-DE11-896A-001D09F2438A.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/28A20CDE-32E7-DE11-9FBF-0019B9F72F97.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/16C870B1-37E7-DE11-9778-003048D2BE08.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/023/14987277-33E7-DE11-B1E7-000423D98EC8.root'
)
)

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/hcal_ecal_HSCP_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT HCALHighEnergy ecalhighenergy and stoppedhSCP skim')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )


#################################HCAL High Energy#########################################

## process.extend(DPGAnalysis.Skims.HCALHighEnergyCombinedPath_cff)

## process.outHCAL = cms.OutputModule("PoolOutputModule",
##                                outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
##                                SelectEvents = cms.untracked.PSet(
##     SelectEvents = cms.vstring("HCALHighEnergyPath")
##     ),
##                                dataset = cms.untracked.PSet(
##                                dataTier = cms.untracked.string('RAW-RECO'),
##                                filterName = cms.untracked.string('HCALHighEnergy')),
##                                fileName = cms.untracked.string('/tmp/jbrooke/HCALHighEnergy_filter.root')
##                                )

#################################ECAL High Energy#########################################
## process.skimming = cms.EDFilter("EcalSkim",
##     #cosmic cluster energy threshold in GeV
##     energyCutEB = cms.untracked.double(2.0),
##     energyCutEE = cms.untracked.double(2.0),
##     endcapClusterCollection = cms.InputTag("cosmicSuperClusters","CosmicEndcapSuperClusters"),
##     barrelClusterCollection = cms.InputTag("cosmicSuperClusters","CosmicBarrelSuperClusters")
## )
## process.p = cms.Path(process.skimming)

## process.outECAL = cms.OutputModule("PoolOutputModule",
##     process.RECOEventContent,                               
##     fileName = cms.untracked.string('/tmp/jbrooke/ecalSkim.root'),
##     dataset = cms.untracked.PSet(
##     	      dataTier = cms.untracked.string('RECO'),
##     	      filterName = cms.untracked.string('ecalSkim_fromRECO')),
##     SelectEvents = cms.untracked.PSet(
##         SelectEvents = cms.vstring('p')
##     )
## )

## process.outECAL.outputCommands.append('drop *_MEtoEDMConverter_*_*')

###############################StoppedHSCP################################
process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
process.hltHighLevel.HLTPaths = cms.vstring("HLT_StoppedHSCP*")

process.skim = cms.Path(process.hltHighLevel)

process.outHSCP = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
                               SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring("skim")
    ),
                               dataset = cms.untracked.PSet(
                               dataTier = cms.untracked.string('RAW-RECO'),
                               filterName = cms.untracked.string('StoppedHSCP')),
                               fileName = cms.untracked.string('/tmp/jbrooke/StoppedHSCP_filter.root')
                               )


process.this_is_the_end = cms.EndPath(process.outHSCP)
