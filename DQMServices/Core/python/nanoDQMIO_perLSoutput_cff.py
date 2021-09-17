import FWCore.ParameterSet.Config as cms

#Example version list of MEs to save with singel Luminosity Granularity
#in the nanoDQMIO reduced version of DQMIO data Tier
#It needs process.DQMStore.saveByLumi = cms.untracked.bool(True)
#to make effect in the MEs saved by DQMStore
#DQMIO with per Lumisection data, are a special kind of DQM files 
#containing almost the full set of DQM Monitor Elements (MEs) saved 
#with single lumisection time granularity. 
#Saying "almost" we refer to the fact that only Monitor Elements 
#produced in DQM Step1 are saved, 
#while those produced in the Harvesting step are not, 
#even if they could be obtained with some ad-hoc harvesting on Step1 data
#Hence, DQM Step2 (HARVESTING DQM) should not follow when saveByLumi is True
#since most DQM Harvesting modules expect perRun output
#https://twiki.cern.ch/twiki/bin/view/CMS/PerLsDQMIO

nanoDQMIO_perLSoutput = cms.PSet(
      MEsToSave = cms.untracked.vstring(*( #Using tuple to avoid python limit of 255 arguments
                                           #as suggested in:
                #https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuidePoolInputSources#Example_3_More_than_255_input_fi
              'Muons/MuonRecoAnalyzer/',               #Folder and its subfolders
              'Muons/MuonIdDQM/GlobalMuons/hDT1Pullx'  #particular ME
              )
      )
)
