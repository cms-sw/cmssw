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

                #Examples:
                #'Muons/MuonRecoAnalyzer/',               #Folder and its subfolders
                #'Muons/MuonIdDQM/GlobalMuons/hDT1Pullx'  #particular ME
            
              #Version 0.1 for nanoDQMIO in CMSSW_12_1_0 ReReco of Pilot Test Runs taken in Autumn 2021
 #DT
 'DT/02-Segments/03-MeanT0/T0MeanAllWheels',
            
 #ECAL            
 'EcalBarrel/EBOccupancyTask/EBOT digi occupancy',
 'EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE -',
 'EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE +',

 #Muon POG
 'Muons/MuonRecoAnalyzer/',               
 'Muons/MuonIdDQM/GlobalMuons/',
            
 #Tracker/Tracking
 #PixelPhase1
 'PixelPhase1/Phase1_MechanicalView/',
 'PixelPhase1/Tracks/',

 #SiStrip
 'SiStrip/MechanicalView/',

 #Tracking histograms:
 'Tracking/PrimaryVertices/highPurityTracks/pt_0to1/offline/',
 'Tracking/TrackParameters/generalTracks/LSanalysis/',
 'Tracking/TrackParameters/highPurityTracks/pt_1/LSanalysis/',
 'Tracking/TrackParameters/highPurityTracks/pt_0to1/LSanalysis/',
 'Tracking/TrackParameters/highPurityTracks/dzPV0p1/LSanalysis/',
 'Tracking/TrackParameters/generalTracks/GeneralProperties/',
 'Tracking/TrackParameters/highPurityTracks/pt_1/GeneralProperties/',
 'Tracking/TrackParameters/highPurityTracks/pt_0to1/GeneralProperties/',
 'Tracking/TrackParameters/highPurityTracks/dzPV0p1/GeneralProperties/',
 'Tracking/TrackParameters/generalTracks/HitProperties/',
 'Tracking/TrackParameters/highPurityTracks/pt_1/HitProperties/',
 'Tracking/TrackParameters/highPurityTracks/pt_0to1/HitProperties/',
 'Tracking/TrackParameters/highPurityTracks/dzPV0p1/HitProperties/',
 'Tracking/TrackParameters/generalTracks/HitProperties/Pixel/',
 'Tracking/TrackParameters/highPurityTracks/pt_1/HitProperties/Pixel/',
 'Tracking/TrackParameters/highPurityTracks/pt_0to1/HitProperties/Pixel/',
 'Tracking/TrackParameters/highPurityTracks/dzPV0p1/HitProperties/Pixel/',
 'Tracking/TrackParameters/generalTracks/HitProperties/Strip/',
 'Tracking/TrackParameters/highPurityTracks/pt_1/HitProperties/Strip/',
 'Tracking/TrackParameters/highPurityTracks/pt_0to1/HitProperties/Strip/',
 'Tracking/TrackParameters/highPurityTracks/dzPV0p1/HitProperties/Strip/'
              )
      )
)
