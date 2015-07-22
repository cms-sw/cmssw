import math
import ROOT
from ROOT import *
from DataFormats.FWLite import Events, Handle
from PhysicsTools.PythonAnalysis import *
#import print_options

#print_options.set_float_precision(4)
gSystem.Load("libFWCoreFWLite.so")
FWLiteEnabler::enable()

#import EcalDetId
#from DataFormats.EcalDetId import *
#
import sys,os

allRecHits=False
lumi=393
lumi=-1
lumi=466
eventNumber=-1
eventNumber=685233340
#lumi=376
#eventNumber=344404776
#eventNumber=552337102
#eventNumber=577412878

#eventNumber=347006644
#eventNumber=344911692

lumi=374
eventNumber=548668449
# kRecovered lumi=376 eventNumber=552337102
# kRecovered lumi=393 eventNumber=576932539
# kRecovered lumi=394 eventNumber=578483322
lumi=394
eventNumber=578490502
lumi=394
eventNumber=579700192
lumi=395
eventNumber=579843406
lumi=401
eventNumber=588810213
lumi=402
eventNumber=591275401
# kRecovered lumi=403 eventNumber=593293410
lumi=403
eventNumber=591888388
# kRecovered lumi=406 eventNumber=597401546
# kRecovered lumi=407 eventNumber=598290564
lumi=415
eventNumber=610541757
lumi=415
eventNumber=610541436
lumi=416
eventNumber=612542602
# kRecovered lumi=419 eventNumber=616682572
# kRecovered lumi=419 eventNumber=615876590
lumi=422
eventNumber=620689835

lumi=466
eventNumber=685900276
lumi=467
eventNumber=687572911
lumi=472
eventNumber=694966852
# kRecovered
lumi=466
eventNumber=685233340

eventMin=-1

# for now look for events in two files with a given lumi section
maxEvents=-1
event_counter=0

for arg in sys.argv:
    if (arg=='testAlca1'):
        print "testAlca1"
        file="/tmp/"+os.environ["USER"]+"/testAlca1.root"
        file_format = "AlcaFromAOD"
        break
    elif(arg=='testAlca2'):
        print 'testAlca2'
        file="/tmp/"+os.environ["USER"]+"/testAlca2.root"
        file_format = "AlcaFromAOD_Recalib"
        break
    elif(arg=='testAlca3'):
        print 'testAlca3'
        file="/tmp/"+os.environ["USER"]+"/testAlca3.root"
        file_format = "AlcaFromAOD_Recalib"
        break
    elif(arg=='testAlca4'):
        print 'testAlca4'
        file="/tmp/"+os.environ["USER"]+"/testAlca4.root"
        file_format = "AlcaFromAOD_Recalib"
        break
    elif(arg=='AOD'):
        print "AOD"
        file="/tmp/"+os.environ["USER"]+"/rereco30Nov-AOD.root"
        file_format="AOD"
        break
    elif(arg=='AlcaFromAOD'):
        print "AlcaFromAOD"
        file="/tmp/"+os.environ["USER"]+"/AlcarecoFromAOD.root"
        file_format="AlcaFromAOD"
        break
    elif(arg=='AlcaFromAOD-recalib'):
        print "AlcaFromAOD-recalib"
        file="/tmp/"+os.environ["USER"]+"/AlcarecoFromAOD-recalib.root"
        file_format="AlcaFromAOD_Recalib"
        break
    elif(arg=='sandbox'):
        print 'sandbox'
        #        file="/tmp/"+os.environ["USER"]+"/sandbox.root"
        #        file="/tmp/"+os.environ["USER"]+"/alcaRecoSkim-2.root"
        file="/tmp/"+os.environ["USER"]+"/alcaSkimSandbox.root"
        #file="/tmp/"+os.environ["USER"]+"/alcaSkimSandbox-noADCtoGeV.root"
        file_format="sandbox"
#sandbox"
        break
    elif(arg=='sandboxRecalib'):
        print 'sandbox recalib'
#        file="/tmp/"+os.environ["USER"]+"/Test-RecalibSandbox-GT_IC_LC.root"
#        file="/tmp/"+os.environ["USER"]+"/SandboxReReco-noADCtoGeV.root"
        file="/tmp/"+os.environ["USER"]+"/SandboxReReco.root"
#        file="/tmp/"+os.environ["USER"]+"/SandboxReReco-GTprompt-ALCARECO.root"
#        file="/tmp/"+os.environ["USER"]+"/SANDBOX/SandboxReReco.root"
#        file="/tmp/"+os.environ["USER"]+"/SANDBOX/SandboxReReco-noFranz2.root"
        file_format="sandboxRecalib"
        
        break
    elif(arg=='RECO'):
        print 'RECO'
        file="/tmp/"+os.environ["USER"]+"/SANDBOX/RAW-RECO.root"
        file_format="RECO"
        break
    else:
         continue
#         exit(0)



events = Events (file)
print file
handleElectrons = Handle('std::vector<reco::GsfElectron>')
handleRecHitsEB = Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')
handleRecHitsEE = Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')
handleRecHitsES = Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')

handleRecHitsEB_RECO = Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')
handleRecHitsEE_RECO = Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')
handleRecHitsEB_ALCASKIM = Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')
handleRecHitsEE_ALCASKIM = Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')
handleRecHitsEB_ALCARECO = Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')
handleRecHitsEE_ALCARECO = Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')

if (file_format == 'ALCARECO'):
    processName="ALCASKIM"
    electronTAG = 'electronRecalibSCAssociator'
elif(file_format == 'sandboxRecalib'):
    processName = "ALCARERECO"
    electronTAG = 'electronRecalibSCAssociator'
    recHitsTAG = "alCaIsolatedElectrons"
elif(file_format == 'sandbox'):
    processName = "ALCASKIM"
    electronTAG = 'electronRecalibSCAssociator'
elif(file_format == "AOD"):
    processName = "RECO"
    electronTAG = 'gedGsfElectrons'
elif(file_format == "AlcaFromAOD"):
    processName = "ALCASKIM"
    electronTAG = 'gedGsfElectrons'
elif(file_format == "AlcaFromAOD_Recalib"):
    electronTAG = 'electronRecalibSCAssociator'
    processName = 'ALCASKIM' 
elif(file_format == "RECO"):
    electronTAG = "gedGsfElectrons"
    processName = "RECO"
    



EErecHitmap_ele1 = TH2F("EErecHitmap_ele1", "EErecHitmap_ele1",
                   100,0,100,
                   100,0,100)

EBrecHitmap_ele1 = TH2F("EBrecHitmap_ele1", "EBrecHitmap_ele1",
                   171,-85,85,
                   360,0,360)

EErecHitmap_ele2 = TH2F("EErecHitmap_ele2", "EErecHitmap_ele2",
                   100,0,100,
                   100,0,100)

EBrecHitmap_ele2 = TH2F("EBrecHitmap_ele2", "EBrecHitmap_ele2",
                   171,-85,85,
                   360,0,360)

print file_format, file, electronTAG, processName, maxEvents

print "run\tlumi, event, energy, eSC, rawESC, e5x5, E_ES, etaEle, phiEle, etaSC, phiSC, clustersSize, nRecHits"
for event in events:

    if(maxEvents > 0 and event_counter > maxEvents):
        break
    #if(event.eventAuxiliary.run()== 145351895):
    if lumi > 0 and int(event.eventAuxiliary().luminosityBlock()) != lumi :
            continue

    if(eventNumber > 0 and event.eventAuxiliary().event()!= eventNumber ):
        continue

        #    event.getByLabel(electronTAG, "", processName, handleElectrons)
    event.getByLabel(electronTAG, handleElectrons)
    #    print file_format, file, electronTAG        
    electrons = handleElectrons.product()

    #    event.getByLabel("reducedEcalRecHitsEB", "", processName, handleRecHitsEB)
    #    event.getByLabel("reducedEcalRecHitsEE", "", processName, handleRecHitsEE)
    if(file_format=="sandbox"):
        event.getByLabel("ecalRecHit",    "EcalRecHitsEB",   "RECO", handleRecHitsEB_RECO)
        event.getByLabel("ecalRecHit",    "EcalRecHitsEE",   "RECO", handleRecHitsEE_RECO)
        event.getByLabel("ecalRecHit",    "EcalRecHitsEB",   "ALCASKIM", handleRecHitsEB_ALCASKIM)
        event.getByLabel("ecalRecHit",    "EcalRecHitsEE",   "ALCASKIM", handleRecHitsEE_ALCASKIM)
    else:
        event.getByLabel("alCaIsolatedElectrons","alcaBarrelHits", "ALCARERECO", handleRecHitsEB_ALCARECO)
        event.getByLabel("alCaIsolatedElectrons","alcaEndcapHits", "ALCARERECO", handleRecHitsEE_ALCARECO)

    if(file_format=="sandbox"):

       for electron in electrons:
           if(abs(electron.eta()) < 1.4442):
               recHits_RECO = handleRecHitsEB_RECO.product()
               recHits_ALCASKIM = handleRecHitsEB_ALCASKIM.product()
           else:
               recHits_RECO = handleRecHitsEE_RECO.product()
               recHits_ALCASKIM = handleRecHitsEE_ALCASKIM.product()
       
           nRecHits_RECO=0
           for recHit in recHits_RECO:
               nRecHits_RECO=nRecHits_RECO+1
#                if(recHit.checkFlag(EcalRecHit.kTowerRecovered)):
#                   print recHit.id().rawId()
       
           nRecHits_ALCASKIM=0
           for recHit in recHits_ALCASKIM:
               nRecHits_ALCASKIM=nRecHits_ALCASKIM+1
               #               if(recHit.checkFlag(EcalRecHit.kTowerRecovered)):
               print recHit.id().rawId(), recHit.checkFlag(EcalRecHit.kTowerRecovered)
               
           if(nRecHits_ALCASKIM != nRecHits_RECO):
               print nRecHits_RECO, nRecHits_ALCASKIM
               print recHits_RECO
               print "------------------------------"
               print recHits_ALCASKIM
    else:
       for electron in electrons:
           if(abs(electron.eta()) < 1.4442):
               recHits_ALCARECO = handleRecHitsEB_ALCARECO.product()
           else:
               recHits_ALCARECO = handleRecHitsEE_ALCARECO.product()
       
           nRecHits_ALCARECO=0
           for recHit in recHits_ALCARECO:
               nRecHits_ALCARECO=nRecHits_ALCARECO+1
               #               if(recHit.checkFlag(EcalRecHit.kTowerRecovered)):
               print recHit.id().rawId(), recHit.checkFlag(EcalRecHit.kTowerRecovered)
               
print event_counter



