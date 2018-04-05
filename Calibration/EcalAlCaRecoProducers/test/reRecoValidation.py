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
eventListPrint=False
lumi=-1
#lumi=247
eventNumbers=[
343500081,
343652254,
344326842,
344404776,
345023043,
344758536,
344533037,
343780994,
344736267,
344828101,
344846924,
344263362,
344466033,
343687292,
344102889,
344150484,
343978421,
344657593,
344966978,
345140483,
343618665,
344354767,
344911692]
#eventNumbers=[64944437] # evento con trackerDrivenEle
eventNumbers=[]
eventMin=-1

# for now look for events in two files with a given lumi section
maxEvents=-1
event_counter=0

for arg in sys.argv:
    if(arg=='AOD'):
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
        file="./scratch/sandbox.root"
        #file="/tmp/"+os.environ["USER"]+"/alcaSkimSandbox-noADCtoGeV.root"
        file_format="sandbox"
#sandbox"
        break
    elif(arg=='sandboxRereco'):
        print 'sandbox recalib'

        
        #file="/tmp/"+os.environ["USER"]+"/SANDBOX/sandboxRereco.root"
        file="./alcaSkimSandbox1.root"
#        file="root://eoscms//eos/cms/store/group/alca_ecalcalib/sandboxRecalib/7TeV/fromUncalib/rereco30Nov/DoubleElectron-RUN2011B-v1/sandboxRereco-007.root"
        file_format="sandboxRecalib"
        break
    elif(arg=='RECO'):
        print 'RECO'
        file="root://eoscms//eos/cms/store/data/Run2012A/DoubleElectron/RECO/PromptReco-v1/000/193/336/BC442450-7997-E111-8177-003048D3C90E.root"
        #        file="/tmp/"+os.environ["USER"]+"/SANDBOX/RAW-RECO.root"
        file_format="RECO"
        break
    elif(arg=='MC'):
        print 'MC'
        file="/tmp/"+os.environ["USER"]+"/MC-AODSIM.root"
        file_format="AOD"
        break
    elif(arg=='ALCARECO'):
        print 'ALCARECO'
        file="alcaSkimSandbox_numEvent10000.root"
#        file="alcaSkimSandbox.root"
        file_format="ALCARECO"
        break

    else:
         continue
#         exit(0)


# AOD
# sandbox
#file_format = "ALCARECO"
#file_format = 'sandbox'
#file_format = 'AOD'
#file_format = "AlcaFromAOD"
#file_format = "AlcaFromAOD_Recalib"



#file='./alcaRecoSkim.root'
# if(file_format == 'AOD'):
#     file='/tmp/shervin/AOD.root'
# elif(file_format == 'AlcaFromAOD'):
#     file='/tmp/shervin/alcaRecoSkimFromAOD.root'
# elif(file_format == 'AlcaFromAOD_Recalib'):
#     file='/tmp/shervin/alcaRecoSkimFromAOD_Recalib.root'
#     file='/tmp/shervin/alcaRecoSkimFromAOD_Recalib_IC_LC-2.root'
    
#file='./root/alcaRecoSkimFromAOD_LC_IC.root'
#file='./root/alcaRecoSkimFromAOD_Recalib_IC_LC.root'
#file='./root/alcaRecoSkimFromSandBox_Recalib.root'
#file='/tmp/shervin/alcaRecoSkim-1.root'
#file='/tmp/shervin/myAlcaRecoSkim.root'

print file
events = Events (file)

handleElectrons = Handle('std::vector<reco::GsfElectron>')
handleRecHitsEB = Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')
handleRecHitsEE = Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')
handleRecHitsES = Handle('edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >')
handleRhoFastJet = Handle('double')

if (file_format == 'ALCARECO'):
    processName="ALCARECO"
    electronTAG = 'gedGsfElectrons'
elif(file_format == 'sandboxRecalib'):
    processName = "ALCARERECO"
    electronTAG = 'electronRecalibSCAssociator'
#    recHitsTAG = "alCaIsolatedElectrons"
elif(file_format == 'sandbox'):
    processName = "ALCASKIM"
#    electronTAG = 'electronRecalibSCAssociator'
    electronTAG = 'gedGsfElectrons'
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

print "run    lumi  event     isEB    energy     eSC  rawESC    e5x5    E_ES, etaEle, phiEle, etaSC, phiSC, clustersSize, nRecHits"
for event in events:

#    if(event_counter % 100):
#        print "[STATUS] event ", event_counter
        
    if(maxEvents > 0 and event_counter > maxEvents):
        break
    if(eventListPrint==True):
        print event.eventAuxiliary().run(), event.eventAuxiliary().luminosityBlock(), event.eventAuxiliary().event()
        continue
    #if(event.eventAuxiliary.run()== 145351895):
    if lumi > 0 and int(event.eventAuxiliary().luminosityBlock()) != lumi :
            continue

    if(len(eventNumbers)>0 and not (event.eventAuxiliary().event() in eventNumbers)):
       continue


        #    event.getByLabel(electronTAG, "", processName, handleElectrons)
    event.getByLabel(electronTAG, handleElectrons)
    #    print file_format, file, electronTAG        
    electrons = handleElectrons.product()


    if(file_format=="AOD"):
        event.getByLabel("reducedEcalRecHitsEB", "", processName, handleRecHitsEB)
        event.getByLabel("reducedEcalRecHitsEE", "", processName, handleRecHitsEE)
        event.getByLabel("reducedEcalRecHitsES", "", processName, handleRecHitsES)
#        print "##############", 
        #        rhoTAG=edm.InputTag()
        #        rhoTAG=("kt6PFJets","rho","RECO")
#        event.getByLabel("kt6PFJets","rho","RECO",handleRhoFastJet)
        # elif(file_format=="sandboxRecalib" or file_format=="RECO"):
    elif(file_format=="RECO"):
        event.getByLabel("ecalRecHit", "EcalRecHitsEB", processName, handleRecHitsEB)
        event.getByLabel("ecalRecHit", "EcalRecHitsEE", processName, handleRecHitsEE)
    elif(file_format=="ALCARECO" or file_format=="sandboxRecalib"):
        event.getByLabel("alCaIsolatedElectrons", "alCaRecHitsEB", processName, handleRecHitsEB)
        event.getByLabel("alCaIsolatedElectrons", "alCaRecHitsEE", processName, handleRecHitsEE)
#        event.getByLabel("kt6PFJets","rho","RECO",handleRhoFastJet)
#        event.getByLabel("reducedEcalRecHitsES", "", processName, handleRecHitsES)
 
        #        event.getByLabel("ecalRecHit", "EcalRecHitsES", processName, handleRecHitsES)

#    elif(file_format=="sandbox"):
#        event.getByLabel("ecalRecHit", "EcalRecHitsEB", processName, handleRecHitsEB)
#        event.getByLabel("ecalRecHit", "EcalRecHitsEE", processName, handleRecHitsEE)
#    else:
        
    
#    print "Num of electrons: ",len(electrons)
    if(len(electrons)>=2):
     ele_counter=0
     for electron in electrons:
        if(not electron.ecalDrivenSeed()): 
            print "trackerDriven",
#            sys.exit(0)
        electron.superCluster().energy()

         
        #        ESrecHits = handleRecHitsES.product()
        #        if(abs(electron.eta()) > 1.566):
        #            for ESrecHit in ESrecHits:
        #                if(eventNumber >0):
        #                    esrecHit = ESDetId(ESrecHit.id().rawId())
        #                    print ESrecHit.id()(), esrecHit.strip(), esrecHit.six(), esrecHit.siy(), esrecHit.plane()
        print "------------------------------"
        if(not file_format=="sandbox"):
         if(electron.isEB()):
             recHits = handleRecHitsEB.product()
         else:
             recHits = handleRecHitsEE.product()
         nRecHits=0
         for recHit in recHits:
             nRecHits=nRecHits+1
             if(len(eventNumbers)==1):
                 if(electron.isEB()):
                     EBrecHit = EBDetId(recHit.id().rawId())
                     if(allRecHits):
                         if(ele_counter==0):
                             EBrecHitmap_ele1.Fill(EBrecHit.ieta(), EBrecHit.iphi(), recHit.energy());
                         elif(ele_counter==1):
                             EBrecHitmap_ele2.Fill(EBrecHit.ieta(), EBrecHit.iphi(), recHit.energy());
                         
                     print recHit.id()(), EBrecHit.ieta(), EBrecHit.iphi(), recHit.energy(), recHit.checkFlag(0)
                 else:
                     EErecHit = EEDetId(recHit.id().rawId())
                     if(allRecHits):
                         if(ele_counter==0):
                             EErecHitmap_ele1.Fill(EErecHit.ix(), EErecHit.iy(), recHit.energy());
                         elif(ele_counter==1):
                             EErecHitmap_ele2.Fill(EErecHit.ix(), EErecHit.iy(), recHit.energy());
                     print recHit.id()(), EErecHit.ix(), EErecHit.iy(), recHit.energy()
 
         hits = electron.superCluster().hitsAndFractions() 
         nRecHitsSC=0
         for hit in hits:
             nRecHitsSC=nRecHitsSC+1
             if(len(eventNumbers)==1):
                 if(electron.isEB()):
                     EBrecHit = EBDetId(hit.first.rawId())
                     if(not allRecHits):
                         if(ele_counter==0):
                             EBrecHitmap_ele1.Fill(EBrecHit.ieta(), EBrecHit.iphi(), hit.second*electron.superCluster().energy());
                         elif(ele_counter==1):
                             EBrecHitmap_ele2.Fill(EBrecHit.ieta(), EBrecHit.iphi(), hit.second*electron.superCluster().energy());
                     print "SC", (hit.first).rawId(), (hit.first)(), EBrecHit.ieta(), EBrecHit.iphi(),  EBrecHit.tower().iTT(), EBrecHit.ism(), EBrecHit.im()
                 else:
                     EErecHit = EEDetId(hit.first.rawId())
                     if(not allRecHits):
                         if(ele_counter==0):
                             EErecHitmap_ele1.Fill(EErecHit.ix(), EErecHit.iy(), recHit.energy());
                         elif(ele_counter==1):
                             EErecHitmap_ele2.Fill(EErecHit.ix(), EErecHit.iy(), recHit.energy());
 
 #                print "SC ", (hit.first)()
            
        print event.eventAuxiliary().run(), event.eventAuxiliary().luminosityBlock(), event.eventAuxiliary().event(),
        print "isEB=",electron.isEB(),
        print '{0:7.3f} {1:7.3f} {2:7.3f} {3:7.3f} {4:7.3f}'.format(electron.energy(), electron.superCluster().energy(), electron.superCluster().rawEnergy(), electron.e5x5(), electron.superCluster().preshowerEnergy()),
        print '{0:6.3f} {1:6.3f} {2:6.3f} {3:6.3f}'.format(electron.eta(), electron.phi(), electron.superCluster().eta(), electron.superCluster().phi()),
        print electron.superCluster().clustersSize(), nRecHits, nRecHitsSC
        ele_counter+=1
        
        
    event_counter+=1

print event_counter

# setting maps to -999
gStyle.SetPaintTextFormat("1.1f")
EBrecHitmap_ele1.SaveAs("EBrecHitmap_ele1.root")
EErecHitmap_ele1.SaveAs("EErecHitmap_ele1.root")
EBrecHitmap_ele2.SaveAs("EBrecHitmap_ele2.root")
EErecHitmap_ele2.SaveAs("EErecHitmap_ele2.root")



