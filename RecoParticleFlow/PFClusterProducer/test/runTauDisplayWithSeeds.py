import ROOT
import itertools
import math
from DataFormats.FWLite import Events, Handle
from Display import *

hfH  = Handle('std::vector<reco::PFRecHit>')
ecalH  = Handle ('std::vector<reco::PFRecHit>')
ekH = Handle('std::vector<reco::PFRecHit>')
hcalH  = Handle ('std::vector<reco::PFRecHit>')
genParticlesH  = Handle ('std::vector<reco::GenParticle>')
tracksH  = Handle ('std::vector<reco::PFRecTrack>')
ecalClustersH  = Handle ('std::vector<reco::PFCluster>')
hcalClustersH  = Handle ('std::vector<reco::PFCluster>')
#arborH  = Handle ('std::vector<reco::PFCluster>')
hcalSeedsH  = Handle ('std::vector<reco::PFRecHit>')
simH = Handle('std::vector<reco::PFSimParticle>')
simHitsH = Handle('std::vector<PCaloHit>')
simTrackH = Handle('std::vector<SimTrack>')
simVertexH = Handle('std::vector<SimVertex>')
events = Events('reco.root')


for event in events:
    event.getByLabel('particleFlowRecHitHF',hfH)
    #event.getByLabel('particleFlowRecHitECALWithTime',ecalH)
    #event.getByLabel('particleFlowRecHitEK',ekH)
    event.getByLabel('particleFlowRecHitHBHEHO',hcalH)
    event.getByLabel('genParticles',genParticlesH)
    event.getByLabel('pfTrack',tracksH)
    #event.getByLabel('particleFlowClusterECAL',ecalClustersH)
    event.getByLabel('hcalSeeds',hcalSeedsH)
    event.getByLabel('particleFlowClusterHCALSemi3D',hcalClustersH)
    event.getByLabel('particleFlowSimParticle',simH)
    event.getByLabel('g4SimHits','HcalHits',simHitsH)
    event.getByLabel('g4SimHits',simTrackH)
    event.getByLabel('g4SimHits',simVertexH)
#    event.getByLabel('arbor','allLinks',arborH)
    

    hf = hfH.product()
    #ecal = ecalH.product()
    #ek = ekH.product()
    hcal = hcalH.product()
    genParticles = genParticlesH.product()
    tracks = tracksH.product()
    #ecalClusters = ecalClustersH.product()
    hcalClusters = hcalClustersH.product()
    hcalSeeds = hcalSeedsH.product()
    sim = simH.product()
    simCaloHits = simHitsH.product()
    simTracks = simTrackH.product()
    simVtxs = simVertexH.product()
#    allArborLinks = arborH.product()

    
    #find taus:
    for particle in genParticles:
        if abs(particle.pdgId())==211 and abs(particle.eta())<3.0 and particle.status()==1 and particle.pt()>5:
            #tau found define displays
            print particle.pdgId(),particle.energy(),particle.eta(),particle.phi()
            displayHCAL = DisplayManager('HCAL',particle.eta(),particle.phi(),0.5)
            displayECAL = DisplayManager('ECAL',particle.eta(),particle.phi(),0.5)
            displayAll = DisplayManager('EVERYTHING',particle.eta(),particle.phi(),0.5)
            displayHCALSeeds = DisplayManager('HCALSeeds',particle.eta(),particle.phi(),0.5)
            
            #reloop on gen particles and add them in view
            for particle1 in genParticles:
                if particle.status()!=1:
                    continue
                displayHCAL.addGenParticle(particle1) 
                displayHCALSeeds.addGenParticle(particle1) 
                displayECAL.addGenParticle(particle1) 
                displayAll.addGenParticle(particle1) 

            for simpart in sim:
                displayHCAL.addSimParticle(simpart) 
                displayHCALSeeds.addSimParticle(simpart) 
                displayECAL.addSimParticle(simpart) 
                displayAll.addSimParticle(simpart) 
            
            #add HF hits    
            for hit in hf:
                displayHCAL.addRecHit(hit,hit.depth())
                displayAll.addRecHit(hit,hit.depth())
            #add HCAL hits    
            for hit in hcal:
                displayHCAL.addRecHit(hit,hit.depth())
                displayAll.addRecHit(hit,hit.depth())
                
            #add HCAL seeds    
            for hit in hcalSeeds:
                displayHCALSeeds.addRecHit(hit,hit.depth())

            #add HCAL clusters   
            for cluster in hcalClusters:
                #print cluster.recHitFractions().size()                    
                #for rhf in cluster.recHitFractions():
                #    print rhf.recHitRef().energy(), rhf.fraction()
                displayHCAL.addCluster(cluster,True)
                displayAll.addCluster(cluster,True)

            #add ECAL hits    
            #for hit in ecal:
            #    displayECAL.addRecHit(hit,hit.depth())
            #    displayAll.addRecHit(hit,hit.depth())

            #for hit in ek:
            #    displayECAL.addRecHit(hit,hit.depth())
            #    displayAll.addRecHit(hit,hit.depth())

            #Add tracks
            for track in tracks:
                displayHCAL.addTrack(track) 
                displayHCALSeeds.addTrack(track) 
                displayECAL.addTrack(track) 
                displayAll.addTrack(track)

            #Add cluster
            #for cluster in ecalClusters:
            #    displayECAL.addCluster(cluster,True) 

            #add all arb.or links     
#            for cluster in allArborLinks:
#                displayHCALSeeds.addCluster(cluster,True) 
                
            displayECAL.viewEtaPhi()    
            displayHCAL.viewEtaPhi()    
            displayHCALSeeds.viewEtaPhi()    
            displayAll.viewEtaPhi()    


            try:
                input("Press enter to continue")
            except SyntaxError:
                pass



