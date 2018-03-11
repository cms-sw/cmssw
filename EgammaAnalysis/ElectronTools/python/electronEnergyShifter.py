import FWCore.ParameterSet.Config as cms
from DataFormats.FWLite import Handle, Events

class electronEnergyShifter:
    
    def __init__(self, pythonpset):
        self.calibratedElectronLabel = label

        self.electrons, self.electronLabel = Handle("std::vector<pat::Electron>"), "calibratedPatElectrons::".split(":")
        scaleUp, scaleUpLabel = Handle("edm::ValueMap<float>"), "calibratedPatElectrons:EGMscaleUpUncertainty:".split(":")
        scaleDown, scaleDownLabel = Handle("edm::ValueMap<float>"), "calibratedPatElectrons:EGMscaleDownUncertainty:".split(":")
        resolutionUp, resolutionUpLabel = Handle("edm::ValueMap<float>"), "calibratedPatElectrons:EGMresolutionUpUncertainty:".split(":")
        resolutionDown, resolutionDownLabel = Handle("edm::ValueMap<float>"), "calibratedPatElectrons:EGMresolutionDownUncertainty:".split(":")
        
for iev,event in enumerate(events):
    
    print iev
    event.getByLabel(electronLabel[0],electronLabel[1],electronLabel[2], electrons)
    event.getByLabel(scaleUpLabel[0],scaleUpLabel[1],scaleUpLabel[2], scaleUp)
    event.getByLabel(scaleDownLabel[0],scaleDownLabel[1],scaleDownLabel[2], scaleDown)
    event.getByLabel(resolutionUpLabel[0],resolutionUpLabel[1],resolutionUpLabel[2], resolutionUp)
    event.getByLabel(resolutionDownLabel[0],resolutionDownLabel[1],resolutionDownLabel[2], resolutionDown)
    
    for i,el in enumerate(electrons.product()):
        if el.pt() < 5: continue
        print el.energy(), scaleUp.product().get(i), scaleDown.product().get(i), resolutionUp.product().get(i), resolutionDown.product().get(i)
#        print scaleUp.
