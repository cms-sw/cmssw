import FWCore.ParameterSet.Config as cms
import ROOT

from DataFormats.FWLite import Handle, Events
from copy import deepcopy

class egmSmearer:

    uncertainties = ['ScaleStatUp', 'ScaleStatDown', 
                     'ScaleSystUp', 'ScaleSystDown',
                     'ScaleGainUp', 'ScaleGainDown',
                     'ResolutionRhoUp', 'ResolutionRhoDown',
                     'ResolutionPhiUp', 'ResolutionPhiDown']

    simplifiedUncertainties = ['ScaleUp', 'ScaleDown', 
                               'ResolutionUp', 'ResolutionDown']

class uncertaintyType(egmSmearer):
    
    for i,j in zip(egmSmearer.uncertainties, range(len(egmSmearer.uncertainties))):
        locals()[i] = j
            
class simplifiedUncertaintyType(egmSmearer):
    
    for i,j in zip(egmSmearer.simplifiedUncertainties, range(len(egmSmearer.uncertainties))):
        locals()[i] = j

class egammaEnergyShifter(object):

    def __init__(self):
        print ' --- Creating EGM uncertainty shifter --- '

    def setConsume(self, pythonpset):
        for i in egmSmearer.uncertainties + egmSmearer.simplifiedUncertainties:
            setattr(self, i[0].lower()+i[1:]+'Uncertainty', Handle('edm::ValueMap<float>'))
            setattr(self, i[0].lower()+i[1:]+'UncertaintyLabel', getattr(pythonpset, i[0].lower()+i[1:]+'Uncertainty').value())

    def setEvent(self, event):
        for i in egmSmearer.uncertainties + egmSmearer.simplifiedUncertainties:
            event.getByLabel(getattr(self, i[0].lower()+i[1:]+'UncertaintyLabel'), getattr(self, i[0].lower()+i[1:]+'Uncertainty'))

    def getSimpleShiftedCalibratedEnergy(self, index, uncertaintyValue): 
        uncertaintyName = egmSmearer.simplifiedUncertainties[uncertaintyValue]
        return getattr(self, uncertaintyName[0].lower()+uncertaintyName[1:]+'Uncertainty').product().get(index)

    def getShiftedCalibratedEnergy(self, index, uncertaintyValue): 
        uncertaintyName = egmSmearer.uncertainties[uncertaintyValue]
        return getattr(self, uncertaintyName[0].lower()+uncertaintyName[1:]+'Uncertainty').product().get(index)

class electronEnergyShifter(egammaEnergyShifter):

    def __init__(self, pythonpset):
        self.electrons = Handle('std::vector<pat::Electron>')
        self.electronLabel = [pythonpset.calibratedElectrons.moduleLabel, pythonpset.calibratedElectrons.processName, pythonpset.calibratedElectrons.productInstanceLabel]
        super(electronEnergyShifter, self).setConsume(pythonpset)

    def setEvent(self, event):
        event.getByLabel(self.electronLabel, self.electrons)
        super(electronEnergyShifter, self).setEvent(event)

    def getSimpleShiftedObject(self, electronIndex, uncertaintyValue):        
        shiftedElectron = deepcopy(self.electrons.product().at(electronIndex))
        energyResolution = shiftedElectron.p4Error(1)
        trackResolution = shiftedElectron.trackMomentumError()
        newEnergy = super(electronEnergyShifter, self).getSimpleShiftedCalibratedEnergy(electronIndex, uncertaintyValue)
        shiftedElectron.correctMomentum(shiftedElectron.p4(1) * (newEnergy/shiftedElectron.energy()), trackResolution, energyResolution)
        return shiftedElectron

    def getShiftedObject(self, electronIndex, uncertaintyValue):        
        shiftedElectron = deepcopy(self.electrons.product().at(electronIndex))
        energyResolution = shiftedElectron.p4Error(1)
        trackResolution = shiftedElectron.trackMomentumError()
        newEnergy = super(electronEnergyShifter, self).getShiftedCalibratedEnergy(electronIndex, uncertaintyValue)
        shiftedElectron.correctMomentum(shiftedElectron.p4(1) * (newEnergy/shiftedElectron.energy()), trackResolution, energyResolution)
        return shiftedElectron

class photonEnergyShifter(egammaEnergyShifter):

    def __init__(self, pythonpset):
        self.photons = Handle('std::vector<pat::Photon>')
        self.photonLabel = [pythonpset.calibratedPhotons.moduleLabel, pythonpset.calibratedPhotons.processName, pythonpset.calibratedPhotons.productInstanceLabel]
        super(photonEnergyShifter, self).setConsume(pythonpset)

    def setEvent(self, event):
        event.getByLabel(self.photonLabel, self.photons)
        super(photonEnergyShifter, self).setEvent(event)

    def getSimpleShiftedObject(self, photonIndex, uncertaintyValue):        
        shiftedPhoton = deepcopy(self.photons.product().at(photonIndex))
        energyResolution = shiftedPhoton.getCorrectedEnergyError(3)
        newEnergy = super(photonEnergyShifter, self).getSimpleShiftedCalibratedEnergy(photonIndex, uncertaintyValue)
        shiftedPhoton.setCorrectedEnergy(3, newEnergy, energyResolution, True)
        return shiftedPhoton

    def getShiftedObject(self, photonIndex, uncertaintyValue):        
        shiftedPhoton = deepcopy(self.photons.product().at(photonIndex))
        energyResolution = shiftedPhoton.getCorrectedEnergyError(3)
        newEnergy = super(photonEnergyShifter, self).getShiftedCalibratedEnergy(photonIndex, uncertaintyValue)
        shiftedPhoton.setCorrectedEnergy(3, newEnergy, energyResolution, True)
        return shiftedPhoton
            
        
