import FWCore.ParameterSet.Config as cms

file = open('tpDead.txt')
wiresToDebug = cms.untracked.vstring()
for line in file:
    corrWire = line.split()[:6]
    #switch station/sector
    corrWire[1:3] = corrWire[2:0:-1]
    wire = ' '.join(corrWire)
    #print wire
    wiresToDebug.append(wire)
file.close()

from CalibMuon.DTCalibration.dtT0WireCalibration_cfg import process
process.source.fileNames = ['/store/data/Commissioning12/MiniDaq/RAW/v1/000/185/686/A86FFC8B-DE5C-E111-AD81-003048D3C932.root']
process.GlobalTag.globaltag = 'GR_P_V28::All'

process.muonDTDigis.inputLabel = 'rawDataCollector'
process.dtTPmonitor.dtDigiLabel = 'muonDTDigis'
process.dtTPmonitor.readDB = False 
process.dtTPmonitor.defaultTtrig = 600
process.dtTPmonitor.defaultTmax = 100
#process.dtTPmonitor.readDB = True 
#process.dtTPmonitor.defaultTtrig = 300
#process.dtTPmonitor.defaultTmax = 100

process.dtT0WireCalibration.correctByChamberMean = False
process.dtT0WireCalibration.cellsWithHisto = wiresToDebug
#process.dtT0WireCalibration.eventsForLayerT0 = 3000
#process.dtT0WireCalibration.eventsForWireT0 = 12000
