import FWCore.ParameterSet.Config as cms

def customise_hgcalmapper(process,
                          modules = 'Geometry/HGCalMapping/data/ModuleMaps/modulelocator_test.txt',
                          sicells = 'Geometry/HGCalMapping/data/CellMaps/WaferCellMapTraces.txt',
                          sipmcells = 'Geometry/HGCalMapping/data/CellMaps/channels_sipmontile.hgcal.txt'):

    process.load('Geometry.HGCalMapping.hgCalMappingESProducer_cfi')
    process.hgCalMappingESProducer.modules = cms.FileInPath(modules)
    process.hgCalMappingESProducer.si = cms.FileInPath(sicells)
    process.hgCalMappingESProducer.sipm = cms.FileInPath(sipmcells)

    if not hasattr(process, 'ProcessAcceleratorCUDA'):
        print('Looks like Configuration.StandardSequences.Accelerators_cff was not yet loaded...loading')
        process.load('Configuration.StandardSequences.Accelerators_cff')
        
    process.hgCalMappingCellESProducer = cms.ESProducer('hgcal::HGCalMappingCellESProducer@alpaka',
                                                        filelist=cms.vstring(sicells, sipmcells),
                                                        cellindexer=cms.ESInputTag(''))
    process.hgCalMappingModuleESProducer = cms.ESProducer('hgcal::HGCalMappingModuleESProducer@alpaka',
                                                          filename=cms.FileInPath(modules),
                                                          moduleindexer=cms.ESInputTag(''))
    process.hgCalDenseIndexInfoESProducer = cms.ESProducer('hgcal::HGCalDenseIndexInfoESProducer@alpaka',
                                                           moduleindexer=cms.ESInputTag('') )
    
    return process
