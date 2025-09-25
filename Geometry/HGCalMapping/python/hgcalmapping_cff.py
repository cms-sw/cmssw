import FWCore.ParameterSet.Config as cms

def customise_hgcalmapper(process,
                          modules = 'Geometry/HGCalMapping/data/ModuleMaps/modulelocator_test.txt',
                          sicells = 'Geometry/HGCalMapping/data/CellMaps/WaferCellMapTraces.txt',
                          sipmcells = 'Geometry/HGCalMapping/data/CellMaps/channels_sipmontile.hgcal.txt',
                          offsetfile = 'Geometry/HGCalMapping/data/CellMaps/calibration_to_surrounding_offsetMap.txt',
                          trigmodule = 'Geometry/HGCalMapping/data/ModuleMaps/modulelocator_trigger_test.txt'):
    """the following function configures the mapping producers
    NOTE: for production-targetted configs should be avoided as it checks if the process as 
    already the Accelerators sequence loaded, if not it loads it to the process"""

    process.load('Geometry.HGCalMapping.hgCalMappingESProducer_cfi')
    process.hgCalMappingESProducer.modules = cms.FileInPath(modules)
    process.hgCalMappingESProducer.si = cms.FileInPath(sicells)
    process.hgCalMappingESProducer.sipm = cms.FileInPath(sipmcells)

    process.load('Geometry.HGCalMapping.hgCalMappingTriggerESProducer_cfi')
    process.hgCalMappingTriggerESProducer.modules = cms.FileInPath(trigmodule)
    process.hgCalMappingTriggerESProducer.si = cms.FileInPath(sicells)
    process.hgCalMappingTriggerESProducer.sipm = cms.FileInPath(sipmcells)

    if not hasattr(process, 'ProcessAcceleratorCUDA'):
        process.load('Configuration.StandardSequences.Accelerators_cff')
        
    process.hgCalMappingCellESProducer = cms.ESProducer('hgcal::HGCalMappingCellESProducer@alpaka',
                                                        filelist=cms.vstring(sicells, sipmcells),
                                                        cellindexer=cms.ESInputTag(''),
                                                        offsetfile=cms.FileInPath(offsetfile))
    process.hgCalMappingModuleESProducer = cms.ESProducer('hgcal::HGCalMappingModuleESProducer@alpaka',
                                                          filename=cms.FileInPath(modules),
                                                          moduleindexer=cms.ESInputTag(''))
    process.hgCalDenseIndexInfoESProducer = cms.ESProducer('hgcal::HGCalDenseIndexInfoESProducer@alpaka',
                                                           moduleindexer=cms.ESInputTag('') )

    process.hgCalMappingModuleTriggerESProducer = cms.ESProducer('hgcal::HGCalMappingTriggerModuleESProducer@alpaka',
                                                                filename=cms.FileInPath(trigmodule),
                                                                moduleindexer=cms.ESInputTag(''))
    process.hgCalDenseIndexTriggerInfoESProducer = cms.ESProducer('hgcal::HGCalDenseIndexTriggerInfoESProducer@alpaka',
                                                           moduleindexer=cms.ESInputTag('') )

    return process
