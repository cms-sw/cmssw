import FWCore.ParameterSet.Config as cms

def customise_hgcalmapper(process,
                          modules = 'Geometry/HGCalMapping/data/ModuleMaps/modulelocator_test.txt',
                          sicells = 'Geometry/HGCalMapping/data/CellMaps/WaferCellMapTraces.txt',
                          sipmcells = 'Geometry/HGCalMapping/data/CellMaps/channels_sipmontile.hgcal.txt',
                          offsetfile = 'Geometry/HGCalMapping/data/CellMaps/offsetMap.txt'):
    """the following function configures the mapping producers
    NOTE: for production-targetted configs should be avoided as it checks if the process as 
    already the Accelerators sequence loaded, if not it loads it to the process"""

    process.load('Geometry.HGCalMapping.hgCalMappingESProducer_cfi')
    process.hgCalMappingESProducer.modules = cms.FileInPath(modules)
    process.hgCalMappingESProducer.si = cms.FileInPath(sicells)
    process.hgCalMappingESProducer.sipm = cms.FileInPath(sipmcells)

    if not hasattr(process, 'ProcessAcceleratorCUDA'):
        process.load('Configuration.StandardSequences.Accelerators_cff')
        
    process.hgCalMappingCellESProducer = cms.ESProducer('hgcal::HGCalMappingCellESProducer@alpaka',
                                                        filelist=cms.vstring(sicells, sipmcells),
                                                        cellindexer=cms.ESInputTag(''),
                                                        offsetfile=cms.string(offsetfile))
    process.hgCalMappingModuleESProducer = cms.ESProducer('hgcal::HGCalMappingModuleESProducer@alpaka',
                                                          filename=cms.FileInPath(modules),
                                                          moduleindexer=cms.ESInputTag(''))
    process.hgCalDenseIndexInfoESProducer = cms.ESProducer('hgcal::HGCalDenseIndexInfoESProducer@alpaka',
                                                           moduleindexer=cms.ESInputTag('') )
    
    return process
