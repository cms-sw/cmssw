import FWCore.ParameterSet.Config as cms

BtagPerformanceESProducer_SYSTEM8SSVM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('SYSTEM8SSVM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagSYSTEM8SSVMtable_v1_offline'),
    WorkingPointName = cms.string('BTagSYSTEM8SSVMwp_v1_offline')
)
BtagPerformanceESProducer_SYSTEM8SSVT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('SYSTEM8SSVT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagSYSTEM8SSVTtable_v1_offline'),
    WorkingPointName = cms.string('BTagSYSTEM8SSVTwp_v1_offline')
)
BtagPerformanceESProducer_SYSTEM8TCHEL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('SYSTEM8TCHEL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagSYSTEM8TCHELtable_v1_offline'),
    WorkingPointName = cms.string('BTagSYSTEM8TCHELwp_v1_offline')
)
BtagPerformanceESProducer_SYSTEM8TCHEM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('SYSTEM8TCHEM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagSYSTEM8TCHEMtable_v1_offline'),
    WorkingPointName = cms.string('BTagSYSTEM8TCHEMwp_v1_offline')
)
BtagPerformanceESProducer_SYSTEM8TCHET = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('SYSTEM8TCHET'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagSYSTEM8TCHETtable_v1_offline'),
    WorkingPointName = cms.string('BTagSYSTEM8TCHETwp_v1_offline')
)
BtagPerformanceESProducer_SYSTEM8TCHPL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('SYSTEM8TCHPL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagSYSTEM8TCHPLtable_v1_offline'),
    WorkingPointName = cms.string('BTagSYSTEM8TCHPLwp_v1_offline')
)
BtagPerformanceESProducer_SYSTEM8TCHPM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('SYSTEM8TCHPM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagSYSTEM8TCHPMtable_v1_offline'),
    WorkingPointName = cms.string('BTagSYSTEM8TCHPMwp_v1_offline')
)
BtagPerformanceESProducer_SYSTEM8TCHPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('SYSTEM8TCHPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagSYSTEM8TCHPTtable_v1_offline'),
    WorkingPointName = cms.string('BTagSYSTEM8TCHPTwp_v1_offline')
)
BtagPerformanceESProducer_MISTAGJPL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGJPLtable_v1_offline'),
    WorkingPointName = cms.string('BTagMISTAGJPLwp_v1_offline')
)
BtagPerformanceESProducer_MISTAGJPM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGJPMtable_v1_offline'),
    WorkingPointName = cms.string('BTagMISTAGJPMwp_v1_offline')
)
BtagPerformanceESProducer_MISTAGJPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGJPTtable_v1_offline'),
    WorkingPointName = cms.string('BTagMISTAGJPTwp_v1_offline')
)
BtagPerformanceESProducer_MISTAGSSVM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGSSVM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGSSVMtable_v1_offline'),
    WorkingPointName = cms.string('BTagMISTAGSSVMwp_v1_offline')
)
BtagPerformanceESProducer_MISTAGTCHEL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGTCHEL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGTCHELtable_v1_offline'),
    WorkingPointName = cms.string('BTagMISTAGTCHELwp_v1_offline')
)
BtagPerformanceESProducer_MISTAGTCHEM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGTCHEM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGTCHEMtable_v1_offline'),
    WorkingPointName = cms.string('BTagMISTAGTCHEMwp_v1_offline')
)
BtagPerformanceESProducer_MISTAGTCHPM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGTCHPM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGTCHPMtable_v1_offline'),
    WorkingPointName = cms.string('BTagMISTAGTCHPMwp_v1_offline')
)
BtagPerformanceESProducer_MISTAGTCHPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGTCHPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGTCHPTtable_v1_offline'),
    WorkingPointName = cms.string('BTagMISTAGTCHPTwp_v1_offline')
)
BtagPerformanceESProducer_PTRELJBPL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELJBPL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELJBPLtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELJBPLwp_v1_offline')
)
BtagPerformanceESProducer_PTRELJBPM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELJBPM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELJBPMtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELJBPMwp_v1_offline')
)
BtagPerformanceESProducer_PTRELJBPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELJBPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELJBPTtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELJBPTwp_v1_offline')
)
BtagPerformanceESProducer_PTRELJPL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELJPL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELJPLtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELJPLwp_v1_offline')
)
BtagPerformanceESProducer_PTRELJPM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELJPM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELJPMtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELJPMwp_v1_offline')
)
BtagPerformanceESProducer_PTRELJPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELJPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELJPTtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELJPTwp_v1_offline')
)
BtagPerformanceESProducer_PTRELSSVL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELSSVL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELSSVLtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELSSVLwp_v1_offline')
)
BtagPerformanceESProducer_PTRELSSVM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELSSVM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELSSVMtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELSSVMwp_v1_offline')
)
BtagPerformanceESProducer_PTRELSSVT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELSSVT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELSSVTtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELSSVTwp_v1_offline')
)
BtagPerformanceESProducer_PTRELTCHEL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELTCHEL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELTCHELtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELTCHELwp_v1_offline')
)
BtagPerformanceESProducer_PTRELTCHEM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELTCHEM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELTCHEMtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELTCHEMwp_v1_offline')
)
BtagPerformanceESProducer_PTRELTCHET = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELTCHET'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELTCHETtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELTCHETwp_v1_offline')
)
BtagPerformanceESProducer_PTRELTCHPL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELTCHPL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELTCHPLtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELTCHPLwp_v1_offline')
)
BtagPerformanceESProducer_PTRELTCHPM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELTCHPM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELTCHPMtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELTCHPMwp_v1_offline')
)
BtagPerformanceESProducer_PTRELTCHPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('PTRELTCHPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagPTRELTCHPTtable_v1_offline'),
    WorkingPointName = cms.string('BTagPTRELTCHPTwp_v1_offline')
)
