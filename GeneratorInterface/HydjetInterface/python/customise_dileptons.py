
def customise(process):

    process.generator.PythiaParameters.hydjetPythiaDefault = cms.vstring('MSEL=0   ! user processes',
                                                                         'CKIN(3)=6.',# ! ptMin
                                                                         'MSTP(81)=0'
                                                                         )

    process.generator.PythiaParameters.parameterSets = cms.vstring('pythiaUESettings',
                                                                   'hydjetPythiaDefault',
                                                                   'pythiaJets',
                                                                   'pythiaPromptPhotons',
                                                                   'pythiaZjets',
                                                                   'pythiaBottomoniumNRQCD',
                                                                   'pythiaCharmoniumNRQCD',
                                                                   'pythiaQuarkoniaSettings',
                                                                   'pythiaWeakBosons'
                                                                   )
    return process
