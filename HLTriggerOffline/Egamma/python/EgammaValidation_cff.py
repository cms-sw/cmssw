import FWCore.ParameterSet.Config as cms

# whether to use the old or newer (automatically adapting
# to the MC menu) method of configuring the monitoring
# modules for the HLT paths
use_new_method = False


if not use_new_method:
    #----------------------------------------------------------------------
    # traditional method of configuring the HLT paths
    #----------------------------------------------------------------------

    class dummy:
        pass

    samples=dummy()
    paths=dummy()

    ##########################################################
    # Define which preselections to run                      #
    ##########################################################

    samples.names = ['Wenu',
                     'Zee',
                     'GammaJet',
                     'DiGamma']
    samples.pdgid = [ 11,
                      11,
                      22,
                      22]
    samples.num   = [1,
                     2,
                     1,
                     2]

    #which triggers for which sample

    paths.Wenu = [
                  'HLT_Ele17_SW_TighterEleIdIsol_L1RDQM',

                  'HLT_Ele10_LW_L1RDQM',
                  'HLT_Ele15_SW_L1RDQM',
                  'HLT_Ele10_LW_EleId_L1RDQM',
                  'HLT_Ele15_SiStrip_L1RDQM']

    paths.Zee = paths.Wenu + ['HLT_DoubleEle5_SW_L1RDQM']

    paths.GammaJet = ['HLT_Photon10_L1R_DQM',
                      'HLT_Photon15_TrackIso_L1R_DQM',
                      'HLT_Photon15_LooseEcalIso_L1R_DQM',
                      'HLT_Photon20_Cleaned_L1R_DQM',
                      'HLT_Photon25_LooseEcalIso_TrackIso_L1R_DQM']

    paths.DiGamma  = ['HLT_Photon10_L1R_DQM','HLT_DoublePhoton10_L1R_DQM']

    pathlumi = {
                 'HLT_Ele17_SW_TighterEleIdIsol_L1RDQM': '8e29',

                 'HLT_Ele10_LW_L1RDQM':'8e29',
                 'HLT_Ele15_SW_L1RDQM':'1e31',
                 'HLT_Ele10_LW_EleId_L1RDQM':'8e29',
                 'HLT_Ele15_SiStrip_L1RDQM':'8e29',
                 'HLT_DoubleEle5_SW_L1RDQM':'8e29',
                 'HLT_Photon10_L1R_DQM':'8e29',
                 'HLT_Photon15_TrackIso_L1R_DQM':'8e29',
                 'HLT_Photon15_LooseEcalIso_L1R_DQM':'8e29',
                 'HLT_Photon20_Cleaned_L1R_DQM':'8e29',
                 'HLT_DoublePhoton10_L1R_DQM':'8e29',
                 'HLT_Photon25_L1R_DQM':'1e31',
                 'HLT_Photon25_LooseEcalIso_TrackIso_L1R_DQM':'1e31'}

    lumiprocess = { '8e29':'HLT',
                    '1e31':'HLT'
                    }


    ##########################################################
    # produce generated paricles in acceptance               #
    ##########################################################

    genp = cms.EDFilter("PdgIdAndStatusCandViewSelector",
        status = cms.vint32(3),
        src = cms.InputTag("genParticles"),
        pdgId = cms.vint32(11)  # replaced in loop
    )

    fiducial = cms.EDFilter("EtaPtMinCandViewSelector",
        src = cms.InputTag("genp"),
        etaMin = cms.double(-2.5),  # to be replaced in loop ?
        etaMax = cms.double(2.5),   # to be replaced in loop ?
        ptMin = cms.double(2.0)     # to be replaced in loop ?
    )

    ##########################################################
    # loop over samples to create modules and sequence       #
    ##########################################################

    tmp = cms.SequencePlaceholder("tmp")
    egammaSelectors = cms.Sequence(tmp) # no empty sequences allowed, start with dummy
    egammaValidators= cms.Sequence(tmp) # same

    #loop over samples
    for samplenum in range(len(samples.names)):

        # clone genparticles and select correct type
        genpartname = "genpart"+samples.names[samplenum]
        globals()[genpartname] = genp.clone()
        setattr(globals()[genpartname],"pdgId",cms.vint32(samples.pdgid[samplenum]) ) # set pdgId
        egammaSelectors *= globals()[genpartname]                            # add to sequence

        # clone generator fiducial region
        fiducialname = "fiducial"+samples.names[samplenum]
        globals()[fiducialname] = fiducial.clone()
        setattr(globals()[fiducialname],"src",cms.InputTag(genpartname) ) # set input collection
        egammaSelectors *= globals()[fiducialname]               # add to sequence

        # loop over triggers for each sample
        for trig in getattr(paths,samples.names[samplenum]):
            trigname = trig + samples.names[samplenum] 
            #import appropriate config snippet
            filename = "HLTriggerOffline.Egamma."+trig+"_cfi"
            trigdef =__import__( filename )
            import sys
            globals()[trigname] = getattr(sys.modules[filename],trig).clone()    # clone imported config
            setattr(globals()[trigname],"cutcollection",cms.InputTag(fiducialname))        # set preselacted generator collection
            setattr(globals()[trigname],"cutnum",cms.int32( samples.num[samplenum]  )) # cut value for preselection
            setattr(globals()[trigname],"pdgGen",cms.int32( samples.pdgid[samplenum])) #correct pdgId for MC matching
            getattr(globals()[trigname],'triggerobject').setProcessName( lumiprocess[pathlumi[trig]] )         #set proper process name
            for filterpset in getattr(globals()[trigname],'filters'):
                getattr(filterpset,'HLTCollectionLabels').setProcessName( lumiprocess[pathlumi[trig]] )
                for isocollections in getattr(filterpset,'IsoCollections'):
                    isocollections.setProcessName( lumiprocess[pathlumi[trig]])

            egammaValidators *= globals()[trigname]                      # add to sequence


    egammaSelectors.remove(tmp)  # remove the initial dummy
    egammaValidators.remove(tmp)

    # selectors go into separate "prevalidation" sequence
    egammaValidationSequence   = cms.Sequence( egammaValidators )
    egammaValidationSequenceFS = cms.Sequence( egammaValidators )

else:
    #----------------------------------------------------------------------
    # new method
    #----------------------------------------------------------------------

    import sys, os

    # prefix for printouts
    msgPrefix = "[" + os.path.basename(__file__) + "]"

    import HLTriggerOffline.Egamma.EgammaHLTValidationUtils as EgammaHLTValidationUtils


    # maps from Egamma HLT path category to number of type and number of generated
    # particles required for the histogramming
    configData = {
        "singleElectron": { "genPid" : 11, "numGenerated" : 1,},
        "doubleElectron": { "genPid" : 11, "numGenerated" : 2 },
        "singlePhoton":   { "genPid" : 22, "numGenerated" : 1 },
        "doublePhoton":   { "genPid" : 22, "numGenerated" : 2 },
        }

    #----------------------------------------
    # generate generator level selection modules
    #
    # note that this is common between full and
    # fast simulation
    #----------------------------------------
    egammaSelectors = []

    for hltPathCategory, thisCategoryData in configData.iteritems():
        # all paths in the current category share the same
        # generator level requirement
        #
        # add a sequence for this generator level requirement

        generatorRequirementSequence = EgammaHLTValidationUtils.makeGeneratedParticleAndFiducialVolumeFilter(None,
                                                                                                             thisCategoryData['genPid'],
                                                                                                             thisCategoryData['numGenerated'])

        # dirty hack: get all modules of this sequence and add them
        # to globals() (which is not the same as calling globals() in makeGeneratedParticleAndFiducialVolumeFilter)
        # so that they will be added to the process
        for module in EgammaHLTValidationUtils.getModulesOfSequence(generatorRequirementSequence):
            globals()[module.label_()] = module

        # avoid that this variable is added to the process object when importing this _cff
        # (otherwise the last filter will appear with module name 'module' instead
        # of the name given by us...)
        del module

        egammaSelectors.append(generatorRequirementSequence)
        
    #----------------------------------------
    # compose the DQM anlyser paths
    #----------------------------------------

    egammaValidators = []
    egammaValidatorsFS = []

    for isFastSim, validators in (
        (False, egammaValidators),
        (True,  egammaValidatorsFS),
        ):

        #--------------------
        # a 'reference' process to take (and analyze) the HLT menu from
        #--------------------
        refProcess = cms.Process("REF")

        if isFastSim:
            refProcess.load("FastSimulation.Configuration.HLT_GRun_cff")
        else:
            refProcess.load("HLTrigger.Configuration.HLT_GRun_cff")

        #--------------------

        pathsByCategory = EgammaHLTValidationUtils.findEgammaPaths(refProcess)

        for hltPathCategory, thisCategoryData in configData.iteritems():

            # get the HLT path objects for this category
            paths = pathsByCategory[hltPathCategory]

            # fix: if there are no paths for some reason,
            # provide some dummy objects which we can delete
            # after the loop over the paths 
            path = None
            dqmModule = None

            for path in paths:

                # name of the HLT path
                pathName = path.label_()

                # we currently exclude a few 'problematic' paths (for which we
                # don't have a full recipe how to produce a monitoring path
                # for them).
                #
                # we exclude paths which contain EDFilters which we don't know
                # how to handle in the DQM modules
                moduleCXXtypes = EgammaHLTValidationUtils.getCXXTypesOfPath(refProcess,path)
                # print >> sys.stderr,"module types:", moduleCXXtypes

                hasProblematicType = False

                for problematicType in [
                    # this list was collected empirically
                    'HLTEgammaTriggerFilterObjectWrapper', 
                    'EgammaHLTPhotonTrackIsolationProducersRegional',
                    ]:

                    if problematicType in moduleCXXtypes:
                        ## print >> sys.stderr,msgPrefix, "SKIPPING PATH",pathName,"BECAUSE DON'T KNOW HOW TO HANDLE A MODULE WITH C++ TYPE",problematicType
                        hasProblematicType = True
                        break

                if hasProblematicType:
                    continue

                ## print >> sys.stderr,msgPrefix, "adding E/gamma HLT dqm module for path",pathName

                dqmModuleName = pathName
                if isFastSim:
                    dqmModuleName = dqmModuleName + "FastSim"

                dqmModuleName = dqmModuleName + "_DQM"

                dqmModule = EgammaHLTValidationUtils.EgammaDQMModuleMaker(refProcess, pathName,
                                                                          thisCategoryData['genPid'],        # type of generated particle
                                                                          thisCategoryData['numGenerated']   # number of generated particles
                                                                          ).getResult()

                # add the module to the process object
                globals()[dqmModuleName] = dqmModule

                # and to the sequence
                validators.append(dqmModule)

            # end of loop over paths

            # if we don't do the following deletes, loading this configuration
            # will pick these variables up and add it to the process object...
            del path
            del dqmModule

        # end of loop over analysis types (single electron, ...)

        #--------------------
        # we don't need the MC HLT Menu path any more
        del refProcess

    # end of loop over full/fast sim

    #--------------------
    
    # convert from list to sequence ('concatenate' them using '*')
    import operator

    egammaSelectors = cms.Sequence(reduce(operator.mul, egammaSelectors))
    
    # selectors go into separate "prevalidation" sequence
    egammaValidationSequence   = cms.Sequence(reduce(operator.mul, egammaValidators))
    egammaValidationSequenceFS = cms.Sequence(reduce(operator.mul, egammaValidatorsFS))


    #--------------------
