import FWCore.ParameterSet.Config as cms
import os

## default location of high-PU filelist
path = os.getenv( "CMSSW_BASE" ) + "/src/GEMCode/SimMuL1/test/"
def_filelist = path + "filelist_minbias_61M_good.txt"

## helper function for pile-up
def addPileUp(process, pu = 140, filelist = def_filelist):
    ff = open(filelist, "r")
    pu_files = ff.read().split('\n')
    ff.close()
    pu_files = filter(lambda x: x.endswith('.root'),  pu_files)
    
    process.mix.input = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(pu)
        ),
        type = cms.string('poisson'),
        sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(*pu_files)
    )
    return process

## helper for files on dCache/EOS (LPC)
def useInputDir(process, suffix, onEOS = True):
    from GEMCode.SimMuL1.GEMCSCTriggerSamplesLib import files
    inputDir = files[suffix]
    theInputFiles = []
    for d in range(len(inputDir)):
        my_dir = inputDir[d]
        if not os.path.isdir(my_dir):
            print "ERROR: This is not a valid directory: ", my_dir
            if d==len(inputDir)-1:
                print "ERROR: No input files were selected"
                exit()
            continue
        print "Proceed to next directory"
        ls = os.listdir(my_dir)
        ## this works only if you pass the location on pnfs - FIXME for files staring with store/user/...
        theInputFiles.extend([my_dir[16:] + x for x in ls if x.endswith('root')])
                                                                                                        
    process.source.fileNames = cms.untracked.vstring(*theInputFiles)
    return process
