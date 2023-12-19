
#!python3

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

import os
import sys

process = cms.Process('LHCInfoPerAnalyzeMultiple')

TEST_DIR = os.environ['CMSSW_BASE'] + "/src/CondTools/RunInfo/test"

options = VarParsing.VarParsing()
options.register( 'record'
                , '' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "the class to analyze, accepted values LHCInfoPerLS, LHCInfoPerFill"
                  )
options.register( 'db'
                , 'frontier://FrontierProd/CMS_CONDITIONS' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Connection string to the DB where payloads are going to be read from"
                  )
options.register( 'tag'
                , 'LHCInfo_PopCon_test'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag to read from in source"
                  )
options.register( 'csv'
                , False
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.bool
                , "Weather or not to print the values in csv format"
                  )
options.register( 'header'
                , False
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.bool
                , "Weather or not to print header for the csv, works only in when csv=True"
                  )
options.register( 'separator'
                , ','
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "separator for the csv format, works only in when csv=True"
                  )
options.parseArguments()



for line in map(str.rstrip, sys.stdin):
    for iov in line.split():
        if options.record == "LHCInfoPerLS":
            # print(iov, end = ";")
            os.system(f"cmsRun {TEST_DIR}/LHCInfoPerLSAnalyzer_cfg.py tag={options.tag} \
            db={options.db} csv={options.csv} header={options.header} separator={options.separator} timestamp={iov}")
        elif options.record == "LHCInfoPerFill":
            os.system(f"cmsRun {TEST_DIR}/LHCInfoPerFillAnalyzer_cfg.py tag={options.tag} db={options.db} timestamp={iov}")

        options.header = False
