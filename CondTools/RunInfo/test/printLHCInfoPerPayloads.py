#!/usr/bin/env python3

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

import os
import sys

process = cms.Process('dumpLHCInfoPerPyalods')

TEST_DIR = os.environ['CMSSW_BASE'] + "/src/CondTools/RunInfo/test"


options = VarParsing.VarParsing()
supported_records = {"LHCInfoPerLS", "LHCInfoPerFill"}
options.register( 'record'
                , None #default value is None because this argument is required
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , f"the class to analyze, accepted values: {supported_records}"
                  )
options.register( 'db'
                , 'frontier://FrontierProd/CMS_CONDITIONS' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Connection string to the DB where payloads are going to be read from"
                  )
options.register( 'tag'
                , None #default value is None because this argument is required
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

if options.record is None:
  print(f"Please specify the record name argument (accepted values: {supported_records})", file=sys.stderr)
  exit(1)
if options.record not in supported_records:
  print(f"Provided record name '{options.record}' is not supported (accepted values: {supported_records})", file=sys.stderr)
  exit(1)

if options.tag is None:
  print(f"Please specify the tag by adding to the command: tag=<tag to be printed>", file=sys.stderr)
  exit(1)

for line in map(str.rstrip, sys.stdin):
    for iov in line.split():
        if options.record == "LHCInfoPerLS":
            os.system(f"cmsRun {TEST_DIR}/LHCInfoPerLSAnalyzer_cfg.py tag={options.tag} \
            db={options.db} csv={options.csv} header={options.header} separator={options.separator} timestamp={iov}")
        elif options.record == "LHCInfoPerFill":
            os.system(f"cmsRun {TEST_DIR}/LHCInfoPerFillAnalyzer_cfg.py tag={options.tag} db={options.db} timestamp={iov}")

        options.header = False
