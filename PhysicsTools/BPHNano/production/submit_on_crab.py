#!/usr/bin/env python

import os
from argparse import ArgumentParser
from fnmatch import fnmatch
import yaml

import re
import datetime

from schema import Schema, And, Or, Optional, SchemaError

import CRABClient

from CRABAPI.RawCommand import crabCommand

from CRABClient.ClientExceptions import ClientException
from http.client import HTTPException 

from CRABClient.UserUtilities import config
from multiprocessing import Process


production_tag = datetime.date.today().strftime('%Y%b%d')


def parse_args():
    parser = ArgumentParser(description="A multicrab submission script")
    parser.add_argument('-y', '--yaml', default = 'test_samples.yml', help = 'File with dataset descriptions')
    parser.add_argument('-c', '--cmd', default='submit', choices = ['submit', 'status'], help= 'Crab command')
    parser.add_argument('-f', '--filter', default='*', help = 'filter samples, POSIX regular expressions allowed') 
    parser.add_argument('-w', '--workarea', default='BPHNANO_%s' % production_tag, help = 'Crab working area name')
    parser.add_argument('-o', '--outputdir', default= '/store/group/cmst3/group/bpark/gmelachr/', help='LFN Output high-level directory: the LFN will be saved in outputdir+workarea ')
    parser.add_argument('-s', '--site', default='T2_CH_CERN', help='T2 or T3 cite where user has access. To be checked with crab checkout')
    parser.add_argument('-t', '--tag', default=production_tag, help='Production Tag extra')
    parser.add_argument('-p', '--psetcfg', default="../test/run_bphNano_cfg.py", help='Plugin configuration file')
    parser.add_argument('-e', '--extra', nargs='*', default=list(),  help='Optional extra input files')
    parser.add_argument('-tt', '--test', action='store_true', help='Flag a test job')
    return parser.parse_args()
    
def submit(config):
    try:
        crabCommand('-dev submit', config = config)
    except HTTPException as hte:
        print("Failed submitting task: %s" % (hte.headers))
    except ClientException as cle:
        print("Failed submitting task: %s" % (cle))

def status(directory):
    try:
        crabCommand('status', dir=directory)
    except HTTPException as hte:
        print("Failed submitting task: %s" % (hte.headers))
    except ClientException as cle:
        print("Failed submitting task: %s" % (cle))


expected_schema = Schema({
    "common": {
        "data": {
            "lumimask": And(str, error="lumimask should be a string"),
            "splitting": And(int, error="splitting should be an integer"),
            "globaltag": And(str, error="globaltag should be a string"),
        },
        "mc": {
            "splitting": And(int, error="splitting should be an integer"),
            "globaltag": And(str, error="globaltag should be a string"),
        },
    },
    "samples": And(dict, error="samples should be a dict with keys dataset (str), isMC (bool). Optional keys: globaltag (str), parts (list(int))")
    }
    )

samples_schema = Schema({
    "dataset": And(str, error="dataset should be a string"),
    "isMC": And(bool, error="isMC should be a boolean"),
    Optional("decay") : And(str, error="decay to reconstruct"), 
    Optional("goldenjson") : And(str, error="golden json file path should be a string"),
    Optional("globaltag") : And(str, error="globaltag should be a string"),
    Optional("parts"): [And(int, error="parts should be a list of integers")]
})


def validate_yaml(data):
    try:
       expected_schema.validate(data)
       for name, content in data["samples"].items():
           samples_schema.validate(content)
       print("YAML structure is valid.")
    except SchemaError as e:
       print("YAML structure is invalid:", e)
       import sys
       sys.exit(1)
  

if __name__ == '__main__':

    args = parse_args()
    with open(args.yaml, "r") as f:
        samples = yaml.safe_load(f) # Parse YAML file
    validate_yaml(samples)
  
    if args.cmd == "submit":
        print("")
        print(f"Submit Crab jobs for {args.yaml} with filter {args.filter} applied")
        
        common_config = samples['common'] if 'common' in samples else {'data' : {}, 'mc' : {}}
        # loop over samples
        for sample, sample_info in samples['samples'].items():
            # Given we have repeated datasets check for different parts
    
            config_ = config()
    
            config_.General.transferOutputs = True
            config_.General.transferLogs = True
            config_.General.workArea = args.workarea

            config_.Data.publication = False
            config_.Data.outLFNDirBase = args.outputdir + '/'+ config_.General.workArea
            config_.Data.inputDBS = 'global'

            config_.JobType.pluginName = 'Analysis'
            config_.JobType.psetName = args.psetcfg
            config_.JobType.maxJobRuntimeMin = 2700 #can not use with Automatic 
            config_.JobType.allowUndistributedCMSSW = True
            config_.JobType.inputFiles = args.extra

            config_.Site.storageSite = args.site

        
            parts = sample_info['parts'] if 'parts' in sample_info else [None]
            for part in parts:
                name = sample % part if part is not None else sample
            
                # filter names according to what we need
                if not fnmatch(name, args.filter): continue
                print(name)
                config_.Data.outLFNDirBase = args.outputdir + config_.General.workArea
                config_.General.workArea = args.workarea + "_" + name
        
                config_.Data.inputDataset = sample_info['dataset'] % part \
                                         if part is not None else \
                                                  sample_info['dataset']
               
                data_type = 'mc' if sample_info['isMC'] else 'data'

                if sample_info['isMC']: config_.Data.splitting = 'FileBased'
                else: config_.Data.splitting = 'LumiBased'

                if sample_info['isMC']:
                    config_.Data.lumiMask = ''                    
                else:
                    config_.Data.lumiMask = sample_info.get('lumimask', None)

                config_.Data.unitsPerJob = common_config[data_type].get('splitting', None)

                globaltag = sample_info.get('globaltag', "auto:run3_data")
                if globaltag == "auto:run3_data":
                    globaltag = common_config[data_type].get('globaltag', "auto:run3_data")

                decay = sample_info.get('decay', 'all')
     
                maxevents = -1
       
                config_.JobType.pyCfgParams = [
                    'isMC=%s' % sample_info['isMC'], 'reportEvery=1000',
                    'tag=%s' % production_tag,
                    'globalTag=%s' % globaltag,
                    'decay=%s' % decay,
                    'maxEvents=%s' % maxevents,
                 ]
            
                if args.test:
                   config_.Data.totalUnits = 10

                config_.General.requestName = name + "_" + production_tag
                config_.JobType.outputFiles = ['_'.join(['bph_nano', production_tag, 'mc' if sample_info['isMC'] else 'data', decay])+'.root']
 

                print(f"Submit Crab job for {name}")
                print(config_)   
                submit(config_)
    elif args.cmd == "status":
        print(f"Getting crab status for {args.dir}")
        status(args.dir)
    else:
        print(f"Invalid Crab command : {args.cmd}")
    

