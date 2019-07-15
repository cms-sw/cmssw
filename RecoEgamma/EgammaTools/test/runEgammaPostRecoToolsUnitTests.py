#!/usr/bin/env python

import subprocess
import os

def get_config_options(all_cmd_strs,cmd_str,opts,opt_vals):
    if opts == list():
        all_cmd_strs.append(cmd_str)
    else:
        opts_new = opts[1:]
        for val in opt_vals[opts[0]]:
            cmd_str_new = str(cmd_str)
            cmd_str_new+=" {}={}".format(opts[0],val)
            get_config_options(all_cmd_strs,cmd_str_new,opts_new,opt_vals)
    
def get_opt_val(options,opt_name):
    for opt in options:
        opt_split = opt.split("=")
        if opt_split[0] == opt_name: 
            return opt_split[1]
    return ""

def is_valid_option_combination(cmd_str):
    options = cmd_str.split()[2:]
    
    if get_opt_val(options,'applyVIDOnCorrectedEgamma') != get_opt_val(options,'applyEnergyCorrections'):
        return False

    if get_opt_val(options,'isMiniAOD')=='False' and get_opt_val(options,'applyVIDOnCorrectedEgamma')=='True':
        return False
    return True
    
def run_static_tests():
    """
    Runs the python of the config file only, checks for syntax errors
    Runs over all possible input parameters
    """
    opt_vals = {
        'isMiniAOD' : ["True","False"],
        'runVID' : ["True","False"],
        'runEnergyCorrections' : ["True","False"],
        'applyEnergyCorrections' : ["True","False"],
        'applyVIDOnCorrectedEgamma' : ["True","False"],
        'applyEPCombBug' : ["True","False"],
        'era' : ['2016-Legacy','2017-Nov17ReReco','2018-Prompt']
        }
    
    all_cmd_strs = []
#    config = "python src/RecoEgamma/EgammaTools/test/runEgammaPostRecoTools.py".format(os.environ['CMSSW_BASE'])
    config = "python RecoEgamma/EgammaTools/test/runEgammaPostRecoTools.py"
    get_config_options(all_cmd_strs,config,opt_vals.keys(),opt_vals)
    for cmd_str in all_cmd_strs:
#        print cmd_str
        if is_valid_option_combination(cmd_str):
            out,err=subprocess.Popen(cmd_str.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
        #        print out
            if err!="":
                print cmd_str
                print err
    



def main():
    run_static_tests()
    run_live_tests()
if __name__ == '__main__':
    main()
    
