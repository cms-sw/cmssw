#!/usr/bin/env python3
import sys
argv = sys.argv
sys.argv = argv[:1]

import os
import json
import yaml
import copy
import ROOT
import argparse

from Alignment.OfflineValidation.TkAlMap import TkAlMap

def parser():
    sys.argv = argv
    parser = argparse.ArgumentParser(description = "run the python plots for the AllInOneTool validations", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config", metavar='config', type=str, action="store", help="GCP AllInOneTool config (json/yaml format)")
    parser.add_argument("-b", "--batch", action = "store_true", help ="Batch mode")

    sys.argv.append( '-b' )
    ROOT.gROOT.SetBatch()
    return parser.parse_args()

def TkAlMap_plots(config):
    root_file = str(config['output']) + '/GCPtree.root'
    destination = str(config['output'])
    plot_dir = destination + '/TkAlMaps'

    com_vs_ref = config['alignments']['comp']['title'] + ' vs ' + config['alignments']['ref']['title']
    iov_vs_iov = str(config['validation']['IOVcomp']) + ' - ' + str(config['validation']['IOVref'])
    title = '#splitline{'+com_vs_ref+'}{'+iov_vs_iov+'}'

    if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
    
    plot_png = False
    if 'plotPng' in config['validation']['GCP']:
        plot_png = config['validation']['GCP']['plotPng'] 

    palette = 2 #1 is rainbow palette, 2 is diverging color palette (blue to red)

    var_list = ['dr', 'dx', 'dy', 'dz', 'rdphi', 'dphi'] #, 'dalpha', 'dbeta', 'dgamma', 'du', 'dv', 'dw', 'da', 'db', 'dg']
    TkMap_pixel         = TkAlMap(
        'dummy_var', title, root_file, 
        use_default_range=True, two_sigma_cap=False, 
        tracker='pixel', 
        palette=palette, check_tracker=True
    )
    TkMap_strips        = TkAlMap(
        'dummy_var', title, root_file, 
        use_default_range=True, two_sigma_cap=False, 
        tracker='strips', 
        palette=palette, check_tracker=True
    )

    for var in var_list:
        print(' --- Creating maps for variable: '+var)
        var_range = [None, None]
        if var+'_min' in config['validation']['GCP']: var_range[0] = float(config['validation']['GCP'][var+'_min'])
        if var+'_max' in config['validation']['GCP']: var_range[1] = float(config['validation']['GCP'][var+'_max'])

        TkMap_pixel.set_var(var, var_range=var_range)
        TkMap_pixel.analyse()
        TkMap_pixel.save(out_dir=plot_dir)
        if plot_png: TkMap_pixel.save(out_dir=plot_dir, extension='png')
        TkMap_pixel.plot_variable_distribution(out_dir=plot_dir)

        TkMap_strips.set_var(var)
        TkMap_strips.analyse()
        TkMap_strips.save(out_dir=plot_dir)
        if plot_png: TkMap_strips.save(out_dir=plot_dir, extension='png')
        TkMap_strips.plot_variable_distribution(out_dir=plot_dir)

if __name__ == '__main__':

    args = parser()

    with open(args.config, "r") as configFile:
        if args.config.split(".")[-1] == "json":
            config = json.load(configFile)
    
        elif args.config.split(".")[-1] == "yaml":
            config = yaml.load(configFile, Loader=yaml.Loader)
    
        else:
            raise Exception("Unknown config extension '{}'. Please use json/yaml format!".format(args.config.split(".")[-1]))

    print(' ----- TkAlMaps -----') 
    TkAlMap_plots(config) 

