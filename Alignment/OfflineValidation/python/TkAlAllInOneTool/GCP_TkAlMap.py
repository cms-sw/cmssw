import sys
import copy
import time
import argparse
from Alignment.OfflineValidation.TkAlMap import TkAlMap

def parser():
    parser = argparse.ArgumentParser(description = "Parse AllInOne config to TkAlMap. Only ment for condor jobs.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config", metavar='config', type=str, action="store", help="Global AllInOneTool config (json/yaml format)")
    return parser.parse_args()

def main():
    print('*---------------------------------------*')
    print('|             GCP TkAlMap               |')
    print('*---------------------------------------*')

    # Get arguments 
    args = parser()

    # Get configuration 
    with open(args.config, 'r') as configFile:
        if args.config.split('.')[-1] == 'json':
            config = json.load(configFile)

        elif args.config.split('.')[-1] == 'yaml':
            config = yaml.load(configFile, Loader=yaml.Loader)

        else:
            raise Exception('Unknown config extension "{}". Please use json/yaml format!'.format(args.config.split('.')[-1]))

    # Init variables
    var_list = ['dr', 'dx', 'dy', 'dz', 'rdphi', 'dphi', 'dalpha', 'dbeta', 'dgamma', 'du', 'dv', 'dw', 'da', 'db', 'dg']
    var_ranges = {}
    for var in var_list:
        var_ranges[var] = [None, None]

    # Digest
    al_ref  = config['alignments']['ref']['title']
    al_comp = config['alignments']['comp']['title']
    in_file = config['input']
    out_dir = config['output']
    phase = 1
    auto_tk = True
    if 'detector_phase' in config['validation']['GCP']:
        phase = int(config['validation']['GCP']['detector_phase'])
        auto_tk = False
    palette = 2
    if 'map_palette' in config['validation']['GCP']: palette = int(config['validation']['GCP']['map_palette'])
    save_png = False
    if 'save_png' in config['validation']['GCP']: save_png = config['validation']['GCP']['save_png']
    save_pdf = True
    if 'save_pdf' in config['validation']['GCP']: save_pdf = config['validation']['GCP']['save_pdf']
    
    print('Current setup:')
    print(' - reference alingment              : '+al_ref)
    print(' - compared alingment               : '+al_comp)
    print(' - tracker version                  : phase '+str(phase))
    print(' - auto detect tracker version      : '+str(auto_tk))
    print(' - color palette                    : '+str(palette))
    print(' - input root file                  : '+in_file)
    print(' - output directory                 : '+out_dir)
    print(' - saving as png                    : '+str(save_png))
    print(' - saving as pdf                    : '+str(save_pdf))
    print('')
    print('Active plots:')
    print(' - plot 4 sigma capped values       : '+str(do_4scap))
    print(' - plot default range capped values : '+str(do_drange))
    print(' - plot un-capped values            : '+str(do_frange))
    print(' - plot full detector               : '+str(do_full))
    print(' - plot pixel detector              : '+str(do_pixel))
    print(' - plot strips detector             : '+str(do_strips))
    print('')
    print('Changed default ranges:')
    for var in var_ranges:
        if var_ranges[var][0] is None and var_ranges[var][1] is None: continue
        prt_srt = ' - '+var+'\t: [ '
        if var_ranges[var][0] is None: prt_srt += 'default'
        else: prt_srt += str(var_ranges[var][0])
        prt_srt += '\t, '
        if var_ranges[var][1] is None: prt_srt += 'default'
        else: prt_srt += str(var_ranges[var][1])
        prt_srt += '\t]'
        print(prt_srt)

    print('Loading maps')
    TkMap_full          = TkAlMap('test', title, in_file, use_default_range=False, two_sigma_cap=False, GEO_file=geometry_file, tracker='full',   palette=palette, check_tracker=auto_tk)
    TkMap_pixel         = TkAlMap('test', title, in_file, use_default_range=False, two_sigma_cap=False, GEO_file=geometry_file, tracker='pixel',  palette=palette, check_tracker=auto_tk)
    TkMap_strips        = TkAlMap('test', title, in_file, use_default_range=False, two_sigma_cap=False, GEO_file=geometry_file, tracker='strips', palette=palette, check_tracker=auto_tk)
    TkMap_cap_full      = TkAlMap('test', title, in_file, use_default_range=False, two_sigma_cap=True,  GEO_file=geometry_file, tracker='full',   palette=palette, check_tracker=auto_tk)
    TkMap_cap_pixel     = TkAlMap('test', title, in_file, use_default_range=False, two_sigma_cap=True,  GEO_file=geometry_file, tracker='pixel',  palette=palette, check_tracker=auto_tk)
    TkMap_cap_strips    = TkAlMap('test', title, in_file, use_default_range=False, two_sigma_cap=True,  GEO_file=geometry_file, tracker='strips', palette=palette, check_tracker=auto_tk)
    TkMap_drange_full   = TkAlMap('test', title, in_file, use_default_range=True,  two_sigma_cap=False, GEO_file=geometry_file, tracker='full',   palette=palette, check_tracker=auto_tk)
    TkMap_drange_pixel  = TkAlMap('test', title, in_file, use_default_range=True,  two_sigma_cap=False, GEO_file=geometry_file, tracker='pixel',  palette=palette, check_tracker=auto_tk)
    TkMap_drange_strips = TkAlMap('test', title, in_file, use_default_range=True,  two_sigma_cap=False, GEO_file=geometry_file, tracker='strips', palette=palette, check_tracker=auto_tk)
