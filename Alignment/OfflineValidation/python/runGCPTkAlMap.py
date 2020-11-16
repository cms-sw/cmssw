import sys
import copy
import time
from Alignment.OfflineValidation.TkAlMap import TkAlMap

'''
Script for plotting TkAlMaps
How to run:
    python runGCPTkAlMap.py -b inFile=<file_path> compAl=<c_alignment_name> refAl=<r_alignment_name> TkVersion=<phase> outDir=<out_dir> colPal=<col_int> defRanges=<range_str> TkautoVersion= <tk_version_bool> savePNG=<png_bool> savePDF=<pdf_bool> do4sigCap=<4sig_bool> doDefRange=<drange_bool> doFullRange=<frange_bool> doFull=<full_bool> doPixel=<pixel_bool> doStrips=<strips_bool>

Explanation:
 - Mandatory arguments:
    inFile=<file_path>                 path to root file containing geometry comparison tree "alignTree"
    compAl=<c_alignment_name>          name of alignment beeing compared (for title)
    refAl=<r_alignment_name>           name of reference alignment (for title)
 - Optional arguments:
    TkVersion=<phase>                  tracker version valid options: phase0, phase1
    outDir=<out_dir>                   directory where to store the images
    colPal=<col_int>                   color palette: 1 is rainbow palette, 2 is diverging color palette (blue to red)
    defRanges=<range_str>              string containing changes to default range in format "<var>_range=[<min>,<max>];<var2>_..." example: "dr_range=[-10,10];rdphi_range=[-2.02,120];"
    TkautoVersion=<tk_version_bool>    string boolean telling wheter or not to auto detect TkVersion (will override the TkVersion=<phase> selection)
    savePNG=<png_bool>                 string boolean to save or not save as png
    savePDF=<pdf_bool>                 string boolean to save or not save as pdf
    do4sigCap=<4sig_bool>              string boolean to plot 4sigma capped plots or not   
    doDefRange=<drange_bool>           string boolean to plot default range capped plots or not 
    doFullRange=<frange_bool>          string boolean to plot un-capped plots or not
    doFull=<full_bool>                 string boolean to plot full detector or not
    doPixel=<pixel_bool>               string boolean to plot separate pixel detector or not
    doStrips=<strips_bool>             string boolean to plot separate strips detector or not
'''


print('*---------------------------------------*')
print('|             GCP TkAlMap               |')
print('*---------------------------------------*')

#var_list = ['dr']
#var_list = ['dx', 'dy', 'dz']
var_list = ['dr', 'dx', 'dy', 'dz', 'rdphi', 'dphi', 'dalpha', 'dbeta', 'dgamma', 'du', 'dv', 'dw', 'da', 'db', 'dg']
var_ranges = {}
for var in var_list:
    var_ranges[var] = [None, None]

# Our own parser
print('Reading arguments')
arguments = sys.argv
al_ref = 'Reference Alignment'
al_comp = 'Compared Alignment'
out_dir = '.'
phase_str = ''
auto_tk_str = ''
palette_str = ''
range_str = ''

save_png  = False
save_pdf  = True
do_4scap  = False
do_drange = False
do_frange = True
do_full   = True
do_pixel  = False
do_strips = False

save_png_str  = ''
save_pdf_str  = ''
do_4scap_str  = ''
do_drange_str = ''
do_frange_str = ''
do_full_str   = ''
do_pixel_str  = ''
do_strips_str = ''

for arg in arguments:
    if 'inFile=' in arg      : in_file      = arg.replace('inFile=', '')
    if 'refAl=' in arg       : al_ref       = arg.replace('refAl=', '')
    if 'compAl=' in arg      : al_comp      = arg.replace('compAl=', '')
    if 'outDir=' in arg      : out_dir      = arg.replace('outDir=', '')
    if 'TkVersion='in arg    : phase_str    = arg.replace('TkVersion=', '')
    if 'TkautoVersion='in arg: auto_tk_str  = arg.replace('TkautoVersion=', '')
    if 'colPal='in arg       : palette_str  = arg.replace('colPal=', '')
    if 'defRanges=' in arg   : range_str    = arg.replace('defRanges=', '')
    # Limit outputs
    if 'savePNG=' in arg     : save_png_str    = arg.replace('savePNG=', '')
    if 'savePDF=' in arg     : save_pdf_str    = arg.replace('savePDF=', '')
    if 'do4sigCap=' in arg   : do_4scap_str    = arg.replace('do4sigCap=', '')
    if 'doDefRange=' in arg  : do_drange_str   = arg.replace('doDefRange=', '')
    if 'doFullRange=' in arg : do_frange_str   = arg.replace('doFullRange=', '')
    if 'doFull=' in arg      : do_full_str     = arg.replace('doFull=', '')
    if 'doPixel=' in arg     : do_pixel_str    = arg.replace('doPixel=', '')
    if 'doStrips=' in arg    : do_strips_str   = arg.replace('doStrips=', '')

# Digest arguments
phase = 1
title = al_comp + ' - ' + al_ref
auto_tk = True
if 'FALSE' in auto_tk_str.upper(): auto_tk = False
if 'PHASE0' in phase_str.upper() : phase = 0
geometry_file = 'TkAlMapDesign_phase1_cfg.py'
if phase == 1: geometry_file = 'TkAlMapDesign_phase0_cfg.py'
palette = 2
if '1' in palette_str: palette = 1

if 'TRUE' in save_png_str .upper(): save_png  = True 
if 'TRUE' in save_pdf_str .upper(): save_pdf  = True 
if 'TRUE' in do_4scap_str .upper(): do_4scap  = True 
if 'TRUE' in do_drange_str.upper(): do_drange = True 
if 'TRUE' in do_frange_str.upper(): do_frange = True 
if 'TRUE' in do_full_str  .upper(): do_full   = True 
if 'TRUE' in do_pixel_str .upper(): do_pixel  = True 
if 'TRUE' in do_strips_str.upper(): do_strips = True 

if 'FALSE' in save_png_str .upper(): save_png  = False 
if 'FALSE' in save_pdf_str .upper(): save_pdf  = False 
if 'FALSE' in do_4scap_str .upper(): do_4scap  = False 
if 'FALSE' in do_drange_str.upper(): do_drange = False 
if 'FALSE' in do_frange_str.upper(): do_frange = False 
if 'FALSE' in do_full_str  .upper(): do_full   = False 
if 'FALSE' in do_pixel_str .upper(): do_pixel  = False 
if 'FALSE' in do_strips_str.upper(): do_strips = False 


range_str_splt = range_str.split(';')
for var_range_str in range_str_splt:
    cur_var = var_range_str.split('=')[0]
    if cur_var == '': continue
    cur_range = eval(var_range_str.split('=')[1])
    for var in var_ranges:
        if var+'_range' == cur_var:
            if cur_range[0] != -99999: var_ranges[var][0] = cur_range[0]
            if cur_range[1] != -99999: var_ranges[var][1] = cur_range[1]
    #max_val = float(var_range_str.split('=')[1].split(',')[0].replace('[', '')) 
    #min_val = float(var_range_str.split('=')[1].split(',')[1].replace(']', '')) 

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
  

# Load maps for different configurations
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

ts_start = time.time()
for var in var_list:
    print('----- Evaluating variable: '+var)
    # Usual setup
    if do_frange:
        if do_full:
            tmp_full = TkMap_full
            tmp_full.set_var(var)
            tmp_full.analyse()  
            if save_pdf: tmp_full.save(out_dir=out_dir)  
            if save_png: tmp_full.save(out_dir=out_dir, extension='png')  
            tmp_full.plot_variable_distribution(out_dir=out_dir)

        if do_pixel:
            tmp_pixel = TkMap_pixel
            tmp_pixel.set_var(var)
            tmp_pixel.analyse()  
            if save_pdf: tmp_pixel.save(out_dir=out_dir)  
            if save_png: tmp_pixel.save(out_dir=out_dir, extension='png')  
            tmp_pixel.plot_variable_distribution(out_dir=out_dir)

        if do_strips:
            tmp_strips = TkMap_strips
            tmp_strips.set_var(var)
            tmp_strips.analyse()  
            if save_pdf: tmp_strips.save(out_dir=out_dir)  
            if save_png: tmp_strips.save(out_dir=out_dir, extension='png')  
            tmp_strips.plot_variable_distribution(out_dir=out_dir)

    # 4 sigma capping
    if do_4scap:
        if do_full:
            tmp_cap_full = TkMap_cap_full
            tmp_cap_full.set_var(var)
            tmp_cap_full.analyse()  
            if save_pdf: tmp_cap_full.save(out_dir=out_dir)  
            if save_png: tmp_cap_full.save(out_dir=out_dir, extension='png')  

        if do_pixel:
            tmp_cap_pixel = TkMap_cap_pixel
            tmp_cap_pixel.set_var(var)
            tmp_cap_pixel.analyse()  
            if save_pdf: tmp_cap_pixel.save(out_dir=out_dir)  
            if save_png: tmp_cap_pixel.save(out_dir=out_dir, extension='png')  

        if do_strips:
            tmp_cap_strips = TkMap_cap_strips
            tmp_cap_strips.set_var(var)
            tmp_cap_strips.analyse()  
            if save_pdf: tmp_cap_strips.save(out_dir=out_dir)
            if save_png: tmp_cap_strips.save(out_dir=out_dir, extension='png')  

    # default ranges
    if do_drange:
        if do_full:
            tmp_drange_full = TkMap_drange_full
            tmp_drange_full.set_var(var, var_ranges[var])
            tmp_drange_full.analyse()  
            if save_pdf: tmp_drange_full.save(out_dir=out_dir)  
            if save_png: tmp_drange_full.save(out_dir=out_dir, extension='png')  

        if do_pixel:
            tmp_drange_pixel = TkMap_drange_pixel
            tmp_drange_pixel.set_var(var, var_ranges[var])
            tmp_drange_pixel.analyse()  
            if save_pdf: tmp_drange_pixel.save(out_dir=out_dir)  
            if save_png: tmp_drange_pixel.save(out_dir=out_dir, extension='png')  

        if do_strips:
            tmp_drange_strips = TkMap_drange_strips
            tmp_drange_strips.set_var(var, var_ranges[var])
            tmp_drange_strips.analyse()  
            if save_pdf: tmp_drange_strips.save(out_dir=out_dir)
            if save_png: tmp_drange_strips.save(out_dir=out_dir, extension='png')  

TkMap_full.clean_up()         
TkMap_pixel.clean_up()        
TkMap_strips.clean_up()       
TkMap_cap_full.clean_up()     
TkMap_cap_pixel.clean_up()    
TkMap_cap_strips.clean_up()   
TkMap_drange_full.clean_up()  
TkMap_drange_pixel.clean_up() 
TkMap_drange_strips.clean_up()

print('TOOK: '+str(time.time()-ts_start)+' s') 
