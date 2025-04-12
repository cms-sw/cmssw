import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from ROOT import TFile, TH1

parser = argparse.ArgumentParser()
parser.add_argument("-b", type=int, dest="bits", nargs='+', default=[], help="bit to be studied for barycenter")
parser.add_argument("-w", type=int, dest="widths", nargs='+', default=[], help="bit to be studied for width")
parser.add_argument("-a", type=int, dest="avgCharges", nargs='+', default=[], help="bit to be studied for avgCharge")
parser.add_argument("-o", dest="output", default='output', help="directory name where inputs are")
parser.add_argument("-v", dest="version", default='', help="which version you want to compare")
parser.add_argument("-e", action='store_true', dest="events", default=False, help="want to see # of events")

options = parser.parse_args()
bits = options.bits
widths = options.widths
avgCharges = options.avgCharges
output = options.output

x = np.array(bits)
x = np.sort(x)
widths = np.array(widths)
avgCharge = np.array(avgCharges)

colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']

def readfile(input_file):

  with open(input_file, 'r') as f:
     lines = f.readlines()
  return lines

def update_list(dirname, bary_bit, chrg_bit, rawtype, sizes, yvals, texts, ver, events=0):

  input = f'/scratch/nandan/{dirname}_barycenter_{bary_bit}bit_width_8bit_avgCharge_{chrg_bit}bit/'
  input_file = os.path.join(input, 'size.log')
  lines = readfile(input_file)

  for idx, line in enumerate(lines):

     if 'SiStripApproximateClusterCollection_hltSiStripClusters2ApproxClusters__reHLT' in line:
         sizes[ver].append(float(line.split(' ')[-1]))

  input_file = os.path.join(input, 'object.log' if not events else 'object_study.root')
  if not events:
   lines = readfile(input_file)
   for idx, line in enumerate(lines):
      if f'not matched {obj}' in line and  f'{rawtype} ' in line:
            val = float(line.split(f'in {rawtype} ')[-1].split('%')[0])
            yvals[ver].append(val)
  else:
    f = TFile(input_file, 'r')
    yvals[ver].append(f.Get(f'{rawtype}_trk_cutflow_z4').GetBinContent(1,1))

  #print(sizes)
  #print(yvals)
  texts[ver].append((f'{bary_bit}', f'{chrg_bit}'))

def draw(x_vals, y_vals, texts, ytitle, obj, rawtype, filename=''):

  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot(111)
  for idx, key in enumerate(texts.keys()):
    plt.scatter(x_vals[key], y_vals[key], color=colors[idx], label=key)
    for i, text in enumerate(texts[key]):
      ax.text(x_vals[key][i], y_vals[key][i], text)#, color=colors[idx])
  plt.title(f'size vs {obj}', fontsize=20)
  plt.xlabel('size of approx cluster in Byte', fontsize=20)
  plt.ylabel(ytitle, fontsize=20)
  plt.legend()
  ax.grid(True)
  plt.savefig(f'singleplot_{obj}_{rawtype}.png' if filename=='' else f'{filename}.png')
  plt.close('all')

def build_array(obj, rawtype):

  texts = {}
  sizes = {}
  yvals = {}

  compare = 'cutflow'

  texts['rawp'] = []
  yvals['rawp'] = []
  sizes['rawp'] = []

  for avgCharge in avgCharges:
    for bit in x:
      update_list(output, bit, avgCharge, rawtype, sizes, yvals, texts, 'rawp', options.events)
  
  texts['HI_rawp'] = []
  yvals['HI_rawp'] = []
  sizes['HI_rawp'] = []
  
  update_list('HI_wchargecut', 16, 8, rawtype, sizes, yvals, texts, 'HI_rawp', options.events)
  if options.version == 'v2':
    texts['v2'] = []
    yvals['v2'] = []
    sizes['v2'] = []
    update_list('HI_wchargecut_v2', 15, 8, rawtype, sizes, yvals, texts, 'v2')
    update_list('HI_wchargecut_v2', 15, 5, rawtype, sizes, yvals, texts, 'v2')
    update_list('HI_wchargecut_v2', 15, 4, rawtype, sizes, yvals, texts, 'v2')
    #update_list('HI_wchargecut_v2', 15, 7, rawtype, sizes, yvals, texts, 'v2')
    update_list('HI_wchargecut_v2', 15, 6, rawtype, sizes, yvals, texts, 'v2')
    '''update_list('HI_wchargecut_v2', 14, 5, rawtype, sizes, yvals, texts, 'v2')
    update_list('HI_wchargecut_v2', 14, 4, rawtype, sizes, yvals, texts, 'v2')
    update_list('HI_wchargecut_v2', 14, 8, rawtype, sizes, yvals, texts, 'v2')
    update_list('HI_wchargecut_v2', 14, 7, rawtype, sizes, yvals, texts, 'v2')
    update_list('HI_wchargecut_v2', 14, 6, rawtype, sizes, yvals, texts, 'v2')
    update_list('HI_wchargecut_v2', 13, 8, rawtype, sizes, yvals, texts, 'v2')'''
  elif options.version == 'v1':
    texts['v1'] = []
    yvals['v1'] = []
    sizes['v1'] = []
    update_list('remove_beginindices_v1_compression_LZMA', 14, 8, rawtype, sizes, yvals, texts, 'v1')
    update_list('remove_beginindices_v1_compression_LZMA', 14, 5, rawtype, sizes, yvals, texts, 'v1')
    update_list('remove_beginindices_v1_compression_LZMA', 14, 6, rawtype, sizes, yvals, texts, 'v1')
    update_list('remove_beginindices_v1_compression_LZMA', 14, 7, rawtype, sizes, yvals, texts, 'v1')
    update_list('remove_beginindices_v1_compression_LZMA', 14, 4, rawtype, sizes, yvals, texts, 'v1')
    update_list('remove_beginindices_v1_compression_LZMA', 14, 3, rawtype, sizes, yvals, texts, 'v1')
  elif options.version == 'v1.1':
    texts['v1.1'] = []
    yvals['v1.1'] = []
    sizes['v1.1'] = []
    update_list('HI_wchargecut_v1p1', 14, 8, rawtype, sizes, yvals, texts, 'v1.1')
    update_list('HI_wchargecut_v1p1', 14, 5, rawtype, sizes, yvals, texts, 'v1.1')
    update_list('HI_wchargecut_v1p1', 14, 6, rawtype, sizes, yvals, texts, 'v1.1')
    update_list('HI_wchargecut_v1p1', 14, 7, rawtype, sizes, yvals, texts, 'v1.1')
    update_list('HI_wchargecut_v1p1', 13, 7, rawtype, sizes, yvals, texts, 'v1.1')
    update_list('HI_wchargecut_v1p1', 13, 5, rawtype, sizes, yvals, texts, 'v1.1')
    update_list('HI_wchargecut_v1p1', 13, 6, rawtype, sizes, yvals, texts, 'v1.1')
  
  draw(sizes, yvals, texts, f'unmatched {obj} in %', obj, rawtype)

if not options.events:
  for raw in ['raw', 'rawp']:
     for obj in ['tracks_lowpt', 'tracks_highpt']:#, 'jet']:
        build_array(obj, raw)
else:

  texts = {}
  sizes = {}
  yvals = {}

  texts['wchargecut_rawp'] = []
  yvals['wchargecut_rawp'] = []
  sizes['wchargecut_rawp'] = []

  update_list('test_compression_LZMA', 16, 8, 'rawp', sizes, yvals, texts, 'wchargecut_rawp', options.events)

  texts['wochargecut_rawp'] = []
  yvals['wochargecut_rawp'] = []
  sizes['wochargecut_rawp'] = []

  update_list('test_wochargecut_compression_LZMA', 16, 8, 'rawp', sizes, yvals, texts, 'wochargecut_rawp', options.events)

  texts['wchargecut_HI_rawp'] = []
  yvals['wchargecut_HI_rawp'] = []
  sizes['wchargecut_HI_rawp'] = []

  update_list('default_10_compression_LZMA', 16, 8, 'rawp', sizes, yvals, texts, 'wchargecut_HI_rawp', options.events)

  texts['wochargecut_HI_rawp'] = []
  yvals['wochargecut_HI_rawp'] = []
  sizes['wochargecut_HI_rawp'] = []

  update_list('default_10_wochargcut_compression_LZMA', 16, 8, 'rawp', sizes, yvals, texts, 'wochargecut_HI_rawp', options.events)

  draw(sizes, yvals, texts, '# of tracks', 'tracks', 'rawp', 'chargecut')
