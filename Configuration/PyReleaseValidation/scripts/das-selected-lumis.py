#!/bin/env python
from __future__ import print_function
import json , sys
input_range = []
output_files_list = []
lumi_ranges  = sys.argv[1].split(':')
for lumi_range in lumi_ranges:
  input_range.append(tuple(lumi_range.split(',')))

jdata = sys.stdin.read()
try:
  lumi_data = json.loads(jdata) 
except:
  print(jdata)
  exit (1)
lumi_data = lumi_data['data']

def match_in(sub_list,lumi_list):
  for i in range(int(sub_list[0]),int(sub_list[1])+1):
    if i >= int(lumi_list[0]) and i <= int(lumi_list[1]): return True
  return False

def check_lumi_ranges(given_lumi_list , sub_range):
  for lumi_r in given_lumi_list:
    if match_in(sub_range, lumi_r):
      return True 
  return False

def process_lumi(data):
  for lumi_info in data:
    if isinstance(lumi_info['lumi'], list):
      lumi_nums = lumi_info['lumi'][0]['number']
      lumi_file = lumi_info['file'][0]['name']
    else:
      lumi_nums = lumi_info['lumi']['number']
      lumi_file = lumi_info['file']['name']
    if not isinstance(lumi_nums[0], list): lumi_rang = [ [n,n] for n in lumi_nums ]
    else: lumi_rang = lumi_nums
    for sub_list in lumi_rang:
      if check_lumi_ranges(input_range,tuple(sub_list)):
        output_files_list.append(lumi_file)
        break
  for out_file_name in output_files_list:
    print(out_file_name)

#Get file names for desired lumi ranges
process_lumi(lumi_data)

