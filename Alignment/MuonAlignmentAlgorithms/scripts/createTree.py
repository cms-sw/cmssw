#! /usr/bin/env python

import re,os,sys
import optparse

# python 2.6 has json modue; <2.6 could use simplejson
try:
  import json
except ImportError:
  import simplejson as json

from mutypes import * 

import pprint
pp = pprint.PrettyPrinter(indent=2)


NAME_TO_TITLE = {
"map_DTvsphi_dxdz.png" : "map of dxdz residual vs phi",
"map_DTvsphi_dydz.png" : "map of dydz residual vs phi",
"map_DTvsphi_x.png" : "map of x residual vs phi",
"map_DTvsphi_y.png" : "map of y residual vs phi",
"map_DTvsz_dxdz.png" : "map of dxdz residual vs z",
"map_DTvsz_dydz.png" : "map of dydz residual vs z",
"map_DTvsz_x.png" : "map of x residual vs z",
"map_DTvsz_y.png" : "map of y residual vs z",
"map_CSCvsphi_dxdz.png" : "map of d(rphi)/dz residual vs phi",
"map_CSCvsphi_x.png" : "map of rphi residual vs phi",
"map_CSCvsr_dxdz.png" : "map of d(rphi)/dz residual vs r",
"map_CSCvsr_x.png" : "map of rphi residual vs r",
"segdifphi_dt13_resid.png" : "segdiff in x residuals vs phi",
"segdifphi_dt13_slope.png" : "segdiff in dxdz residuals vs phi",
"segdifphi_dt2_resid.png" : "segdiff in y residuals vs phi",
"segdifphi_dt2_slope.png" : "segdiff in dydz residuals vs phi",
"segdif_dt13_resid.png" : "segdiff in x residuals",
"segdif_dt13_slope.png" : "segdiff in dxdz residuals",
"segdif_dt2_resid.png" : "segdiff in y residuals",
"segdif_dt2_slope.png" : "segdiff in dydz residuals",
"segdif_csc_resid.png" : "segdiff in rphi residuals",
"segdif_csc_slope.png" : "segdiff in d(rphi)/dz residuals",
"dt_bellcurves.png" : "residuals distributions",
"dt_polynomials.png" : "residuals relations to misalignments",
"csc_bellcurves.png" : "residuals distributions",
"csc_polynomials.png" : "residuals relations to misalignments",
'dt_curvature_deltax.png' : 'Delta x residuals vs. curvature',
'dt_curvature_deltadxdz.png' : 'Delta dxdz residuals vs. curvature',
"medians.png" : "medians distribution"
}
######################################################
# functions definitions

######################################################
# To parse commandline args

usage='%prog [options]\n'+\
  'Creates a tree_items.js data file for a browsable JavaScript tree using results produced '+\
  'by running alignment_validation_plots.py.'

parser=optparse.OptionParser(usage)

parser.add_option("-i", "--inputDir",
  help="[REQUIRED] input directory: should contain 'iter1', 'iterN' and 'common' directories filled with alignment_validation_plots.py. The resulting tree_items.js is also dumped into this directory",
  type="string",
  default='',
  dest="inputDir")

parser.add_option("-v", "--verbose",
  help="Degree of debug info verbosity",
  type="int",
  default=0,
  dest="verbose")

options,args=parser.parse_args()

if options.inputDir=='':
  print "\nOne or more of REQUIRED options is missing!\n"
  parser.print_help()
  # See \n"+sys.argv[0]+" --help"
  sys.exit()

######################################################



############################################################################################################
############################################################################################################
# main script

# create directory structure

#basedir='/disks/sdb5/home_reloc/khotilov/db/cms/alignment'
#os.chdir(basedir)
os.chdir(options.inputDir)

#iteration1 = "iteration_01"
#iteration3 = "iteration_03"
#iteration1 = "NOV4DT_PASS3noweight_TkHIP_01"
#iteration3 = "NOV4DT_PASS3noweight_TkHIP_05"
iteration1 = "iter1"
iterationN = "iterN"
comdir = "common/"

######################################################
# open root and py result files

iteration_directory = iterationN


def parseDir(dir,label,it1="",itN=""):
  """it1 and itN   are the first and the last iterations' directory names
     dir           is some directory with the results from for the LAST 
                   iteration, so it must contain a itN substring 
     label         is a label for tree's folder for this directory"""
  if len(itN)>0 and dir.find(itN)==-1:
    print "directory ", dir, "has no ", itN, " in it!!"
    return ["problem!!!",""]
  res = [label,dir]
  files = os.listdir(dir)
  files.sort()
  for f in files:
    if re.match(".+\.png", f):
      if len(it1)>0 and len(itN)>0:
        lnN = [itN,dir+'/'+f]
        dir1 = dir.replace(itN,it1)
        if not os.access(dir1+'/'+f,os.F_OK):
          print "WARNING: no ",dir1+'/'+f," file found!!!"
        ln1 = [it1,dir1+'/'+f]
        ln = [NAME_TO_TITLE[f],dir+'/'+f,ln1,lnN]
        res.append(ln)
      else:
        ln = [NAME_TO_TITLE[f],dir+'/'+f]
        #print ln
        res.append(ln)
  #pp.pprint(res)
  return res


mytree = []
tree_level1 = ['test','']

# DT
dt_basedir = iteration_directory+'/MB/'
tree_level2 = parseDir(dt_basedir,"MB",iteration1,iterationN)
for wheel in DT_TYPES:
  dd = dt_basedir + wheel[0]
  print dd
  tree_level3 = parseDir(dd,wheel[0],iteration1,iterationN)
  for station in wheel[2]:
    dd = dt_basedir + wheel[0]+'/'+station[1]
    print dd 
    tree_level4 = parseDir(dd,station[0],iteration1,iterationN)
    for sector in range(1,station[2]+1):
      ssector = "%02d" % sector
      dd = dt_basedir+wheel[0]+'/'+station[1]+'/'+ssector
      #print dd
      tree_level5 = parseDir(dd,"%s/%d" % (station[0],sector),iteration1,iterationN)
      if len(tree_level5) == 2: tree_level5.append(['none',''])
      tree_level4.append(tree_level5)
    if len(tree_level4) == 2: tree_level4.append(['none',''])
    tree_level3.append(tree_level4)
  if len(tree_level3) == 2: tree_level3.append(['none',''])
  tree_level2.append(tree_level3)
if len(tree_level2) == 2: tree_level2.append(['none',''])
tree_level1.append(tree_level2)

# CSC
csc_basedir = iteration_directory+'/'
for endcap in CSC_TYPES:
  dd = csc_basedir+endcap[0]
  print dd
  tree_level2 = parseDir(dd,endcap[0],iteration1,iterationN)
  for station in endcap[2]:
    dd = csc_basedir+endcap[0]+'/'+station[1]
    print dd
    tree_level3 = parseDir(dd,station[0],iteration1,iterationN)
    for ring in station[2]:
      dd = csc_basedir+endcap[0]+'/'+station[1]+'/'+ring[1]
      print dd
      tree_level4 = parseDir(dd,"%s/%s" % (station[0],ring[1]),iteration1,iterationN)
      for chamber in range(1,ring[2]+1):
        schamber = "%02d" % chamber
        dd = csc_basedir+endcap[0]+'/'+station[1]+'/'+ring[1]+'/'+schamber
        #print dd
        tree_level5 = parseDir(dd,"%s/%s/%d" % (station[0],ring[1],chamber),iteration1,iterationN)
        tree_level4.append(tree_level5)
      if len(tree_level4) == 2: tree_level4.append(['none',''])
      tree_level3.append(tree_level4)
    if len(tree_level3) == 2: tree_level3.append(['none',''])
    tree_level2.append(tree_level3)
  if len(tree_level2) == 2: tree_level2.append(['none',''])
  tree_level1.append(tree_level2)

# Common plots
common_basedir = comdir
tree_level2 = parseDir(common_basedir,"All")
tree_level1.append(tree_level2)


mytree.append(tree_level1)
print " "
#pp.pprint(mytree)
print

ff = open("tree_items.js",mode="w")
print >>ff, "var TREE_ITEMS = "
json.dump(mytree,ff)
ff.close()
