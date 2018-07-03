#! /usr/bin/env python

import re,os,sys,shutil
import optparse

from mutypes import *

execfile("plotscripts.py")

ROOT.gROOT.SetBatch(1);

######################################################
# To parse commandline args

usage='%prog [options]\n'+\
  'a script to run CSC ring alignment procedure'+\
  'It is designed to run right after alignmentValidation.py script that was run with --map option. '

parser=optparse.OptionParser(usage)

parser.add_option("-l", "--runLabel",
  help="[REQUIRED] label to use for a run",
  type="string",
  default='',
  dest="runLabel")

parser.add_option("-d", "--dir",
  help="[REQUIRED] directory where tmp_test_results_map__{runLabel}.pkl fit results file is located",
  type="string",
  default='',
  dest="dir")

parser.add_option("-x", "--xml",
  help="[REQUIRED] xml file with input geometry",
  type="string",
  default='',
  dest="xml")

parser.add_option("--ring2only",
  help="if invoked, use only ring 2 results to align all rings in corresponding disks",
  action="store_true",
  dest="ring2only")

options,args=parser.parse_args()


if options.runLabel=='' or options.dir=='' or options.xml=='':
  print "\nOne or more of REQUIRED options is missing!\n"
  parser.print_help()
  sys.exit()

allOptions = "-l "+options.runLabel+" -i "+options.dir+" -x "+options.xml
#if options.diagnostic: allOptions += " --diagnostic"
print sys.argv[0]+" "+allOptions

pwd = str(os.getcwd())

if not os.access(options.dir,os.F_OK):
  print "Directory " + options.dir + " does not exist. Exiting..."
  sys.exit()
os.chdir(options.dir)

if not loadTestResultsMap(options.runLabel):
  print "Cant open pickle file with mapplots fit results. Exiting..."
  sys.exit()


xml_corr = {}

print "       \tdX(mm)   \t dY(mm)   \tdPhiZ(mrad)"
for endcap in CSC_TYPES:
  for station in endcap[2]:
    for ring in station[2]:
      if ring[1]=="ALL": continue
      # skip ME4/2 for now
      if station[1]=="4" and ring[1]=="2": continue
      
      ring_id = "%s%s/%s" % (endcap[0], station[1],ring[1])
      
      if ring_id in MAP_RESULTS_FITSIN:
        postal_address = idToPostalAddress(ring_id+'/01')
        
        fits = MAP_RESULTS_FITSIN[ring_id]
        d_x, de_x = fits['sin'][0]/10., fits['sin'][1]/10.
        d_y, de_y = -fits['cos'][0]/10., fits['cos'][1]/10.
        d_phiz, de_phiz = -fits['a'][0]/10./signConventions[postal_address][3], fits['a'][1]/10./signConventions[postal_address][3]
        
        print "%s \t%+.2f+-%.2f  \t%+.2f+-%.2f \t%+.2f+-%.2f" % (ring_id, d_x*10 , de_x*10, d_y*10 , de_y*10 , d_phiz*1000, de_phiz*1000)
        
        e = endcap[3]
        s = station[1]
        r = ring[1]
        xml_corr[ring_id] = "<setposition relativeto=\"none\" x=\"%(d_x)s\" y=\"%(d_y)s\" phiz=\"%(d_phiz)s\" />" % vars()

xml_str = """
"""
for endcap in CSC_TYPES:
  for station in endcap[2]:
    for ring in station[2]:
      if ring[1]=="ALL": continue
      # skip ME4/2 for now
      #if station[1]=="4" and ring[1]=="2": continue
      
      r_with_corr = ring[1]
      s_with_corr = station[1]
      # use ME4/1 for aligning ME4/2
      if station[1]=="4" and ring[1]=="2": r_with_corr = "1"
      # ring 2 only option
      if options.ring2only : r_with_corr = "2"
      if options.ring2only and station[1]=="3": s_with_corr = "2"
      # no matter what, always use ME11 to correct ME11
      if station[1]=="1" and ring[1]=="1": r_with_corr = "1"

      # for jim's BH cross-check
      #if station[1]=="1" and ring[1]=="3": r_with_corr = "2"
      #if station[1]=="2" or  station[1]=="3": 
      #   r_with_corr = "1"
      #   s_with_corr = "2"
      
      ring_id = "%s%s/%s" % (endcap[0], s_with_corr, r_with_corr)

      if ring_id in xml_corr:
        corr = xml_corr[ring_id]
        e = endcap[3]
        s = station[1]
        r = ring[1]
        xml_str += """<operation>
  <CSCRing endcap=\"%(e)s\" station=\"%(s)s\" ring=\"%(r)s\" />
  %(corr)s
</operation>

""" % vars()
        # if it's ME1/1, move ME1/4 the same amount as well
        if s=="1" and r=="1":
          xml_str += """<operation>
  <CSCRing endcap=\"%(e)s\" station=\"%(s)s\" ring=\"4\" />
  %(corr)s
</operation>

""" % vars()

print xml_str
xml_str += "</MuonAlignment>"

os.chdir(pwd)

ff = open(options.xml+".ring",mode="w")
print >>ff, xml_str
ff.close()

os.system('grep -v "</MuonAlignment>" %s > %s' % (options.xml, options.xml+".tmp"))
os.system('cat %s %s > %s' % (options.xml+".tmp", options.xml+".ring", options.xml+".ring.xml") )
os.system('rm %s %s' % (options.xml+".tmp", options.xml+".ring") )
