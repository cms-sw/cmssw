#! /usr/bin/env python

import sys, ROOT

from geometryDiffVisualization import *

ROOT.gROOT.SetBatch(1)

cargs = sys.argv[:]

if len(cargs) != 5:
  print "Makes schematic drawings of the detector in various projections with overlayed chambers"
  print "shifted and rotated by their correction amounts times some scale (geom2-geom1)"
  print "usage: ./geometryDiffVisualizer.py label svg_template_dir geometry2.xml geometry1.xml"
  print "The label will be included into the filenames as geoVisual_label__specifier.png"
  print ""
  sys.exit()

label = cargs[1]
svg_template_dir = cargs[2]

xmlfile2 = cargs[3]
xmlfile1 = cargs[4]

g2 = MuonGeometry(xmlfile2)
g1 = MuonGeometry(xmlfile1)

pfx = "geoVisual__" + label + "__"
sf_dt = 200.
sf_csc = 100.

draw_station(g2, g1, 1, pfx+"st_1_DT.svg", length_factor=sf_dt, angle_factor=sf_dt, template_dir=svg_template_dir)
draw_station(g2, g1, 2, pfx+"st_2_DT.svg", length_factor=sf_dt, angle_factor=sf_dt, template_dir=svg_template_dir)
draw_station(g2, g1, 3, pfx+"st_3_DT.svg", length_factor=sf_dt, angle_factor=sf_dt, template_dir=svg_template_dir)
draw_station(g2, g1, 4, pfx+"st_4_DT.svg", length_factor=sf_dt, angle_factor=sf_dt, template_dir=svg_template_dir)

draw_wheel(g2, g1, -2, pfx+"wh_a_DT.svg", length_factor=sf_dt, angle_factor=sf_dt, template_dir=svg_template_dir)
draw_wheel(g2, g1, -1, pfx+"wh_b_DT.svg", length_factor=sf_dt, angle_factor=sf_dt, template_dir=svg_template_dir)
draw_wheel(g2, g1,  0, pfx+"wh_c_DT.svg", length_factor=sf_dt, angle_factor=sf_dt, template_dir=svg_template_dir)
draw_wheel(g2, g1, +1, pfx+"wh_d_DT.svg", length_factor=sf_dt, angle_factor=sf_dt, template_dir=svg_template_dir)
draw_wheel(g2, g1, +2, pfx+"wh_e_DT.svg", length_factor=sf_dt, angle_factor=sf_dt, template_dir=svg_template_dir)

draw_disk(g2, g1, 1, 1, pfx+"e1_st1_CSC.svg", length_factor=sf_csc, angle_factor=sf_csc, template_dir=svg_template_dir)
draw_disk(g2, g1, 1, 2, pfx+"e1_st2_CSC.svg", length_factor=sf_csc, angle_factor=sf_csc, template_dir=svg_template_dir)
draw_disk(g2, g1, 1, 3, pfx+"e1_st3_CSC.svg", length_factor=sf_csc, angle_factor=sf_csc, template_dir=svg_template_dir)
draw_disk(g2, g1, 1, 4, pfx+"e1_st4_CSC.svg", length_factor=sf_csc, angle_factor=sf_csc, template_dir=svg_template_dir)
draw_disk(g2, g1, 2, 1, pfx+"e2_st1_CSC.svg", length_factor=sf_csc, angle_factor=sf_csc, template_dir=svg_template_dir)
draw_disk(g2, g1, 2, 2, pfx+"e2_st2_CSC.svg", length_factor=sf_csc, angle_factor=sf_csc, template_dir=svg_template_dir)
draw_disk(g2, g1, 2, 3, pfx+"e2_st3_CSC.svg", length_factor=sf_csc, angle_factor=sf_csc, template_dir=svg_template_dir)
draw_disk(g2, g1, 2, 4, pfx+"e2_st4_CSC.svg", length_factor=sf_csc, angle_factor=sf_csc, template_dir=svg_template_dir)
