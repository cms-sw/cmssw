from __future__ import print_function
import os,sys, glob

spyInput = '/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/'
spyRun   = '234824/'

for path in glob.glob(spyInput+spyRun+'*.root'):
    print("'file:"+path+"',")
