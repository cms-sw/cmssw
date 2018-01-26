import os,sys, glob

spyInput = '/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/streamer/SiStripSpy/Commissioning11/'
spyRun   = '298270/'

for path in glob.glob(spyInput+spyRun+'*.root'):
    print "'file:"+path+"',"
