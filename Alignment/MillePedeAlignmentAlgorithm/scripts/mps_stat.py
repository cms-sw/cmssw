#!/usr/bin/env python
import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib
import os

#use mps_update.py in shell and push output to nirvana
os.system("mps_update.py >| /dev/null")  #add >| /dev/null

lib = mpslib.jobdatabase()	#create object of class jobdatabase
lib.read_db()				#read mps.db into the jobdatabase
lib.print_memdb()			#print the jobdatabase in memory
