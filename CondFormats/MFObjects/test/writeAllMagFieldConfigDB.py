#!/usr/bin/python

import os
import sys

SETUPS = ('71212',  '',           ('0T','2T','3T','3_5T','4T')), \
         ('90322',  '2pi_scaled', (['3_8T'])), \
         ('120812', 'Run1',       (['3_8T'])), \
         ('120812', 'Run2',       (['3_8T'])), \
         ('130503', 'Run1',       ('3_5T','3_8T')), \
         ('130503', 'Run2',       ('3_5T','3_8T')),


for SETUP in SETUPS :
    SET = SETUP[0]
    SUBSET = SETUP[1]
    for B_NOM in SETUP[2] : 
       print SET, SUBSET, B_NOM
       sys.stdout.flush()
       namespace = {'SET':SET, 'SUBSET':SUBSET, 'B_NOM':B_NOM}
       execfile("writeMagFieldConfigDB.py",namespace)
       process = namespace.get('process') 
       cfgFile = open('run.py','w')       
       cfgFile.write( process.dumpPython() )
       cfgFile.write( '\n' )
       cfgFile.close()
       os.system("cmsRun run.py")
       del namespace
       del process
       print ""
       
