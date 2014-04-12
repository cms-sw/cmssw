#! /usr/bin/env python
import os
import re
import sys
import subprocess

#******************   template file  **********************************
templateFile = open('templateForDropbox.txt', 'r')
fileContents = templateFile.read(-1)
print '--------------- TEMPLATE :  -----------------'
print fileContents
p1 = re.compile(r'TAGNAME')
p2 = re.compile(r'PRODNAME')

#******************   definitions  **********************************
jec_type    = 'JetCorrectorParametersCollection'
ERA         = 'Jec11_V10'
ALGO_LIST   = [#'IC5Calo','IC5PF',
               'AK5Calo','AK5PF','AK5PFchs','AK5JPT'#,#'AK5TRK',
               'AK7Calo','AK7PF',
               'KT4Calo','KT4PF',
               #'KT6Calo','KT6PF'
               ]
#*********************************************************************

files = []


### L2+L3 Corrections
for aa in ALGO_LIST: #loop for jet algorithms

    s1 = jec_type + '_' + ERA + '_' + aa
    s2 = jec_type + '_' + ERA + '_' + aa
    k1 = p1.sub( s1, fileContents )
    k2 = p2.sub( s2, k1 )
    k2outfile = s2 + '.txt'
    print '--------------------------------------'
    print 'ORCOFF File for jet correction : ' + s2
    print 'Written to ' + k2outfile
    FILE = open(k2outfile,"w")
    FILE.write(k2)       
    files.append( k2outfile )
    


for ifile in files :
    s = "./dropBoxOffline_test.sh "+ERA+".db " + ifile
    print s
    subprocess.call( ["./dropBoxOffline_test.sh", ERA+".db", ifile])
  
