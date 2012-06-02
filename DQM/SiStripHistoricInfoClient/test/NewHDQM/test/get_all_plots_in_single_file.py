#!/usr/bin/python
import os
import re
import subprocess
import sys
from ROOT import MakeNullPointer, TH1, TFile,TObject




print sys.argv

dir=sys.argv[1]
outName=sys.argv[2]
fileOut = TFile.Open(outName, "CREATE")

for item in os.listdir(dir):
    file1="%s" %item
    type=file1[-4]+file1[-3]+file1[-2]+file1[-1]
    if (type=='root'):
       file = TFile.Open(os.path.join("%s/%s"%(dir,item)), "READ")
       Nhist=file.GetListOfKeys().GetEntries();
       for i in range(0,Nhist):
            name=file.GetListOfKeys().At(i).GetName()
            obj=file.Get(name) 
            fileOut.cd()
            obj.Write()
       file.Close()
fileOut.Close()

