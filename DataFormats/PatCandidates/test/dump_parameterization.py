#!/bin/env python
from __future__ import print_function
from builtins import range
import ROOT
from ROOT import *
import sys
if len(sys.argv) < 2 :
  print("Usage: dump_parameteriza.py filename")
  exit()
fileToRead= TFile(sys.argv[1], "read")

fileToRead.cd()



SchemasMeaningFile = open("SchemasMeaning.txt", "w")
SchemasMeaningFile.write("------------------------LEGEND-------------------------  \n")
SchemasMeaningFile.write("METHOD:  \n")
SchemasMeaningFile.write("\t0 \t float16  \n")
SchemasMeaningFile.write("\t1 \t reduceMantissa  \n")
SchemasMeaningFile.write("\t2 \t logPack  \n")
SchemasMeaningFile.write("\t3 \t tanLogPack  \n")
SchemasMeaningFile.write("\t4 \t zero  \n")
SchemasMeaningFile.write("\t5 \t one  \n")
    
    
SchemasMeaningFile.write("TARGET:  \n")
SchemasMeaningFile.write("\t0 \t realValue  \n")
SchemasMeaningFile.write("\t1 \t ratioToRef  \n")
SchemasMeaningFile.write("\t2 \t differenceToRef  \n\n")



next = TIter ((fileToRead.Get("schemas")).GetListOfKeys())


key = next()
while key != None  :

    cl = gROOT.GetClass(key.GetClassName())  
    schemaNumber = key.ReadObj().GetName();
    schemaN = int(schemaNumber);

    SchemasMeaningFile.write("SCHEMA " + schemaNumber + "\n")
    SchemasMeaningFile.write("ELEMENT \t METHOD \tTARGET  \tBIT USED \t  Lmin \t\t  Lmax" + "\n")
    

    for i in range(0,5) :
        for j in range(i,5) :
            folder = "schemas/"+schemaNumber+"/"+str(i)+str(j)



            method = str(fileToRead.Get(folder+"/method").GetVal())
            target = str(fileToRead.Get(folder+"/target").GetVal())
            bit    = str(fileToRead.Get(folder+"/bit").GetVal())
            param0 = str(fileToRead.Get(folder+"/param")[0])
            param1 = str(fileToRead.Get(folder+"/param")[1])
            
            
            SchemasMeaningFile.write(str(i)+str(j) + "\t\t " + method + "\t\t  " + target + "\t\t  " + bit + "\t\t   " + param0 + "  \t  " + param1 + "\n")


        
    key = next()


fileToRead.Close()
SchemasMeaningFile.close()





