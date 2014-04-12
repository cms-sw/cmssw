#! /usr/bin/env python

import os,sys, DLFCN,getopt
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from CondCore.Utilities import iovInspector as inspect
from pluginCondDBPyInterface import *
import pluginCondDBPyInterface as CondDB
from ROOT import TCanvas,TH1F, TH2F,TFile


def unhashEBDetId(i):
       pseudo_eta= i/360 - 85;
       ieta=0
       if pseudo_eta <0 :
          ieta = pseudo_eta
       else :
          ieta = pseudo_eta +1
          
       iphi = i%360 +1
       return ieta,iphi

def setWhat(w,ret) :
       for key in ret.keys():
              _val = ret[key]
              if (type(_val)==type([])) :
                     _vi = CondDB.VInt()
                     for i in _val :
                            _vi.append(i)
                     exec ('w.set_'+key+'(_vi)')
              else :
                     exec ('w.set_'+key+'(w.'+key+'().'+ret[key]+')')
       return w 



def usage():
    print "inspectEcal -c [connectstring] -P [authpath] -t [tag] -f [outfile] -l -h"
    print "   dump records in xml" 
    print "               -l: list tags and exit"
    print "               -f [file] : dump to file"
    print "               -p plot distribution   "
    print "               -q compare [tag]       "
    print "               -r reference [tag]     "
    print "               -m draw map"
    print "               -h : help"
      
try:
    opts, args = getopt.getopt(sys.argv[1:], "c:P:t:f:lhpq:r:m", ["connect=","authpath=","tag","file","listtags","help","plot","compare","reference","map"])
    
    if not len(opts):
        usage()
        sys.exit(0)

except getopt.GetoptError:
    #* print help information and exit:*
    usage()
    sys.exit(2)


dbName =  "oracle://cms_orcoff_prod/CMS_COND_31X_ECAL"
authpath= "/afs/cern.ch/cms/DB/conddb"
tag='EcalIntercalibConstants_mc'
do_list_tags= 0
dump_to_file =0
outfile=""
do_plot=0
do_compare=0
compare_tag=""
reference_tag=""
drawmap=0

for opt,arg in opts:

    if opt in ("-c","--connect"):
        try: 
            dbname=arg
        except Exception, er :
            print er
    
    if opt in ("-P","--authpath"):
        try: 
            rdbms=RDBMS(arg)
        except Exception, er :
            print er
    if opt in ("-t","--tag"):
        tag=arg

    if opt in ("-l","--listtags"):
        do_list_tags= 1
        
    if opt in ("-f","--file"):
        dump_to_file= 1
        outfile=arg    

    if opt in ("-p","--plot"):
        do_plot= 1       

    if opt in ("-q","--compare"):
        do_compare=1
        compare_tag=arg

    if opt in ("-r","--reference"):
        reference_tag=arg

    if opt in ("-m","--map"):
        drawmap=1
        
    if opt in ("-h","--help"):
        usage()
        sys.exit(0)    

a = FWIncantation()

rdbms = RDBMS(authpath)
db = rdbms.getDB(dbName)

if do_list_tags :
    tags=db.allTags()
    for tag in tags.split():
        print tag
    sys.exit(0)    


try :
    iov = inspect.Iov(db,tag)
    print "===iov list ==="
    iovlist=iov.list()
    print iovlist
    print "===iov summaries ==="
    print iov.summaries()
    print "===payload dump ==="
    for p in iovlist:
        payload=inspect.PayLoad(db,p[0])
        #print payload.summary()
        if dump_to_file:
            print "Dumping to file:", outfile 
            out = open(outfile,"w")
            print >> out, payload
        else:
             #print payload
             if  drawmap:
                payload.plot("plot","",[],[])

    if do_plot:
       exec('import '+db.moduleName(tag)+' as Plug')
           #what = {'how':'singleChannel','which': [0,1,2]}
       what = {'how':'barrel'}
       w = setWhat(Plug.What(),what)
       ex = Plug.Extractor(w)
       for elem in db.iov(tag).elements :
          p = Plug.Object(elem)
          p.extract(ex)
          v = [i for i in ex.values()]        
 #      print v
       histo=TH1F("h","h",100,-2,2) 
       for c in v :
          histo.Fill(c)       
       f=TFile("f.root","recreate")
       histo.Write()

    if do_compare:
       exec('import '+db.moduleName(tag)+' as Plug')
       what = {'how':'barrel'}
       w = setWhat(Plug.What(),what)
       ex = Plug.Extractor(w)
       for elem in db.iov(reference_tag).elements :
          p = Plug.Object(elem)
          p.extract(ex)
          coeff_1 = [i for i in ex.values()]

       for elem in db.iov(compare_tag).elements :
          p = Plug.Object(elem)
          p.extract(ex)
          coeff_2 = [i for i in ex.values()]

       can=TCanvas("c","c")

       histo = TH1F("h","h",100,-2,2)
       for i,c in enumerate(coeff_1):  
          histo.Fill(c-coeff_2[i])
       histo.Draw()   

       can.SaveAs("h.svg")

       can2=TCanvas("cc","cc")
       histo2=TH2F("hh","hh",171,-85,86,360,1,361)
       for i,c in enumerate(coeff_1):  
              factor = c/coeff_2[i]
              ieta,iphi= unhashEBDetId(i)
              histo2.Fill(ieta,iphi,factor)

       histo2.SetStats(0)       
       histo2.Draw("colz")
       can2.SaveAs("h2.svg")
              
except Exception, er :
    print er


