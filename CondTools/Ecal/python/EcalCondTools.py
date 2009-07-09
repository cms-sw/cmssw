#
# Misc functions to manipulate Ecal records
# author: Stefano Argiro
# id: $Id$
#
#
# WARNING: we assume that the list of iovs for a given tag
#          contains one element only, the case of several elements
#          will need to be addressed

from pluginCondDBPyInterface import *
from CondCore.Utilities import iovInspector as inspect
from ROOT import TCanvas,TH1F, TH2F


def unhashEBDetId(i):
    '''From hashedindex to ieta,iphi. Copied from EBDetId, we will have \
       a wrapper hopefully'''
    pseudo_eta= i/360 - 85
    ieta=0
    if pseudo_eta <0 :
        ieta = pseudo_eta
    else :
        ieta = pseudo_eta +1
          
    iphi = i%360 +1
    return ieta,iphi


def setWhat(w,ret) :
    '''This is from Vincenzo Innocente '''
    for key in ret.keys():
       _val = ret[key]
       if (type(_val)==type([])) :
          _vi = VInt()
          for i in _val :
              _vi.append(i)
              exec ('w.set_'+key+'(_vi)')
       else :
           exec ('w.set_'+key+'(w.'+key+'().'+ret[key]+')')
       return w 



def listTags(db):
    '''List all available tags for a given db '''
    tags=db.allTags()
    for tag in tags.split():
        print tag

def dumpXML(db,tag,filename='dump.xml'):
    '''Dump record in XML format for a given tag '''
    try :
       iov = inspect.Iov(db,tag)
       iovlist = iov.list()
       for p in iovlist:
          payload=inspect.PayLoad(db,p[0])
          out = open(filename,'w')
          print >> out, payload

    except Exception,er :
        print er

def plot (db, tag,filename='plot.root'):
    '''Invoke the plot function from the wrapper and save to the specified \
       file. The file format will reflect the extension given.'''
    try :
       iov = inspect.Iov(db,tag)
       iovlist = iov.list()
       for p in iovlist:
          payload=inspect.PayLoad(db,p[0])
          payload.plot(filename,"",[],[])

    except Exception,er :
        print er

def compare(tag1,db1,tag2,db2, filename='compare.root'):
  '''Produce comparison plots for two records. If no db2 is passed, will \
       assume we want to compare tags in the same db. Save plots to file \
       according to format'''

  try:  
    exec('import '+db1.moduleName(tag1)+' as Plug')
    
    what = {'how':'barrel'}
    w = setWhat(Plug.What(),what)
    ex = Plug.Extractor(w)
    for elem in db1.iov(tag1).elements :
        p = Plug.Object(elem)
        p.extract(ex)
        coeff_1 = [i for i in ex.values()]# first set of coefficients
        
  
    for elem in db2.iov(tag2).elements :
        p = Plug.Object(elem)
        p.extract(ex)
        coeff_2 = [i for i in ex.values()]# 2nd set of coefficients

    
    diff_distro_can = TCanvas("difference","difference")
    diff_distro_can.Divide(2)
    diff_distro_h   = TH1F("diffh","diffh",100,-2,2)
    diff_distro_m   = TH2F("diffm","diffm",171,-85,86,360,1,361)
    diff_distro_m.SetStats(0)
 
    for i,c in enumerate(coeff_1):  
        diff = c - coeff_2[i]
        ieta,iphi= unhashEBDetId(i)
        diff_distro_h.Fill(diff) 
        diff_distro_m.Fill(ieta,iphi,diff)

    diff_distro_can.cd(1)
    diff_distro_h.Draw()
    diff_distro_can.cd(2)
    diff_distro_m.Draw("colz")

    diff_distro_can.SaveAs(filename)

  except Exception, er :
    print er 


def histo (db, tag,filename='histo.root'):
    '''Make histograms and save to file'''
    try:  
      exec('import '+db.moduleName(tag)+' as Plug')
    
      what = {'how':'barrel'}
      w = setWhat(Plug.What(),what)
      ex = Plug.Extractor(w)
      for elem in db.iov(tag).elements :
          p = Plug.Object(elem)
          p.extract(ex)
          coeff_barl = [i for i in ex.values()]


      what = {'how':'endcap'}
      w = setWhat(Plug.What(),what)
      ex = Plug.Extractor(w)
      for elem in db.iov(tag).elements :
          p = Plug.Object(elem)
          p.extract(ex)
          coeff_endc = [i for i in ex.values()]     


      c =  TCanvas("CC distribution")
      c.Divide(2)
      eb = TH1F("EB","EB",100, -2,4)
      ee = TH1F("EE","EE",100, -2,4)

      for cb,ce in zip(coeff_barl,coeff_endc):
          eb.Fill(cb)
          ee.Fill(ce)
          
      c.cd(1)  
      eb.Draw()
      c.cd(2)
      ee.Draw()

      c.SaveAs(filename)

    except Exception, er :
       print er 
