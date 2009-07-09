#
# Misc functions to manipulate Ecal records
# author: Stefano Argiro
# id: $Id: EcalCondTools.py,v 1.1 2009/07/09 10:27:01 argiro Exp $
#
#
# WARNING: we assume that the list of iovs for a given tag
#          contains one element only, the case of several elements
#          will need to be addressed

from pluginCondDBPyInterface import *
from CondCore.Utilities import iovInspector as inspect
from ROOT import TCanvas,TH1F, TH2F
import EcalPyUtils



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
       according to format. tag can be an xml file'''
  
  if not tag1.find(".xml"):
      try:  
        exec('import '+db1.moduleName(tag1)+' as Plug')

        what = {'how':'barrel'}
        w = inspect.setWhat(Plug.What(),what)
        ex = Plug.Extractor(w)

        for elem in db1.iov(tag1).elements :
            p = Plug.Object(elem)
            p.extract(ex)
            coeff_1 = [i for i in ex.values()]# first set of coefficients

      except Exception,er :
          print er
  else:
      coeff_1,coeff_1_ee = EcalPyUtils.fromXML(tag1)

  if not tag2.find(".xml"):
      try:  
        exec('import '+db2.moduleName(tag2)+' as Plug')

        what = {'how':'barrel'}
        w = inspect.setWhat(Plug.What(),what)
        ex = Plug.Extractor(w)
        
        for elem in db2.iov(tag2).elements :
            p = Plug.Object(elem)
            p.extract(ex)
            coeff_2 = [i for i in ex.values()]# 2nd set of coefficients
        
      except Exception, er :
          print er
  else:
      coeff_2,coeff_2_ee = EcalPyUtils.fromXML(tag2)

    
  diff_distro_can = TCanvas("difference","difference")
  diff_distro_can.Divide(2)
  diff_distro_h   = TH1F("diffh","diffh",100,-2,2)
  diff_distro_m   = TH2F("diffm","diffm",171,-85,86,360,1,361)
  diff_distro_m.SetStats(0)
  
  for i,c in enumerate(coeff_1):  
      diff = c - coeff_2[i]
      ieta,iphi= EcalPyUtils.unhashEBIndex(i)
      diff_distro_h.Fill(diff) 
      diff_distro_m.Fill(ieta,iphi,diff)

  diff_distro_can.cd(1)
  diff_distro_h.Draw()
  diff_distro_can.cd(2)
  diff_distro_m.Draw("colz")

  diff_distro_can.SaveAs(filename)




def histo (db, tag,filename='histo.root'):
    '''Make histograms and save to file. tag can be an xml file'''
    if not tag.find(".xml"):
        try:  
          exec('import '+db.moduleName(tag)+' as Plug')

          what = {'how':'barrel'}
          w = inspect.setWhat(Plug.What(),what)
          ex = Plug.Extractor(w)      
          for elem in db.iov(tag).elements :
              p = Plug.Object(elem)
              p.extract(ex)
              coeff_barl = [i for i in ex.values()]


          what = {'how':'endcap'}
          w = inspect.setWhat(Plug.What(),what)
          ex = Plug.Extractor(w)
          for elem in db.iov(tag).elements :
              p = Plug.Object(elem)
              p.extract(ex)
              coeff_endc = [i for i in ex.values()]     

        except Exception, er :
          print er 

    else :
        coeff_barl,coeff_endc=EcalPyUtils.fromXML(tag)


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

