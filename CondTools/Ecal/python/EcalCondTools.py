#
# Misc functions to manipulate Ecal records
# author: Stefano Argiro
# id: $Id: EcalCondTools.py,v 1.4 2009/07/14 16:52:11 argiro Exp $
#
#
# WARNING: we assume that the list of iovs for a given tag
#          contains one element only, the case of several elements
#          will need to be addressed

#from pluginCondDBPyInterface import *
from CondCore.Utilities import iovInspector as inspect
from ROOT import TCanvas,TH1F, TH2F
import EcalPyUtils
import sys


def listTags(db):
    '''List all available tags for a given db '''
    tags=db.allTags()
    for tag in tags.split():
        print tag

def listIovs(db,tag):
    '''List all available iovs for a given tag'''

    try :
       iov = inspect.Iov(db,tag)
       iovlist = iov.list()
       print "Available iovs for tag: ",tag
       for p in iovlist:
           print "  Since " , p[1], " Till " , p[2]
     
    except Exception,er :
        print er 

def dumpXML(db,tag,since,till,filename='dump.xml'):
    '''Dump record in XML format for a given tag '''
    try :
       iov = inspect.Iov(db,tag)
       token = getToken(db,tag,since,till)       
       payload=inspect.PayLoad(db,token)
       out = open(filename,'w')
       print >> out, payload
      
    except Exception,er :
        print er

def plot (db, tag,since,till,filename='plot.root'):
    '''Invoke the plot function from the wrapper and save to the specified \
       file. The file format will reflect the extension given.'''
    
    try :
        iov = inspect.Iov(db,tag)
        iovlist = iov.list()
        token = getToken(db,tag,since,till)       
        payload=inspect.PayLoad(db,token)
        payload.plot(filename,"",[],[])
            
    except Exception,er :
        print er
        

def compare(tag1,db1,since1,till1,
            tag2,db2,since2,till2,filename='compare.root'):
  '''Produce comparison plots for two records. Save plots to file \
     according to format. tag can be an xml file'''
  coeff_1=[]
  coeff_2=[]
  
  if  tag1.find(".xml") < 0:
      try:
        exec('import '+db1.moduleName(tag1)+' as Plug')  
        what = {'how':'barrel'}
        w = inspect.setWhat(Plug.What(),what)
        ex = Plug.Extractor(w)
        p = getObject(db1,tag1,since1,till1)
        p.extract(ex)
        coeff_1 = [i for i in ex.values()]# first set of coefficients

      except Exception,er :
          print er
  else:
      coeff_1,coeff_1_ee = EcalPyUtils.fromXML(tag1)

  if  tag2.find(".xml")<0:
      try:  
        exec('import '+db2.moduleName(tag2)+' as Plug')
        what = {'how':'barrel'}
        w = inspect.setWhat(Plug.What(),what)
        ex = Plug.Extractor(w)
        p = getObject(db2,tag2,since2,till2)
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




def histo (db, tag,since,till,filename='histo.root'):
    '''Make histograms and save to file. tag can be an xml file'''
    
    coeff_barl=[]
    coeff_endc=[]
    
    if  tag.find(".xml")< 0:
        try:  
          exec('import '+db.moduleName(tag)+' as Plug')

          what = {'how':'barrel'}
          w = inspect.setWhat(Plug.What(),what)
          ex = Plug.Extractor(w)
          p=getObject(db,tag,since,till)
          p.extract(ex)
          coeff_barl = [i for i in ex.values()]


          what = {'how':'endcap'}
          w = inspect.setWhat(Plug.What(),what)
          ex = Plug.Extractor(w)
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


def getToken(db,tag,since,till):
    ''' Return payload token for a given iov, tag, db'''
    try :
       iov = inspect.Iov(db,tag)
       iovlist = iov.list()
       for p in iovlist:
           tmpsince=p[1]
           tmptill =p[2]
           if tmpsince==since and tmptill==till:
               return p[0]
       print "Could not retrieve token for tag: " , tag, " since: ", since,\
              " till: " ,till
       sys.exit(0)
       
    except Exception, er :
       print er


def getObject(db,tag,since,till):
    ''' Return payload object for a given iov, tag, db'''
    found=0
    try:
       exec('import '+db.moduleName(tag)+' as Plug')  
       for elem in db.iov(tag).elements :       
           if str(elem.since())==str(since) and str(elem.till())==str(till):
               found=1
               return Plug.Object(elem)
           
    except Exception, er :
        print er

    if not found :
        print "Could not retrieve payload for tag: " , tag, " since: ", since,\
          " till: " ,till
        sys.exit(0)
