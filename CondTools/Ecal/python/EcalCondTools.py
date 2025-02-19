#
# Misc functions to manipulate Ecal records
# author: Stefano Argiro
# id: $Id: EcalCondTools.py,v 1.13 2011/10/10 09:02:27 fay Exp $
#
#
# WARNING: we assume that the list of iovs for a given tag
#          contains one element only, the case of several elements
#          will need to be addressed

#from pluginCondDBPyInterface import *
from CondCore.Utilities import iovInspector as inspect
from ROOT import TCanvas,TH1F, TH2F, gStyle, TChain, TTree, TLegend, TFile
import EcalPyUtils
import sys
from math import sqrt

def listTags(db):
    '''List all available tags for a given db '''
    try:
        db.startTransaction()
        tags = db.allTags()
        db.commitTransaction()
        for tag in tags.split():
            print tag
    except:
        return ""

def listIovs(db,tag):
    '''List all available iovs for a given tag'''

    try :
       iov = inspect.Iov(db,tag)
       iovlist = iov.list()
       print "Available iovs for tag: ",tag
       for p in iovlist:
           print "  Since " , p[1], " Till " , p[2]
     
    except Exception,er :
        print " listIovs exception ",er 

def dumpXML(db,tag,since,filename='dump.xml'):
    '''Dump record in XML format for a given tag '''
    try :
       iov = inspect.Iov(db,tag)
       db.startTransaction()
       Plug = __import__(str(db.payloadModules(tag)[0]))
       payload = Plug.Object(db)
       listOfIovElem= [iovElem for iovElem in db.iov(tag).elements]
       inst = 0
       for elem in db.iov(tag).elements :
           inst = inst + 1
           if str(elem.since())==str(since):
               found = inst
               break
       db.commitTransaction()
       payload = inspect.PayLoad(db, tag, elem)
       out = open(filename,'w')
       print >> out, payload
      
    except Exception,er :
        print " dumpXML exception ",er

def plot (db, tag,since,filename='plot.root'):
    '''Invoke the plot function from the wrapper and save to the specified \
       file. The file format will reflect the extension given.'''
   
    try :
       iov = inspect.Iov(db,tag)
       db.startTransaction()
       Plug = __import__(str(db.payloadModules(tag)[0]))
       payload = Plug.Object(db)
       listOfIovElem= [iovElem for iovElem in db.iov(tag).elements]
       inst = 0
       for elem in db.iov(tag).elements :
           inst = inst + 1
           if str(elem.since())==str(since):
               found = inst
               break
       db.commitTransaction()
       payload = inspect.PayLoad(db, tag, elem)
       payload.plot(filename,"",[],[])
            
    except Exception,er :
        print " plot exception ",er
        

def compare(tag1,db1,since1,
            tag2,db2,since2,filename='compare.root'):
  '''Produce comparison plots for two records. Save plots to file \
     according to format. tag can be an xml file'''
  print "EcalCondTools.py compare tag1 ", tag1, "since1 : ", since1, " tag2 ", tag2," s2 ", since2

  coeff_1_b=[]
  coeff_2_b=[]

  coeff_1_e=[]
  coeff_2_e=[]
  
  if  tag1.find(".xml") < 0:
      found=0
      try:
        db1.startTransaction()
        Plug = __import__(str(db1.payloadModules(tag1)[0]))
        payload = Plug.Object(db1)
        listOfIovElem= [iovElem for iovElem in db1.iov(tag1).elements]
        what = {'how':'barrel'}
        w = inspect.setWhat(Plug.What(),what)
        exb = Plug.Extractor(w)
        what = {'how':'endcap'}
        w = inspect.setWhat(Plug.What(),what)
        exe = Plug.Extractor(w)
#        p = getObject(db1,tag1,since1,ex)
#        p.extract(ex)
        print " before loop"
        for elem in db1.iov(tag1).elements :       
          if str(elem.since())==str(since1):
            found=1
            payload.load(elem)
            payload.extract(exb)
            coeff_1_b = [i for i in exb.values()]# first set of coefficients
            payload.extract(exe)
            coeff_1_e = [i for i in exe.values()]
        db1.commitTransaction()

      except Exception,er :
        print " compare first set exception ",er
      if not found :
        print "Could not retrieve payload for tag: " , tag1, " since: ", since1
        sys.exit(0)

  else:
      coeff_1_b,coeff_1_e = EcalPyUtils.fromXML(tag1)

  if  tag2.find(".xml")<0:
      found=0
      try:  
        db2.startTransaction()
        Plug = __import__(str(db2.payloadModules(tag2)[0]))
        what = {'how':'barrel'}
        w = inspect.setWhat(Plug.What(),what)
        exb = Plug.Extractor(w)
        what = {'how':'endcap'}
        w = inspect.setWhat(Plug.What(),what)
        exe = Plug.Extractor(w)
#        p = getObject(db2,tag2,since2)
#        p.extract(ex)
        for elem in db2.iov(tag2).elements :       
          if str(elem.since())==str(since2):
            found=1
            payload.load(elem)
            payload.extract(exb)
            coeff_2_b = [i for i in exb.values()]# second set of coefficients
            payload.extract(exe)
            coeff_2_e = [i for i in exe.values()]
        db2.commitTransaction()
     
      except Exception, er :
          print " compare second set exception ",er
      if not found :
        print "Could not retrieve payload for tag: " , tag2, " since: ", since2
        sys.exit(0)

  else:
      coeff_2_b,coeff_2_e = EcalPyUtils.fromXML(tag2)

  gStyle.SetPalette(1)    

  savefile = TFile(filename,"RECREATE")

  ebhisto,ebmap, profx, profy= compareBarrel(coeff_1_b,coeff_2_b)
  eephisto,eepmap,eemhisto,eemmap=compareEndcap(coeff_1_e,coeff_2_e)

#make more canvas (suppressed : use a root file)

#  cEBdiff = TCanvas("EBdiff","EBdiff")
#  cEBdiff.Divide(2,2)

#  cEBdiff.cd(1)
#  ebhisto.Draw()
#  cEBdiff.cd(2)
#  ebmap.Draw("colz")
#  cEBdiff.cd(3)
#  profx.Draw()
#  cEBdiff.cd(4)
#  profy.Draw()

#  EBfilename = "EB_"+filename
#  cEBdiff.SaveAs(EBfilename)

#  cEEdiff = TCanvas("EEdiff","EEdiff")
#  cEEdiff.Divide(2,2)
  
#  cEEdiff.cd(1)
#  eephisto.Draw()
#  cEEdiff.cd(2)
#  eepmap.Draw("colz")
#  cEEdiff.cd(3)
#  eemhisto.Draw()
#  cEEdiff.cd(4)
#  eemmap.Draw("colz")

#  EEfilename = "EE_"+filename
#  cEEdiff.SaveAs(EEfilename)


  eeborderphisto,eeborderpmap,eebordermhisto,eebordermmap=compareEndcapBorder(coeff_1_e,coeff_2_e)
  ebborderhisto,ebbordermap = compareBarrelBorder(coeff_1_b,coeff_2_b)

  border_diff_distro_can = TCanvas("border_difference","borders difference")
  border_diff_distro_can.Divide(2,3)

  border_diff_distro_can.cd(1)
  ebborderhisto.Draw()
  border_diff_distro_can.cd(2)
  ebbordermap.Draw("colz")
  border_diff_distro_can.cd(3)
  eeborderphisto.Draw()
  border_diff_distro_can.cd(4)
  eeborderpmap.Draw("colz")
  border_diff_distro_can.cd(5)
  eebordermhisto.Draw()
  border_diff_distro_can.cd(6)
  eebordermmap.Draw("colz")

  bordersfilename = "borders_"+filename
  prof_filename = "profiles_"+filename
  
#  border_diff_distro_can.SaveAs(bordersfilename)

  savefile.Write()



def histo (db, tag,since,filename='histo.root'):
    '''Make histograms and save to file. tag can be an xml file'''
    
    coeff_barl=[]
    coeff_endc=[]
    
    if  tag.find(".xml")< 0:
      found=0
      try:  
#          exec('import '+db.moduleName(tag)+' as Plug')
        db.startTransaction()
        Plug = __import__(str(db.payloadModules(tag)[0]))
        payload = Plug.Object(db)
        listOfIovElem= [iovElem for iovElem in db.iov(tag).elements]
        what = {'how':'barrel'}
        w = inspect.setWhat(Plug.What(),what)
        exb = Plug.Extractor(w)
        what = {'how':'endcap'}
        w = inspect.setWhat(Plug.What(),what)
        exe = Plug.Extractor(w)
        for elem in db.iov(tag).elements :       
          if str(elem.since())==str(since):
            found=1
            payload.load(elem)
            payload.extract(exb)
            coeff_barl = [i for i in exb.values()]
            payload.extract(exe)
            coeff_endc = [i for i in exe.values()]
        db.commitTransaction()

      except Exception, er :
          print " histo exception ",er
      if not found :
        print "Could not retrieve payload for tag: " , tag, " since: ", since
        sys.exit(0)

    else :
      coeff_barl,coeff_endc=EcalPyUtils.fromXML(tag)

    savefile = TFile(filename,"RECREATE")

    ebmap, ebeta, ebphi, eePmap, ebdist, eeMmap, prof_eePL, prof_eePR, prof_eeML, prof_eeMR, ebBorderdist = makedist(coeff_barl, coeff_endc)

    gStyle.SetPalette(1)

    c =  TCanvas("CCdist")
    c.Divide(2,2)

    c.cd(1)  
    ebmap.Draw("colz")
    c.cd(2)
    eePmap.Draw("colz")
    c.cd(3)
    ebdist.Draw()
    c.cd(4)
    eeMmap.Draw("colz")

#    c.SaveAs(filename)

    prof_eb_eta = ebeta.ProfileX()
    prof_eb_phi = ebphi.ProfileX()

    c2 = TCanvas("CCprofiles")
    c2.Divide(2,2)

    leg = TLegend(0.1,0.7,0.48,0.9)

    c2.cd(1)
    prof_eb_eta.Draw()
    c2.cd(2)
    prof_eb_phi.Draw()
    c2.cd(3)
    prof_eePL.SetMarkerStyle(8)
    prof_eePL.Draw()
    prof_eePR.SetMarkerStyle(8)
    prof_eePR.SetMarkerColor(2)
    prof_eePR.Draw("same")
    prof_eeML.SetMarkerStyle(8)
    prof_eeML.SetMarkerColor(3)
    prof_eeML.Draw("same")
    prof_eeMR.SetMarkerStyle(8)
    prof_eeMR.SetMarkerColor(5)
    prof_eeMR.Draw("same")
    leg.AddEntry(prof_eePL,"Dee1","lp")
    leg.AddEntry(prof_eePR,"Dee2","lp")
    leg.AddEntry(prof_eeMR,"Dee3","lp")
    leg.AddEntry(prof_eeML,"Dee4","lp")
    leg.Draw()
    c2.cd(4)
    ebBorderdist.Draw()

    extrafilename = "profiles_"+filename
 #   c2.SaveAs(extrafilename)

    savefile.Write()      


def getToken(db,tag,since):
    ''' Return payload token for a given iov, tag, db'''
    try :
       iov = inspect.Iov(db,tag)
       iovlist = iov.list()
       for p in iovlist:
           tmpsince=p[1]
           if str(tmpsince)==str(since) :
               return p[0]
       print "Could not retrieve token for tag: " , tag, " since: ", since
       sys.exit(0)
       
    except Exception, er :
       print er


def getObject(db,tag,since):
    ''' Return payload object for a given iov, tag, db'''
    found=0
    try:
#       exec('import '+db.moduleName(tag)+' as Plug')  
       db.startReadOnlyTransaction()
       Plug = __import__(str(db.payloadModules(tag)[0]))
       print " getObject Plug"
       payload = Plug.Object(db)
       db.commitTransaction()
       listOfIovElem= [iovElem for iovElem in db.iov(tag).elements]
       print " getObject before loop"
       for elem in db.iov(tag).elements :       
           if str(elem.since())==str(since):
               found=1
               print " getObject found ", elem.since()
#               return Plug.Object(elem)
               return elem
           
    except Exception, er :
        print " getObject exception ",er

    if not found :
        print "Could not retrieve payload for tag: " , tag, " since: ", since
        sys.exit(0)


def makedist(coeff_barl, coeff_endc) :

    ebmap = TH2F("EB","EB",360,1,261,171, -85,86)
    eePmap = TH2F("EE","EE",100, 1,101,100,1,101)
    eeMmap = TH2F("EEminus","EEminus",100,1,101,100,1,101)
    ebdist = TH1F("EBdist","EBdist",100,-2,2)
    ebBorderdist = TH1F("EBBorderdist","EBBorderdist",100,-2,2)

    ebeta = TH2F("ebeta","ebeta",171,-85,86,100,-2,2)
    ebphi = TH2F("ebphi","ebphi",360,1,361,100,-2,2)

    eePL = TH2F("EEPL","EEPlus Left",50,10,55,100,-2,2)
    eePR = TH2F("EEPR","EEPlus Right",50,10,55,100,-2,2)
    eeML = TH2F("EEML","EEMinus Left",50,10,55,100,-2,2)
    eeMR = TH2F("EEMR","EEMinus Right",50,10,55,100,-2,2)
    
    for i,c in enumerate(coeff_barl):
        ieta,iphi = EcalPyUtils.unhashEBIndex(i)
        ebmap.Fill(iphi,ieta,c)
        ebdist.Fill(c)
        ebeta.Fill(ieta,c)
        ebphi.Fill(iphi,c)

        if (abs(ieta)==85 or abs(ieta)==65 or abs(ieta)==64 or abs(ieta)==45 or abs(ieta)==44 or abs(ieta)==25 or abs(ieta)==24 or abs(ieta)==1 or iphi%20==1 or iphi%20==0):
            ebBorderdist.Fill(c)


    for i,c in enumerate(coeff_endc):
        ix,iy,iz = EcalPyUtils.unhashEEIndex(i)
        R = sqrt((ix-50)*(ix-50)+(iy-50)*(iy-50))

        if  iz>0:
            eePmap.Fill(ix,iy,c)
            if ix<50:
                eePL.Fill(R,c,1)
            if ix>50:
                eePR.Fill(R,c,1)

        if iz<0:
            eeMmap.Fill(ix,iy,c)
            if ix<50:
                eeML.Fill(R,c,1)
            if ix>50:
                eeMR.Fill(R,c,1)

    prof_eePL = eePL.ProfileX()
    prof_eePR = eePR.ProfileX()
    prof_eeML = eeML.ProfileX()
    prof_eeMR = eeMR.ProfileX()
    
    return ebmap, ebeta, ebphi, eePmap, ebdist, eeMmap, prof_eePL, prof_eePR, prof_eeML, prof_eeMR, ebBorderdist

def compareBarrel(coeff_barl_1,coeff_barl_2) :
  '''Return an histogram and a map of the differences '''

  diff_distro_h   = TH1F("diffh","diffh",100,-2,2)
  diff_distro_m   = TH2F("diffm","diffm",360,1,361,171,-85,86)
  diff_distro_m.SetStats(0)
  ebeta = TH2F("ebeta","ebeta",171,-85,86,100,-2,2)
  ebphi = TH2F("ebphi","ebphi",360,1,361,100,-2,2)

  
  for i,c in enumerate(coeff_barl_1):  
      diff = c - coeff_barl_2[i]      
      ieta,iphi= EcalPyUtils.unhashEBIndex(i)
      diff_distro_h.Fill(diff) 
      diff_distro_m.Fill(iphi,ieta,diff)
      ebeta.Fill(ieta,diff)
      ebphi.Fill(iphi,diff)

  prof_x_h = ebeta.ProfileX()
  prof_y_h = ebphi.ProfileX()
          
  return diff_distro_h, diff_distro_m, prof_x_h, prof_y_h



def compareBarrelBorder(coeff_barl_1,coeff_barl_2) :
  '''Return an histogram and a map of the differences '''

  diff_distro_border_h   = TH1F("diffborderh","diffh",100,-2,2)
  diff_distro_border_m   = TH2F("diffborderm","diffm",360,1,361,171,-85,86)
  diff_distro_border_m.SetStats(0)
  
  for i,c in enumerate(coeff_barl_1):  
      diff = c - coeff_barl_2[i]      
      ieta,iphi= EcalPyUtils.unhashEBIndex(i)
      if (abs(ieta)==85 or abs(ieta)==65 or abs(ieta)==64 or abs(ieta)==45 or abs(ieta)==44 or abs(ieta)==25 or abs(ieta)==24 or abs(ieta)==1 or iphi%20==1 or iphi%20==0):
          diff_distro_border_h.Fill(diff) 
      if (abs(ieta)==85 or abs(ieta)==65 or abs(ieta)==64 or abs(ieta)==45 or abs(ieta)==44 or abs(ieta)==25 or abs(ieta)==24 or abs(ieta)==1 or iphi%20==0 or iphi%20==1): 
          diff_distro_border_m.Fill(iphi,ieta,diff)
          
  return diff_distro_border_h, diff_distro_border_m 




    
def compareEndcap(coeff_endc_1, coeff_endc_2) :
    ''' Return an histogram and a map of the differences for each endcap'''

    diff_distro_h_eep   = TH1F("diff EE+","diff EE+",100,-2,2)
    diff_distro_h_eem   = TH1F("diff EE-","diff EE-",100,-2,2)

    
    diff_distro_m_eep   = TH2F("map EE+","map EE+",100,1,101,100,1,101)
    diff_distro_m_eem   = TH2F("map EE-","map EE-",100,1,101,100,1,101)

    temp_h = TH1F("tempR","tempR",50,0,50)
    
    diff_distro_m_eep.SetStats(0)
    diff_distro_m_eem.SetStats(0)


    for i,c in enumerate(coeff_endc_1):  
      diff = c - coeff_endc_2[i]
      ix,iy,iz = EcalPyUtils.unhashEEIndex(i)
      R = sqrt((ix-50)*(ix-50)+(iy-50)*(iy-50))
      
      if iz >0:
          diff_distro_h_eep.Fill(diff)
          diff_distro_m_eep.Fill(ix,iy,diff)
          
      else:
          diff_distro_h_eem.Fill(diff)
          diff_distro_m_eem.Fill(ix,iy,diff)

    return diff_distro_h_eep, \
           diff_distro_m_eep, \
           diff_distro_h_eem, \
           diff_distro_m_eem



def compareEndcapBorder(coeff_endc_1, coeff_endc_2) :
    ''' Return an histogram and a map of the differences for each endcap'''

    border_diff_distro_h_eep   = TH1F("borderdiff EE+","diff EE+",100,-2,2)
    border_diff_distro_h_eem   = TH1F("borderdiff EE-","diff EE-",100,-2,2)

    
    border_diff_distro_m_eep   = TH2F("bordermap EE+","map EE+",100,1,101,100,1,101)
    border_diff_distro_m_eem   = TH2F("bordermap EE-","map EE-",100,1,101,100,1,101)
    
    border_diff_distro_m_eep.SetStats(0)
    border_diff_distro_m_eem.SetStats(0)


    for i,c in enumerate(coeff_endc_1):  
      diff = c - coeff_endc_2[i]
      ix,iy,iz = EcalPyUtils.unhashEEIndex(i)
      Rsq = ((ix-50.0)**2+(iy-50.0)**2)
      
      if (iz >0 and (Rsq<144.0 or Rsq>2500.0)):
          border_diff_distro_h_eep.Fill(diff)
          border_diff_distro_m_eep.Fill(ix,iy,diff)
      elif (iz<0 and (Rsq<144.0 or Rsq>2500.0)):
          border_diff_distro_h_eem.Fill(diff)
          border_diff_distro_m_eem.Fill(ix,iy,diff)
      

    return border_diff_distro_h_eep, \
           border_diff_distro_m_eep, \
           border_diff_distro_h_eem, \
           border_diff_distro_m_eem
