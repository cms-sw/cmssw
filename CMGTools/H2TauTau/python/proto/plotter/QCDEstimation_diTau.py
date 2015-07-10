import imp
import math
from CMGTools.RootTools.RootInit import *
from CMGTools.H2TauTau.proto.HistogramSet import histogramSet
from CMGTools.H2TauTau.proto.plotter.H2TauTauDataMC_diTau import *

def substractMCbackground(QCDtightSS, plotVarDataSS):
      QCDtightSS.Add(plotVarDataSS.Hist("DYJets")   ,-1)
      QCDtightSS.Add(plotVarDataSS.Hist("DYJets_ZJ"),-1)
      #QCDtightSS.Add(plotVarDataSS.Hist("DYJets_Photon"),-1)
      QCDtightSS.Add(plotVarDataSS.Hist("DYJets_ZL"),-1)
      QCDtightSS.Add(plotVarDataSS.Hist("WJets")    ,-1)
      #QCDtightSS.Add(plotVarDataSS.Hist("WJets_Fakes"),-1)
      #QCDtightSS.Add(plotVarDataSS.Hist("WW"),-1)
      #QCDtightSS.Add(plotVarDataSS.Hist("WZ"),-1)
      #QCDtightSS.Add(plotVarDataSS.Hist("ZZ"),-1)
      #QCDtightSS.Add(plotVarDataSS.Hist("TTJets"),-1)
      QCDtightSS.Add(plotVarDataSS.Hist("TTJetsFullLept"),-1)
      QCDtightSS.Add(plotVarDataSS.Hist("TTJetsSemiLept"),-1)
      QCDtightSS.Add(plotVarDataSS.Hist("TTJetsHadronic"),-1)
      try :
        QCDtightSS.Add(plotVarDataSS.Hist("TTJets_emb")    ,-1)
      except :
        pass
      QCDtightSS.Add(plotVarDataSS.Hist('WWJetsTo2L2Nu') ,-1)
      QCDtightSS.Add(plotVarDataSS.Hist('WZJetsTo2L2Q')  ,-1)
      QCDtightSS.Add(plotVarDataSS.Hist('WZJetsTo3LNu')  ,-1)
      QCDtightSS.Add(plotVarDataSS.Hist('ZZJetsTo4L')    ,-1)
      QCDtightSS.Add(plotVarDataSS.Hist('ZZJetsTo2L2Nu') ,-1)
      QCDtightSS.Add(plotVarDataSS.Hist('ZZJetsTo2L2Q')  ,-1)
      QCDtightSS.Add(plotVarDataSS.Hist("Tbar_tW")       ,-1)
      QCDtightSS.Add(plotVarDataSS.Hist("T_tW")          ,-1)

def QCDEstimate(prefix,prefix1,xmin,xmax,\
                plotVarDataSS, plotVarDataLooseIsoOS, plotVarDataLooseIsoSS,\
                plotVarDataLowControlOS, plotVarDataLowControlSS, plotVarDataAverageControlOS,\
                plotVarDataAverageControlSS, plotVarDataHighControlOS, plotVarDataHighControlSS,\
                plotVarDataLowControlLooseIsoOS, plotVarDataLowControlLooseIsoSS,\
                plotVarDataAverageControlLooseIsoOS, plotVarDataAverageControlLooseIsoSS,\
                plotVarDataHighControlLooseIsoOS, plotVarDataHighControlLooseIsoSS,\
                log):
      ymax=plotVarDataLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoSS.varName+"_qcdLooseSS.pdf")
      gPad.WaitPrimitive()
      
      ymax=plotVarDataSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataSS.varName+"_qcdTightSS.pdf")
      gPad.WaitPrimitive()
      
      ymax=plotVarDataAverageControlOS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataAverageControlOS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataAverageControlOS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataAverageControlOS.varName+"_qcdAverageControlOS.pdf")
      gPad.WaitPrimitive()
      
      ymax=plotVarDataAverageControlLooseIsoOS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataAverageControlLooseIsoOS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataAverageControlLooseIsoOS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataAverageControlLooseIsoOS.varName+"_qcdAverageControlLooseIsoOS.pdf")
      gPad.WaitPrimitive()
      
      ymax=plotVarDataAverageControlLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataAverageControlLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataAverageControlLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataAverageControlLooseIsoSS.varName+"_qcdAverageControlLooseIsoSS.pdf")
      gPad.WaitPrimitive()
      
      QCDtightSS=copy.deepcopy(plotVarDataSS.Hist("Data"))
      substractMCbackground(QCDtightSS, plotVarDataSS)

      QCDlooseSS=copy.deepcopy(plotVarDataLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseSS, plotVarDataLooseIsoSS)

      QCDlooseOS=copy.deepcopy(plotVarDataLooseIsoOS.Hist("Data"))
      substractMCbackground(QCDlooseOS, plotVarDataLooseIsoOS)

      QCDlooseLowControlOS=copy.deepcopy(plotVarDataLowControlOS.Hist("Data"))
      substractMCbackground(QCDlooseLowControlOS, plotVarDataLowControlOS)

      QCDlooseLowControlSS=copy.deepcopy(plotVarDataLowControlSS.Hist("Data"))
      substractMCbackground(QCDlooseLowControlSS, plotVarDataLowControlSS)

      QCDlooseAverageControlOS=copy.deepcopy(plotVarDataAverageControlOS.Hist("Data"))
      substractMCbackground(QCDlooseAverageControlOS, plotVarDataAverageControlOS)

      QCDlooseAverageControlSS=copy.deepcopy(plotVarDataAverageControlSS.Hist("Data"))
      substractMCbackground(QCDlooseAverageControlSS, plotVarDataAverageControlSS)

      QCDlooseHighControlOS=copy.deepcopy(plotVarDataHighControlOS.Hist("Data"))
      substractMCbackground(QCDlooseHighControlOS, plotVarDataHighControlOS)

      QCDlooseHighControlSS=copy.deepcopy(plotVarDataHighControlSS.Hist("Data"))
      substractMCbackground(QCDlooseHighControlSS, plotVarDataHighControlSS)

      QCDlooseLowControlLooseIsoOS=copy.deepcopy(plotVarDataLowControlLooseIsoOS.Hist("Data"))
      substractMCbackground(QCDlooseLowControlLooseIsoOS, plotVarDataLowControlLooseIsoOS)

      QCDlooseLowControlLooseIsoSS=copy.deepcopy(plotVarDataLowControlLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseLowControlLooseIsoSS, plotVarDataLowControlLooseIsoSS)

      QCDlooseAverageControlLooseIsoOS=copy.deepcopy(plotVarDataAverageControlLooseIsoOS.Hist("Data"))
      substractMCbackground(QCDlooseAverageControlLooseIsoOS, plotVarDataAverageControlLooseIsoOS)

      QCDlooseAverageControlLooseIsoSS=copy.deepcopy(plotVarDataAverageControlLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseAverageControlLooseIsoSS, plotVarDataAverageControlLooseIsoSS)

      QCDlooseHighControlLooseIsoOS=copy.deepcopy(plotVarDataHighControlLooseIsoOS.Hist("Data"))
      substractMCbackground(QCDlooseHighControlLooseIsoOS, plotVarDataHighControlLooseIsoOS)

      QCDlooseHighControlLooseIsoSS=copy.deepcopy(plotVarDataHighControlLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseHighControlLooseIsoSS, plotVarDataHighControlLooseIsoSS)

      #print "QCDlooseSS:", QCDlooseSS.Integral()
      #print "QCDtightSS:", QCDtightSS.Integral()
      tightLoose=QCDtightSS.Integral()/QCDlooseSS.Integral()
      tightLooseErr=tightLoose*math.sqrt(1./QCDtightSS.Integral() + 1./QCDlooseSS.Integral())
      print "QCDtightSS / QCDlooseSS", tightLoose, "+-", tightLooseErr
      
      #print "QCDlooseLowControlLooseIsoOS:", QCDlooseLowControlLooseIsoOS.Integral()
      #print "QCDlooseLowControlLooseIsoSS:", QCDlooseLowControlLooseIsoSS.Integral()
      osSSl=QCDlooseLowControlLooseIsoOS.Integral()/QCDlooseLowControlLooseIsoSS.Integral()
      osSSlErr=osSSl*math.sqrt(1./QCDlooseLowControlLooseIsoOS.Integral() + 1./QCDlooseLowControlLooseIsoSS.Integral())
      print "QCDlooseLowControlLooseIsoOS / QCDlooseLowControlLooseIsoSS",osSSl, "+-", osSSlErr
      
      #print "QCDlooseAverageControlLooseIsoOS:", QCDlooseAverageControlLooseIsoOS.Integral()
      #print "QCDlooseAverageControlLooseIsoSS:", QCDlooseAverageControlLooseIsoSS.Integral()
      osSSm=QCDlooseAverageControlLooseIsoOS.Integral()/QCDlooseAverageControlLooseIsoSS.Integral()
      osSSmErr=osSSm*math.sqrt(1./QCDlooseAverageControlLooseIsoOS.Integral() + 1./QCDlooseAverageControlLooseIsoSS.Integral())
      print "QCDlooseAverageControlLooseIsoOS / QCDlooseAverageControlLooseIsoSS",osSSm, "+-", osSSmErr
      
      #print "QCDlooseHighControlLooseIsoOS:", QCDlooseHighControlLooseIsoOS.Integral()
      #print "QCDlooseHighControlLooseIsoSS:", QCDlooseHighControlLooseIsoSS.Integral()
      osSSh=QCDlooseHighControlLooseIsoOS.Integral()/QCDlooseHighControlLooseIsoSS.Integral()
      osSShErr=osSSh*math.sqrt(1./QCDlooseHighControlLooseIsoOS.Integral() + 1./QCDlooseHighControlLooseIsoSS.Integral())
      print "QCDlooseHighControlLooseIsoOS / QCDlooseHighControlLooseIsoSS",osSSh, "+-", osSShErr
      
      #print "QCDlooseLowControlLooseIsoSS:", QCDlooseLowControlLooseIsoSS.Integral()
      tightLoosel=QCDlooseLowControlSS.Integral()/QCDlooseLowControlLooseIsoSS.Integral()
      tightLooselErr=tightLoosel*math.sqrt(1./QCDlooseLowControlSS.Integral() + 1./QCDlooseLowControlLooseIsoSS.Integral())
      print "QCDlooseLowControlSS / QCDlooseLowControlLooseIsoSS",tightLoosel, "+-", tightLooselErr
      
      #print "QCDlooseAverageControlLooseIsoSS:", QCDlooseAverageControlLooseIsoSS.Integral()
      tightLoosem=QCDlooseAverageControlSS.Integral()/QCDlooseAverageControlLooseIsoSS.Integral()
      tightLoosemErr=tightLoosem*math.sqrt(1./QCDlooseAverageControlSS.Integral() + 1./QCDlooseAverageControlLooseIsoSS.Integral())
      print "QCDlooseAverageControlSS / QCDlooseAverageControlLooseIsoSS",tightLoosem, "+-", tightLoosemErr
      
      #print "QCDlooseHighControlLooseIsoSS:", QCDlooseHighControlLooseIsoSS.Integral()
      tightLooseh=QCDlooseHighControlSS.Integral()/QCDlooseHighControlLooseIsoSS.Integral()
      tightLoosehErr=tightLooseh*math.sqrt(1./QCDlooseHighControlSS.Integral() + 1./QCDlooseHighControlLooseIsoSS.Integral())
      print "QCDlooseHighControlSS / QCDlooseHighControlLooseIsoSS",tightLooseh, "+-", tightLoosehErr

      #QCDScale=tightLoosem*osSSm
      
      tightOSLooseSSl=QCDlooseLowControlOS.Integral()/QCDlooseLowControlLooseIsoSS.Integral()
      tightOSLooseSSlErr=tightOSLooseSSl*math.sqrt(1./QCDlooseLowControlOS.Integral() + 1./QCDlooseLowControlLooseIsoSS.Integral())
      print "QCDlooseLowControlOS / QCDlooseLowControlLooseIsoSS",tightOSLooseSSl, "+-", tightOSLooseSSlErr
      
      tightOSLooseSSm=QCDlooseAverageControlOS.Integral()/QCDlooseAverageControlLooseIsoSS.Integral()
      tightOSLooseSSmErr=tightOSLooseSSm*math.sqrt(1./QCDlooseAverageControlOS.Integral() + 1./QCDlooseAverageControlLooseIsoSS.Integral())
      print "QCDlooseAverageControlOS / QCDlooseAverageControlLooseIsoSS",tightOSLooseSSm, "+-", tightOSLooseSSmErr
      
      tightOSLooseSSh=QCDlooseHighControlOS.Integral()/QCDlooseHighControlLooseIsoSS.Integral()
      tightOSLooseSShErr=tightOSLooseSSh*math.sqrt(1./QCDlooseHighControlOS.Integral() + 1./QCDlooseHighControlLooseIsoSS.Integral())
      print "QCDlooseHighControlOS / QCDlooseHighControlLooseIsoSS",tightOSLooseSSh, "+-", tightOSLooseSShErr
      
      if tightOSLooseSSm<tightOSLooseSSmErr/2.:
          tightOSLooseSSm=tightOSLooseSSmErr/2.
      
      QCDScale=tightOSLooseSSm
      
      return QCDlooseSS,QCDScale

def QCDEstimate1(prefix,prefix1,xmin,xmax,\
                 plotVarDataSS, plotVarDataLooseIsoOS, plotVarDataLooseIsoSS,\
                 plotVarDataLowControlOS, plotVarDataLowControlSS, plotVarDataAverageControlOS,\
                 plotVarDataAverageControlSS, plotVarDataHighControlOS, plotVarDataHighControlSS,\
                 plotVarDataLowControlLooseIsoOS, plotVarDataLowControlLooseIsoSS,\
                 plotVarDataAverageControlLooseIsoOS, plotVarDataAverageControlLooseIsoSS,\
                 plotVarDataHighControlLooseIsoOS, plotVarDataHighControlLooseIsoSS,\
                 log):
      ymax=plotVarDataLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoSS.varName+"_qcdLooseSS.pdf")
      #gPad.WaitPrimitive()
      
      ymax=plotVarDataSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataSS.varName+"_qcdTightSS.pdf")
      #gPad.WaitPrimitive()
      
      #ymax=plotVarDataAverageControlOS.Hist("Data").GetMaximum()*1.5
      #if log:
        #  plotVarDataAverageControlOS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	#  gPad.SetLogy()
      #else:
        #  plotVarDataAverageControlOS.DrawStack("HIST",xmin,xmax,0,ymax)
	#  gPad.SetLogy(False)
      #gPad.SaveAs(prefix1+prefix+'_'+plotVarDataAverageControlOS.varName+"_qcdAverageControlOS.pdf")
      #gPad.WaitPrimitive()
      
      ymax=plotVarDataAverageControlLooseIsoOS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataAverageControlLooseIsoOS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataAverageControlLooseIsoOS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataAverageControlLooseIsoOS.varName+"_qcdAverageControlLooseIsoOS.pdf")
      #gPad.WaitPrimitive()
      
      ymax=plotVarDataAverageControlLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataAverageControlLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataAverageControlLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataAverageControlLooseIsoSS.varName+"_qcdAverageControlLooseIsoSS.pdf")
      #gPad.WaitPrimitive()


      QCDtightSS=copy.deepcopy(plotVarDataSS.Hist("Data"))
      substractMCbackground(QCDtightSS, plotVarDataSS)

      QCDlooseSS=copy.deepcopy(plotVarDataLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseSS, plotVarDataLooseIsoSS)

      #QCDlooseOS=copy.deepcopy(plotVarDataLooseIsoOS.Hist("Data"))
      #substractMCbackground(QCDlooseOS, plotVarDataLooseIsoOS)

      #QCDlooseLowControlOS=copy.deepcopy(plotVarDataLowControlOS.Hist("Data"))
      #substractMCbackground(QCDlooseLowControlOS, plotVarDataLowControlOS)

      #QCDlooseLowControlSS=copy.deepcopy(plotVarDataLowControlSS.Hist("Data"))
      #substractMCbackground(QCDlooseLowControlSS, plotVarDataLowControlSS)

      #QCDlooseAverageControlOS=copy.deepcopy(plotVarDataAverageControlOS.Hist("Data"))
      #substractMCbackground(QCDlooseAverageControlOS, plotVarDataAverageControlOS)

      #QCDlooseAverageControlSS=copy.deepcopy(plotVarDataAverageControlSS.Hist("Data"))
      #substractMCbackground(QCDlooseAverageControlSS, plotVarDataAverageControlSS)

      #QCDlooseHighControlOS=copy.deepcopy(plotVarDataHighControlOS.Hist("Data"))
      #substractMCbackground(QCDlooseHighControlOS, plotVarDataHighControlOS)

      #QCDlooseHighControlSS=copy.deepcopy(plotVarDataHighControlSS.Hist("Data"))
      #substractMCbackground(QCDlooseHighControlSS, plotVarDataHighControlSS)

      #QCDlooseLowControlLooseIsoOS=copy.deepcopy(plotVarDataLowControlLooseIsoOS.Hist("Data"))
      #substractMCbackground(QCDlooseLowControlLooseIsoOS, plotVarDataLowControlLooseIsoOS)

      #QCDlooseLowControlLooseIsoSS=copy.deepcopy(plotVarDataLowControlLooseIsoSS.Hist("Data"))
      #substractMCbackground(QCDlooseLowControlLooseIsoSS, plotVarDataLowControlLooseIsoSS)

      QCDlooseAverageControlLooseIsoOS=copy.deepcopy(plotVarDataAverageControlLooseIsoOS.Hist("Data"))
      substractMCbackground(QCDlooseAverageControlLooseIsoOS, plotVarDataAverageControlLooseIsoOS)

      QCDlooseAverageControlLooseIsoSS=copy.deepcopy(plotVarDataAverageControlLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseAverageControlLooseIsoSS, plotVarDataAverageControlLooseIsoSS)

      #QCDlooseHighControlLooseIsoOS=copy.deepcopy(plotVarDataHighControlLooseIsoOS.Hist("Data"))
      #substractMCbackground(QCDlooseHighControlLooseIsoOS, plotVarDataHighControlLooseIsoOS)

      #QCDlooseHighControlLooseIsoSS=copy.deepcopy(plotVarDataHighControlLooseIsoSS.Hist("Data"))
      #substractMCbackground(QCDlooseHighControlLooseIsoSS, plotVarDataHighControlLooseIsoSS)

      #print "QCDlooseSS:", QCDlooseSS.Integral()
      #print "QCDtightSS:", QCDtightSS.Integral()
      tightLoose=QCDtightSS.Integral()/QCDlooseSS.Integral()
      tightLooseErr=tightLoose*math.sqrt(1./QCDtightSS.Integral() + 1./QCDlooseSS.Integral())
      print "QCDtightSS / QCDlooseSS", tightLoose, "+-", tightLooseErr
      
      #print "QCDlooseLowControlLooseIsoOS:", QCDlooseLowControlLooseIsoOS.Integral()
      #print "QCDlooseLowControlLooseIsoSS:", QCDlooseLowControlLooseIsoSS.Integral()
      #osSSl=QCDlooseLowControlLooseIsoOS.Integral()/QCDlooseLowControlLooseIsoSS.Integral()
      #osSSlErr=osSSl*math.sqrt(1./QCDlooseLowControlLooseIsoOS.Integral() + 1./QCDlooseLowControlLooseIsoSS.Integral())
      #print "QCDlooseLowControlLooseIsoOS / QCDlooseLowControlLooseIsoSS",osSSl, "+-", osSSlErr
      
      #print "QCDlooseAverageControlLooseIsoOS:", QCDlooseAverageControlLooseIsoOS.Integral()
      #print "QCDlooseAverageControlLooseIsoSS:", QCDlooseAverageControlLooseIsoSS.Integral()
      osSSm=QCDlooseAverageControlLooseIsoOS.Integral()/QCDlooseAverageControlLooseIsoSS.Integral()
      osSSmErr=osSSm*math.sqrt(1./QCDlooseAverageControlLooseIsoOS.Integral() + 1./QCDlooseAverageControlLooseIsoSS.Integral())
      print "QCDlooseAverageControlLooseIsoOS / QCDlooseAverageControlLooseIsoSS",osSSm, "+-", osSSmErr
      
      #print "QCDlooseHighControlLooseIsoOS:", QCDlooseHighControlLooseIsoOS.Integral()
      #print "QCDlooseHighControlLooseIsoSS:", QCDlooseHighControlLooseIsoSS.Integral()
      #osSSh=QCDlooseHighControlLooseIsoOS.Integral()/QCDlooseHighControlLooseIsoSS.Integral()
      #osSShErr=osSSh*math.sqrt(1./QCDlooseHighControlLooseIsoOS.Integral() + 1./QCDlooseHighControlLooseIsoSS.Integral())
      #print "QCDlooseHighControlLooseIsoOS / QCDlooseHighControlLooseIsoSS",osSSh, "+-", osSShErr
      
      if osSSm<osSSmErr/2.:
          osSSm=osSSmErr/2.
      
      if tightLoose<tightLooseErr/2.:
          tightLoose=tightLooseErr/2.
      
      QCDScale=osSSm*tightLoose
      
      return QCDlooseSS,QCDScale

def QCDEstimate2(prefix,prefix1,var,xmin,xmax,\
                 plotVarDataSS, plotVarDataLooseIsoOS, plotVarDataLooseIsoOSMC, plotVarDataLooseIsoSS,\
                 log):
      ymax=plotVarDataLooseIsoOS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoOS.varName+"_qcdLooseOS.pdf")
      
      ymax=plotVarDataLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoSS.varName+"_qcdLooseSS.pdf")
      
      ymax=plotVarDataSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataSS.varName+"_qcdTightSS.pdf")
      
      QCDlooseOS = copy.deepcopy(plotVarDataLooseIsoOS.Hist("Data"))
      substractMCbackground(QCDlooseOS, plotVarDataLooseIsoOSMC)

      QCDlooseSS = copy.deepcopy(plotVarDataLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseSS, plotVarDataLooseIsoSS)

      QCDtightSS = copy.deepcopy(plotVarDataSS.Hist("Data"))
      substractMCbackground(QCDtightSS, plotVarDataSS)

      QCD_PrintOut = open( os.getcwd()+"/"+prefix+"/QCD_PrintOut_"+var+".txt","w")
      print >> QCD_PrintOut, "QCDlooseOS:", QCDlooseOS.Integral()
      print >> QCD_PrintOut, "QCDlooseSS:", QCDlooseSS.Integral()
      print >> QCD_PrintOut, "QCDtightSS:", QCDtightSS.Integral()
      if QCDlooseSS.Integral()>0:
          tightLoose    = QCDtightSS.Integral()/QCDlooseSS.Integral()
          tightLooseErr = tightLoose*math.sqrt(1./abs(QCDtightSS.Integral()) + 1./abs(QCDlooseSS.Integral()))
      else:
          tightLoose    = 1
          tightLooseErr = 1
      print >> QCD_PrintOut, "QCDtightSS / QCDlooseSS", tightLoose, "+-", tightLooseErr
      QCD_PrintOut.close()
      
      if tightLoose<tightLooseErr/2.:
          tightLoose=tightLooseErr/2.
      
      QCDScale=tightLoose
      
      return QCDlooseOS, QCDScale, QCDlooseSS, QCDtightSS, tightLoose, tightLooseErr/tightLoose


def QCD_WJets_Estimate2(prefix,prefix1,var,xmin,xmax,\
                        plotVarDataSS, plotVarDataLooseIsoOS, plotVarDataLooseIsoSS,\
                        log):
      ymax=plotVarDataLooseIsoOS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoOS.varName+"_qcd_wjets_LooseOS.pdf")
      
      ymax=plotVarDataLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoSS.varName+"_qcd_wjets_LooseSS.pdf")
      
      ymax=plotVarDataSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataSS.varName+"_qcd_wjets_TightSS.pdf")
      
      QCDlooseOS=copy.deepcopy(plotVarDataLooseIsoOS.Hist("Data"))
      substractMCbackground2(QCDlooseOS, plotVarDataLooseIsoOS)

      QCDlooseSS=copy.deepcopy(plotVarDataLooseIsoSS.Hist("Data"))
      substractMCbackground2(QCDlooseSS, plotVarDataLooseIsoSS)

      QCDtightSS=copy.deepcopy(plotVarDataSS.Hist("Data"))
      substractMCbackground2(QCDtightSS, plotVarDataSS)

      QCD_PrintOut = open( os.getcwd()+"/"+prefix+"/QCD_PrintOut_"+var+".txt","w")
      print >> QCD_PrintOut, "QCDlooseOS:", QCDlooseOS.Integral()
      print >> QCD_PrintOut, "QCDlooseSS:", QCDlooseSS.Integral()
      print >> QCD_PrintOut, "QCDtightSS:", QCDtightSS.Integral()
      tightLoose=QCDtightSS.Integral()/QCDlooseSS.Integral()
      tightLooseErr=tightLoose*math.sqrt(1./abs(QCDtightSS.Integral()) + 1./abs(QCDlooseSS.Integral()))
      print >> QCD_PrintOut, "QCDtightSS / QCDlooseSS", tightLoose, "+-", tightLooseErr
      QCD_PrintOut.close()
      
      if tightLoose<tightLooseErr/2.:
          tightLoose=tightLooseErr/2.
      
      QCDScale=tightLoose
      
      return QCDlooseOS, QCDScale, QCDlooseSS, QCDtightSS


def QCDEstimate3(prefix,prefix1,xmin,xmax,\
                 plotVarDataLooseIsoOS, plotVarDataLooseIsoSS,\
                 plotVarDataAverageControlSS,\
                 plotVarDataAverageControlLooseIsoSS,\
                 log):
      ymax=plotVarDataLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoSS.varName+"_qcdLooseSS.pdf")
      #gPad.WaitPrimitive()
      
      ymax=plotVarDataLooseIsoOS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoOS.varName+"_qcdLooseOS.pdf")
      #gPad.WaitPrimitive()
      
      ymax=plotVarDataAverageControlSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataAverageControlSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataAverageControlSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataAverageControlSS.varName+"_qcdAverageControlSS.pdf")
      #gPad.WaitPrimitive()
      
      ymax=plotVarDataAverageControlLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataAverageControlLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataAverageControlLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataAverageControlLooseIsoSS.varName+"_qcdAverageControlLooseIsoSS.pdf")
      #gPad.WaitPrimitive()


      QCDlooseOS=copy.deepcopy(plotVarDataLooseIsoOS.Hist("Data"))
      substractMCbackground(QCDlooseOS, plotVarDataLooseIsoOS)

      QCDlooseSS=copy.deepcopy(plotVarDataLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseSS, plotVarDataLooseIsoSS)

      QCDlooseAverageControlSS=copy.deepcopy(plotVarDataAverageControlSS.Hist("Data"))
      substractMCbackground(QCDlooseAverageControlSS, plotVarDataAverageControlSS)

      QCDlooseAverageControlLooseIsoSS=copy.deepcopy(plotVarDataAverageControlLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseAverageControlLooseIsoSS, plotVarDataAverageControlLooseIsoSS)

      tightLoose=QCDlooseAverageControlSS.Integral()/QCDlooseAverageControlLooseIsoSS.Integral()
      tightLooseErr=tightLoose*math.sqrt(1./QCDlooseAverageControlSS.Integral() + 1./QCDlooseAverageControlLooseIsoSS.Integral())
      print "QCDlooseAverageControlSS / QCDlooseAverageControlLooseIsoSS", tightLoose, "+-", tightLooseErr
      
      osSSm=QCDlooseOS.Integral()/QCDlooseSS.Integral()
      osSSmErr=osSSm*math.sqrt(1./QCDlooseOS.Integral() + 1./QCDlooseSS.Integral())
      print "QCDlooseOS / QCDlooseSS",osSSm, "+-", osSSmErr
      
      #print "QCDlooseHighControlLooseIsoOS:", QCDlooseHighControlLooseIsoOS.Integral()
      #print "QCDlooseHighControlLooseIsoSS:", QCDlooseHighControlLooseIsoSS.Integral()
      #osSSh=QCDlooseHighControlLooseIsoOS.Integral()/QCDlooseHighControlLooseIsoSS.Integral()
      #osSShErr=osSSh*math.sqrt(1./QCDlooseHighControlLooseIsoOS.Integral() + 1./QCDlooseHighControlLooseIsoSS.Integral())
      #print "QCDlooseHighControlLooseIsoOS / QCDlooseHighControlLooseIsoSS",osSSh, "+-", osSShErr
      
      if osSSm<osSSmErr/2.:
          osSSm=osSSmErr/2.
      
      if tightLoose<tightLooseErr/2.:
          tightLoose=tightLooseErr/2.
      
      QCDScale=osSSm*tightLoose
      
      return QCDlooseSS,QCDScale

def QCDEstimate4(prefix,prefix1,xmin,xmax,\
                 plotVarDataSS, plotVarDataLooseIsoOS, plotVarDataLooseIsoSS, plotVarDataSemiLooseIsoSS,\
                 log):
      ymax=plotVarDataLooseIsoOS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoOS.varName+"_qcdLooseOS.pdf")
      #gPad.WaitPrimitive()
      
      ymax=plotVarDataLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoSS.varName+"_qcdLooseSS.pdf")
      #gPad.WaitPrimitive()
      
      ymax=plotVarDataSemiLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataSemiLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataSemiLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataSemiLooseIsoSS.varName+"_qcdSemiLooseSS.pdf")
      #gPad.WaitPrimitive()
      
      ymax=plotVarDataSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataSS.varName+"_qcdTightSS.pdf")
      #gPad.WaitPrimitive()
      
      QCDlooseOS=copy.deepcopy(plotVarDataLooseIsoOS.Hist("Data"))
      substractMCbackground(QCDlooseOS, plotVarDataLooseIsoOS)

      QCDlooseSS=copy.deepcopy(plotVarDataLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseSS, plotVarDataLooseIsoSS)

      QCDtightSS=copy.deepcopy(plotVarDataSS.Hist("Data"))
      substractMCbackground(QCDtightSS, plotVarDataSS)

      print "QCDlooseSS:", QCDlooseSS.Integral()
      print "QCDtightSS:", QCDtightSS.Integral()
      tightLoose=QCDtightSS.Integral()/QCDlooseSS.Integral()
      tightLooseErr=tightLoose*math.sqrt(1./abs(QCDtightSS.Integral()) + 1./abs(QCDlooseSS.Integral()))
      print "QCDtightSS / QCDlooseSS", tightLoose, "+-", tightLooseErr
      
      if tightLoose<tightLooseErr/2.:
          tightLoose=tightLooseErr/2.
      
      QCDScale=tightLoose
      
      return QCDlooseOS,QCDScale

def QCDEstimate5(prefix,prefix1,xmin,xmax,\
                 plotVarDataSS, plotVarDataLooseIsoOS, plotVarDataLooseIsoSS, plotVarDataSemiLooseIsoSS,\
                 log):
      ymax=plotVarDataLooseIsoOS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoOS.varName+"_qcdLooseOS.pdf")
      #gPad.WaitPrimitive()
      
      ymax=plotVarDataLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoSS.varName+"_qcdLooseSS.pdf")
      #gPad.WaitPrimitive()
      
      ymax=plotVarDataSemiLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataSemiLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataSemiLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataSemiLooseIsoSS.varName+"_qcdSemiLooseSS.pdf")
      #gPad.WaitPrimitive()
      
      ymax=plotVarDataSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataSS.varName+"_qcdTightSS.pdf")
      #gPad.WaitPrimitive()
      
      QCDlooseOS=copy.deepcopy(plotVarDataLooseIsoOS.Hist("Data"))
      substractMCbackground(QCDlooseOS, plotVarDataLooseIsoOS)

      QCDlooseSS=copy.deepcopy(plotVarDataLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseSS, plotVarDataLooseIsoSS)

      QCDtightSS=copy.deepcopy(plotVarDataSS.Hist("Data"))
      substractMCbackground(QCDtightSS, plotVarDataSS)

      print "QCDlooseSS:", QCDlooseSS.Integral()
      print "QCDlooseOS:", QCDlooseOS.Integral()
      OSSS=QCDlooseOS.Integral()/QCDlooseSS.Integral()
      OSSSErr=OSSS*math.sqrt(1./abs(QCDlooseOS.Integral()) + 1./abs(QCDlooseSS.Integral()))
      print "QCDlooseOS / QCDlooseSS", OSSS, "+-", OSSSErr
      
      if OSSS<OSSSErr/2.:
          OSSS=OSSSErr/2.
      
      QCDScale=OSSS
      
      return QCDtightSS,QCDScale


def QCDYieldError(prefix,prefix1,var,xmin,xmax,\
                 plotVarDataSS, plotVarDataLooseIsoSS,\
                 plotVarDataSS_sys1, plotVarDataLooseIsoSS_sys1,\
                 plotVarDataSS_sys2, plotVarDataLooseIsoSS_sys2,\
                 plotVarDataSS_sys3, plotVarDataLooseIsoSS_sys3,\
                 log):
      QCDlooseSS=copy.deepcopy(plotVarDataLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseSS, plotVarDataLooseIsoSS)

      QCDtightSS=copy.deepcopy(plotVarDataSS.Hist("Data"))
      substractMCbackground(QCDtightSS, plotVarDataSS)

      tightLoose=QCDtightSS.Integral()/QCDlooseSS.Integral()
      tightLooseErr=tightLoose*math.sqrt(1./abs(QCDtightSS.Integral()) + 1./abs(QCDlooseSS.Integral()))
      
      QCDlooseSS_sys1=copy.deepcopy(plotVarDataLooseIsoSS_sys1.Hist("Data"))
      substractMCbackground(QCDlooseSS_sys1, plotVarDataLooseIsoSS_sys1)

      QCDtightSS_sys1=copy.deepcopy(plotVarDataSS_sys1.Hist("Data"))
      substractMCbackground(QCDtightSS_sys1, plotVarDataSS_sys1)

      tightLoose_sys1=QCDtightSS_sys1.Integral()/QCDlooseSS_sys1.Integral()
      tightLooseErr_sys1=tightLoose_sys1*math.sqrt(1./abs(QCDtightSS_sys1.Integral()) + 1./abs(QCDlooseSS_sys1.Integral()))
      
      QCDlooseSS_sys2=copy.deepcopy(plotVarDataLooseIsoSS_sys2.Hist("Data"))
      substractMCbackground(QCDlooseSS_sys2, plotVarDataLooseIsoSS_sys2)

      QCDtightSS_sys2=copy.deepcopy(plotVarDataSS_sys2.Hist("Data"))
      substractMCbackground(QCDtightSS_sys2, plotVarDataSS_sys2)

      tightLoose_sys2=QCDtightSS_sys2.Integral()/QCDlooseSS_sys2.Integral()
      tightLooseErr_sys2=tightLoose_sys2*math.sqrt(1./abs(QCDtightSS_sys2.Integral()) + 1./abs(QCDlooseSS_sys2.Integral()))
      
      QCDlooseSS_sys3=copy.deepcopy(plotVarDataLooseIsoSS_sys3.Hist("Data"))
      substractMCbackground(QCDlooseSS_sys3, plotVarDataLooseIsoSS_sys3)

      QCDtightSS_sys3=copy.deepcopy(plotVarDataSS_sys3.Hist("Data"))
      substractMCbackground(QCDtightSS_sys3, plotVarDataSS_sys3)

      tightLoose_sys3=QCDtightSS_sys3.Integral()/QCDlooseSS_sys3.Integral()
      tightLooseErr_sys3=tightLoose_sys3*math.sqrt(1./abs(QCDtightSS_sys3.Integral()) + 1./abs(QCDlooseSS_sys3.Integral()))
      
      if tightLoose<tightLooseErr/2.:
          tightLoose=tightLooseErr/2.
      
      if tightLoose_sys1<tightLooseErr_sys1/2.:
          tightLoose_sys1=tightLooseErr_sys1/2.
      
      if tightLoose_sys2<tightLooseErr_sys2/2.:
          tightLoose_sys2=tightLooseErr_sys2/2.
      
      if tightLoose_sys3<tightLooseErr_sys3/2.:
          tightLoose_sys3=tightLooseErr_sys3/2.
      
      print "QCD yield error: stat:", tightLooseErr/tightLoose, "sys1:", tightLoose_sys1/tightLoose-1.0, "sys2:", tightLoose_sys2/tightLoose-1.0, "sys3:", tightLoose_sys3/tightLoose-1.0
      
      overall=math.sqrt(pow(tightLooseErr/tightLoose,2)+max(tightLoose_sys1/tightLoose-1.0,tightLoose_sys2/tightLoose-1.0,tightLoose_sys3/tightLoose-1.0,2))
      
      return tightLooseErr/tightLoose,overall




def QCDEstimate6(prefix,prefix1,var,xmin,xmax,\
                 plotVarDataSS, plotVarDataLooseIsoOS, plotVarDataLooseIsoSS,\
                 log):
      ymax=plotVarDataLooseIsoOS.Hist("Data").GetMaximum()*1.5
      if log:
        plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0.1,ymax)
        gPad.SetLogy()
      else:
        plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0,ymax)
        gPad.SetLogy(False)
        gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoOS.varName+"_qcdLooseOS.pdf")
      
      ymax=plotVarDataLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
        plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
        gPad.SetLogy()
      else:
        plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
        gPad.SetLogy(False)
        gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoSS.varName+"_qcdLooseSS.pdf")
      
      ymax=plotVarDataSS.Hist("Data").GetMaximum()*1.5
      if log:
        plotVarDataSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
        gPad.SetLogy()
      else:
        plotVarDataSS.DrawStack("HIST",xmin,xmax,0,ymax)
        gPad.SetLogy(False)
        gPad.SaveAs(prefix1+prefix+'_'+plotVarDataSS.varName+"_qcdTightSS.pdf")
      
      QCDlooseOS = copy.deepcopy(plotVarDataLooseIsoOS.Hist("Data"))
      substractMCbackground(QCDlooseOS, plotVarDataLooseIsoOS)

      QCDlooseSS = copy.deepcopy(plotVarDataLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseSS, plotVarDataLooseIsoSS)

      QCDtightSS = copy.deepcopy(plotVarDataSS.Hist("Data"))
      substractMCbackground(QCDtightSS, plotVarDataSS)

      QCD_PrintOut = open( os.getcwd()+"/"+prefix+"/QCD_PrintOut_"+var+".txt","w")
      print >> QCD_PrintOut, 'QCD estimation 6: shape from SS tight, normalization from OSLoose/SSLoose'
      print >> QCD_PrintOut, "QCDlooseOS:", QCDlooseOS.Integral()
      print >> QCD_PrintOut, "QCDlooseSS:", QCDlooseSS.Integral()
      print >> QCD_PrintOut, "QCDtightSS:", QCDtightSS.Integral()
      OSoverSS      = QCDlooseOS.Integral()/QCDlooseSS.Integral()
      OSoverSSErr   = OSoverSS*math.sqrt(1./abs(QCDlooseSS.Integral()) + 1./abs(QCDlooseOS.Integral()))
      print >> QCD_PrintOut, "QCDlooseOS / QCDlooseSS", OSoverSS, "+-", OSoverSSErr
      QCD_PrintOut.close()
      
      if OSoverSS < OSoverSSErr/2.:
        OSoverSS = OSoverSSErr/2.
      
      QCDScale = OSoverSS
      
      return QCDtightSS, QCDScale, QCDlooseSS, QCDtightSS


def QCDEstimate7(prefix,prefix1,var,xmin,xmax,\
                 plotVarDataSS, plotVarDataLooseIsoOS, plotVarDataLooseIsoOSMC, plotVarDataLooseIsoSS,\
                 log):
      ymax=plotVarDataLooseIsoOS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoOS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoOS.varName+"_qcdLooseOS.pdf")
      
      ymax=plotVarDataLooseIsoSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataLooseIsoSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataLooseIsoSS.varName+"_qcdLooseSS.pdf")
      
      ymax=plotVarDataSS.Hist("Data").GetMaximum()*1.5
      if log:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0.1,ymax)
	  gPad.SetLogy()
      else:
          plotVarDataSS.DrawStack("HIST",xmin,xmax,0,ymax)
	  gPad.SetLogy(False)
      gPad.SaveAs(prefix1+prefix+'_'+plotVarDataSS.varName+"_qcdTightSS.pdf")
      
      QCDlooseOS = copy.deepcopy(plotVarDataLooseIsoOS.Hist("Data"))
      substractMCbackground(QCDlooseOS, plotVarDataLooseIsoOSMC)

      QCDlooseSS = copy.deepcopy(plotVarDataLooseIsoSS.Hist("Data"))
      substractMCbackground(QCDlooseSS, plotVarDataLooseIsoSS)

      QCDtightSS = copy.deepcopy(plotVarDataSS.Hist("Data"))
      substractMCbackground(QCDtightSS, plotVarDataSS)

      QCD_PrintOut = open( os.getcwd()+"/"+prefix+"/QCD_PrintOut_"+var+".txt","w")
      print >> QCD_PrintOut, "QCDlooseOS:", QCDlooseOS.Integral()
      print >> QCD_PrintOut, "QCDlooseSS:", QCDlooseSS.Integral()
      print >> QCD_PrintOut, "QCDtightSS:", QCDtightSS.Integral()
      tightLoose    = QCDtightSS.Integral()/QCDlooseSS.Integral()
      tightLooseErr = tightLoose*math.sqrt(1./abs(QCDtightSS.Integral()) + 1./abs(QCDlooseSS.Integral()))
      OSSS=QCDlooseOS.Integral()/QCDlooseSS.Integral()
      OSSSErr=OSSS*math.sqrt(1./abs(QCDlooseOS.Integral()) + 1./abs(QCDlooseSS.Integral()))
      print >> QCD_PrintOut, "QCDtightSS / QCDlooseSS", tightLoose, "+-", tightLooseErr
      print >> QCD_PrintOut, "QCDlooseOS / QCDlooseSS", OSSS, "+-", OSSSErr
      QCD_PrintOut.close()
      
      if OSSS<OSSSErr/2.:
          OSSS=OSSSErr/2.
      if tightLoose<tightLooseErr/2.:
          tightLoose=tightLooseErr/2.
      
      QCDScale=tightLoose*OSSS
      
      return QCDlooseSS, QCDScale, QCDlooseOS, QCDtightSS, tightLooseErr/tightLoose
