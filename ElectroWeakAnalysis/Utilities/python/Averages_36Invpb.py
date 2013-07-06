#!/usr/bin/env python
######################################################################################
## Program to average CMS W and Z cross sections for muons and electrons. 
##    Run it as: "python Averages_3Invpb.py"
##
## Notes:
##
## a) For 3 inverse pb statistics, all likelihood profiles are already 
##    Gaussian to a good approximation. Therefore naive combination procedures
##    work.
## b) OPTION allows to average electron and muon measurements in two ways:
##
##    OPTION = "StatisticalAverage" does the average according to statistical 
##    uncertainties. Pros: This average is more rigurous from a statistical point 
##    of view, since some likelihood ansatz is necessary to interpret systematics.
##    Cons: This procedure leads to larger "overall" uncertainties at the level of the 
##    final combination since, for similar statistical uncertainties, measurements 
##    with larger systematics weight the same as measurements with lower systematics.
##
##    OPTION = ""StatisticalPlusSystematicAverage" does the average according
##    to the overall uncertainties. It assumes that systematic ucnertainties 
##    can be treated in a naive Gaussian way and so added quadratically to 
##    statistical uncertainties in the usual way. Correlations are taken into 
##    account.  A covariancia matrix "V" is built, and the solution X corresponds 
##    to the minimization of the expression " sum_{ij}(X-x_i V_{ij}^{-1} (X-x_j)", 
##    where x_i are the electron and muon measurements. Pros: this leads to minimal
##    uncertainties for the overall uncertainty (if stat. and syst. are added in 
##    quadrature, as people usually do). Cons: most of the systematic soruces are 
##    not statistical in origin, so giving them a 68% CL Gaussian meaning is 
##    an ad-hoc assumption. 
##
######################################################################################

from __future__ import division
from math import *

OPTION = "StatisticalPlusSystematicAverage"
#OPTION= "StatisticalAverage"

print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
print ">>>>> METHOD TO AVERAGE MUONS AND ELECTRONS is: '%s'" % (OPTION)
print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"

######################################################################################
###  INPUTS FOLLOW (they are not the final version)
######################################################################################

# Relative luminosity error
relSysLumi = 11e-2

# Electron inputs
Wenu = 10.221
absStatWenu = 0.034
absCorrWenu = 0.144 # theory uncertainty
absUncWenu = 0.309
print "\nWenu cross section = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (theo.) +- %.4f (lumi.) [nb]" % (Wenu, absStatWenu, absUncWenu, absCorrWenu, Wenu*relSysLumi)
print "  Systematics >>>>>>>"
print "\tUncorrelated with muons:           %.2f %%" % (100./Wenu*absUncWenu)
print "\tTheory:                            %.2f %%" % (100./Wenu*absCorrWenu)
print "\tTOTAL (LUMI excluded):             %.2f %%\n" % (100./Wenu*sqrt(absUncWenu**2+absCorrWenu**2))

Wplusenu = 6.045
absStatWplusenu = 0.026
absCorrWplusenu = 0.097
absUncWplusenu = 0.187
print "\nWplusenu cross section = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (theo.) +- %.4f (lumi.) [nb]" % (Wplusenu, absStatWplusenu, absUncWplusenu, absCorrWplusenu, Wplusenu*relSysLumi)
print "  Systematics >>>>>>>"
print "\tUncorrelated with muons:           %.2f %%" % (100./Wplusenu*absUncWplusenu)
print "\tTheory:                            %.2f %%" % (100./Wplusenu*absCorrWplusenu)
print "\tTOTAL (LUMI excluded):             %.2f %%\n" % (100./Wplusenu*sqrt(absUncWplusenu**2+absCorrWplusenu**2))

Wminusenu = 4.196
absStatWminusenu = 0.022
absCorrWminusenu = 0.073 # theory uncertainty
absUncWminusenu = 0.130
print "\nWminusenu cross section = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (theo.) +- %.4f (lumi.) [nb]" % (Wminusenu, absStatWminusenu, absUncWminusenu, absCorrWminusenu, Wminusenu*relSysLumi)
print "  Systematics >>>>>>>"
print "\tUncorrelated with muons:           %.2f %%" % (100./Wminusenu*absUncWminusenu)
print "\tTheory:                            %.2f %%" % (100./Wminusenu*absCorrWminusenu)
print "\tTOTAL (LUMI excluded):             %.2f %%\n" % (100./Wminusenu*sqrt(absUncWminusenu**2+absCorrWminusenu**2))

Zee = 0.9892
absStatZee = 0.0109
absCorrZee = 0.01715 # theory uncertainty
absUncZee = 0.0446
print "\nZee cross section = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (theo.) +- %.4f (lumi.) [nb]" % (Zee, absStatZee, absUncZee, absCorrZee, Zee*relSysLumi)
print "  Systematics >>>>>>>"
print "\tUncorrelated with muons:           %.2f %%" % (100./Zee*absUncZee)
print "\tTheory:                            %.2f %%" % (100./Zee*absCorrZee)
print "\tTOTAL (LUMI excluded):             %.2f %%\n" % (100./Zee*sqrt(absUncZee**2+absCorrZee**2))

Ratioenu = Wplusenu/Wminusenu
absStatRatioenu = 0.0097
absCorrRatioenu = 0.0306 # theory uncertainty
absUncRatioenu = 0.0453
print "\nRatioenu cross section = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (theo.)" % (Ratioenu, absStatRatioenu, absUncRatioenu, absCorrRatioenu)
print "  Systematics >>>>>>>"
print "\tUncorrelated with muons:           %.2f %%" % (100./Ratioenu*absUncRatioenu)
print "\tTheory:                            %.2f %%" % (100./Ratioenu*absCorrRatioenu)
print "\tTOTAL:                             %.2f %%\n" % (100./Ratioenu*sqrt(absUncRatioenu**2+absCorrRatioenu**2))

WZe = Wenu/Zee
absStatWZe = 0.1190
absCorrWZe = 0.1413 # theory uncertainty
absUncWZe = 0.2329
print "\nWZe cross section = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (theo.)" % (WZe, absStatWZe, absUncWZe, absCorrWZe)
print "  Systematics >>>>>>>"
print "\tUncorrelated with muons:           %.2f %%" % (100./WZe*absUncWZe)
print "\tTheory:                            %.2f %%" % (100./WZe*absCorrWZe)
print "\tTOTAL:                             %.2f %%\n" % (100./WZe*sqrt(absUncWZe**2+absCorrWZe**2))

# Muon inputs 
Wmunu = 10.03115
absStatWmunu = 0.02690
absCorrWmunu = Wmunu*sqrt(0.018**2+0.015**2) # theory uncertainty
relUncFit = 0.4e-2
relUncPreTrig = 0.5e-2
relUncSysEff = 0.4e-2
relUncEff = sqrt(relUncFit**2+relUncPreTrig**2+relUncSysEff**2)
relUncMomRes = 0.1e-2
relUncRecoil = 0.4e-2
relUncMCStat = 1.4e-3/sqrt(2) # ??
relUncBkg = 1.5e-2 #sqrt(2.0e-2**2+0.2e-2**2)
absUncWmunu=Wmunu*sqrt(relUncEff**2+relUncMomRes**2+relUncRecoil**2+relUncMCStat**2+relUncBkg**2)
absUncWmunu=Wmunu*sqrt(relUncEff**2+relUncMomRes**2+relUncRecoil**2+relUncBkg**2)
print "\nWmunu cross section = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (theo.) +- %.4f (lumi.) [nb]" % (Wmunu, absStatWmunu, absUncWmunu, absCorrWmunu, Wmunu*relSysLumi)
print "  Systematics >>>>>>>"
print "\tEfficiency(Zfit,Pretrig,Z->W):     %.2f %%" % (100*relUncEff)
print "\tMomentum scale/resolution:         %.2f %%" % (100*relUncMomRes)
print "\tBackground subtraction:            %.2f %%" % (100*relUncBkg)
print "\tSignal Recoil modeling:            %.2f %%" % (100*relUncRecoil)
print "\tMC statistics (acceptance):        %.2f %%" % (100*relUncMCStat)
print "\tTheory:                            %.2f %%" % (100./Wmunu*absCorrWmunu)
print "\tTOTAL (LUMI excluded):             %.2f %%\n" % (100./Wmunu*sqrt(absUncWmunu**2+absCorrWmunu**2))

Wplusmunu = 5.93821
absStatWplusmunu = 0.02033
absCorrWplusmunu = Wplusmunu*sqrt(0.013**2+0.014**2) # theory uncertainty
relUncFit = 1.3e-2
relUncPreTrig = 0.5e-2
relUncSysEff = 0.4e-2
relUncEff = sqrt(relUncFit**2+relUncPreTrig**2+relUncSysEff**2)
relUncMomRes = 0.1e-2
relUncRecoil = 0.4e-2
relUncMCStat = 1.4e-3
relUncBkg = 1.7e-2 #sqrt(1.7e-2**2+0.2e-2**2)
absUncWplusmunu=Wplusmunu*sqrt(relUncEff**2+relUncMomRes**2+relUncRecoil**2+relUncMCStat**2+relUncBkg**2)
print "\nWplusmunu cross section = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (theo.) +- %.4f (lumi.) [nb]" % (Wplusmunu, absStatWplusmunu, absUncWplusmunu, absCorrWplusmunu, Wplusmunu*relSysLumi)
print "  Systematics >>>>>>>"
print "\tEfficiency(Zfit,Pretrig,Z->W):     %.2f %%" % (100*relUncEff)
print "\tMomentum scale/resolution:         %.2f %%" % (100*relUncMomRes)
print "\tBackground subtraction:            %.2f %%" % (100*relUncBkg)
print "\tSignal Recoil modeling:            %.2f %%" % (100*relUncRecoil)
print "\tMC statistics (acceptance):        %.2f %%" % (100*relUncMCStat)
print "\tTheory:                            %.2f %%" % (100./Wplusmunu*absCorrWplusmunu)
print "\tTOTAL (LUMI excluded):             %.2f %%\n" % (100./Wplusmunu*sqrt(absUncWplusmunu**2+absCorrWplusmunu**2))

Wminusmunu = 4.09297
absStatWminusmunu = 0.01662
absCorrWminusmunu = Wminusmunu*sqrt(0.019**2+0.013**2) # theory uncertainty
relUncFit = 1.3e-2
relUncPreTrig = 0.5e-2
relUncSysEff = 0.4e-2
relUncEff = sqrt(relUncFit**2+relUncPreTrig**2+relUncSysEff**2)
relUncMomRes = 0.1e-2
relUncRecoil = 0.4e-2
relUncMCStat = 1.4e-3
relUncBkg = 2.3e-2 #sqrt(2.3e-2**2+0.2e-2**2)
absUncWminusmunu=Wminusmunu*sqrt(relUncEff**2+relUncMomRes**2+relUncRecoil**2+relUncMCStat**2+relUncBkg**2)
print "\nWminusmunu cross section = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (theo.) +- %.4f (lumi.) [nb]" % (Wminusmunu, absStatWminusmunu, absUncWminusmunu, absCorrWminusmunu, Wminusmunu*relSysLumi)
print "  Systematics >>>>>>>"
print "\tEfficiency(Zfit,Pretrig,Z->W):     %.2f %%" % (100*relUncEff)
print "\tMomentum scale/resolution:         %.2f %%" % (100*relUncMomRes)
print "\tBackground subtraction:            %.2f %%" % (100*relUncBkg)
print "\tSignal Recoil modeling:            %.2f %%" % (100*relUncRecoil)
print "\tMC statistics (acceptance):        %.2f %%" % (100*relUncMCStat)
print "\tTheory:                            %.2f %%" % (100./Wminusmunu*absCorrWminusmunu)
print "\tTOTAL (LUMI excluded):             %.2f %%\n" % (100./Wminusmunu*sqrt(absUncWminusmunu**2+absCorrWminusmunu**2))

Zmumu = 0.961 # 0.893*1.025*1.01
absStatZmumu = 0.008 # 0.030*1.025*1.01
absCorrZmumu = Zmumu*sqrt(0.012**2+0.016**2) # theory uncertainty
relUncEff = 0.5e-2 # pre-triggering
#relUncFit= 0.28e-2
relUncMomRes = 0.35e-2
relUncTrigChanges = 0.1e-2
relUncBkg = 0.28e-2 #sqrt(relUncFit**2+0.2e-2**2)
absUncZmumu=Zmumu*sqrt(relUncEff**2+relUncMomRes**2+relUncBkg**2+relUncTrigChanges**2)
print "\nZmumu cross section = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (theo.) +- %.4f (lumi.) [nb]" % (Zmumu, absStatZmumu, absUncZmumu, absCorrZmumu, Zmumu*relSysLumi)
print "  Systematics >>>>>>>"
print "\tEfficiency(Pretriggering):         %.2f %%" % (100*relUncEff)
print "\tMomentum scale/resolution:         %.2f %%" % (100*relUncMomRes)
print "\tBackground subtraction/fit:        %.2f %%" % (100*relUncBkg)
print "\tMC statistics (acceptance):        %.2f %%" % (100*relUncMCStat)
print "\tTheory:                            %.2f %%" % (100./Zmumu*absCorrZmumu)
print "\tTOTAL (LUMI excluded):             %.2f %%\n" % (100./Zmumu*sqrt(absUncZmumu**2+absCorrZmumu**2))

Ratiomunu = Wplusmunu/Wminusmunu
absStatRatiomunu = 0.0078
absCorrRatiomunu = Ratiomunu*sqrt(0.021**2+0.0129**2) # theory uncertainty
relUncEff = 2.8e-2
relUncMomRes = 0.3e-2
relUncMCStat = sqrt(2)*1.4e-3
relUncBkg = 0.7e-2
absUncRatiomunu = Ratiomunu*sqrt(relUncEff**2+relUncMomRes**2+relUncMCStat**2+relUncBkg**2)
print "\nRatiomunu cross section = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (theo.)" % (Ratiomunu, absStatRatiomunu, absUncRatiomunu, absCorrRatiomunu)
print "  Systematics >>>>>>>"
print "\tEfficiency(W+ versus W- tests):    %.2f %%" % (100*relUncEff)
print "\tMomentum scale/resolution:         %.2f %%" % (100*relUncMomRes)
print "\tBackground subtraction:            %.2f %%" % (100*relUncBkg)
print "\tMC statistics (acceptance):        %.2f %%" % (100*relUncMCStat)
print "\tTheory:                            %.2f %%" % (100./Ratiomunu*absCorrRatiomunu)
print "\tTOTAL:                             %.2f %%\n" % (100./Ratiomunu*sqrt(absUncRatiomunu**2+absCorrRatiomunu**2))

WZmu = Wmunu/Zmumu
absStatWZmu = WZmu*sqrt((absStatWmunu/Wmunu)**2+(absStatZmumu/Zmumu)**2)
absCorrWZmu = WZmu*sqrt(0.011**2+0.0135**2) # theory uncertainty
relUncEffW = 1.3e-2
relUncMomResW = 0.3e-2
relUncMomResZ = 0.2e-2
relSysSubtract = sqrt(relUncEffW**2+relUncMomResW**2+relUncMomResZ**2)
relSysAdd = abs(relUncMomResW-relUncMomResZ)
absUncWZmu = WZmu*sqrt((absUncWmunu/Wmunu)**2 + (absUncZmumu/Zmumu)**2 - relSysSubtract**2 + relSysAdd**2)
print "\nWZmu cross section = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (theo.)" % (WZmu, absStatWZmu, absUncWZmu, absCorrWZmu)
print "  STATISTICAL UNCERTAINTY INCLUDES EFFICIENCY (via Zmumu fit)"
print "  Systematics >>>>>>>"
print "\tUncorrelated with electrons:       %.2f %%" % (100./WZmu*absUncWZmu)
print "\tTheory:                            %.2f %%" % (100./WZmu*absCorrWZmu)
print "\tTOTAL:                             %.2f %%\n" % (100./WZmu*sqrt(absUncWZmu**2+absCorrWZmu**2))

######################################################################################
###  Utility functions
######################################################################################

###
def vbtfXSectionCheck(title, xsection, exsection, sysUnc, sysCor, relSysLumi):
      absSysLumi = xsection*relSysLumi
      print "VBTF inputs: %s = %.4f +- %.4f (stat.) +- %.4f (exp.) +- %.4f (the.) +- %.4f (lumi) [nb]" % (title, xsection, exsection, sysUnc, sysCor, absSysLumi) 

###
def vbtfXSectionAverage(title, xsection1, xsection2, exsection1, exsection2, sysUnc1, sysUnc2, sysCor1, sysCor2, relSysLumi):
      if OPTION== "StatisticalAverage":
            V11 = exsection1**2
            V22 = exsection2**2
            V12 = 0
      else:
            V11 = exsection1**2+sysUnc1**2+sysCor1**2
            V22 = exsection2**2+sysUnc2**2+sysCor2**2
            V12 = sysCor1*sysCor2

      a1 = (V22-V12)/(V11+V22-2*V12)
      a2 = (V11-V12)/(V11+V22-2*V12)
      average = a1*xsection1 + a2*xsection2
      errstat = sqrt(a1**2*exsection1**2+a2**2*exsection2**2)
      errunco = sqrt(a1**2*sysUnc1**2 + a2**2*sysUnc2**2)
      errtheo = sqrt(a1**2*sysCor1**2 + a2**2*sysCor2**2 + 2*a1*a2*sysCor1*sysCor2)
      errsyst = sqrt(errunco**2+errtheo**2)

      print "VBTF average: %s = %.4f +- %.4f (stat.) [nb]" % (title, average, errstat) 

      absSysLumi = average*relSysLumi
      print "\tVBTF systematics (1): +- %.4f (exp) +- %.4f (the) +- %.4f (lumi) [nb]" % (errunco, errtheo, absSysLumi) 
      print "\tVBTF systematics (2): +- %.4f (exp+the) +- %.4f (lumi) [nb]" % (errsyst, absSysLumi) 

###
def vbtfRatioCheck(title, ratio, absStat, sysUnc, sysCor):
      print "VBTF inputs: %s = %.4f +- %.4f (stat.) +- %.4f (exp.) +- %.4f (the.)" % (title, ratio, absStat, sysUnc, sysCor) 

###
def vbtfRatioAverage(title, ratio1, ratio2, eratio1, eratio2, sysUnc1, sysUnc2, sysCor1, sysCor2):
      if OPTION== "StatisticalAverage":
            V11 = eratio1**2
            V22 = eratio2**2
            V12 = 0
      else:
            V11 = eratio1**2+(sysUnc1**2+sysCor1**2)
            V22 = eratio2**2+(sysUnc2**2+sysCor2**2)
            V12 = sysCor1*sysCor2

      a1 = (V22-V12)/(V11+V22-2*V12)
      a2 = (V11-V12)/(V11+V22-2*V12)
      average = a1*ratio1 + a2*ratio2
      errstat = sqrt(a1**2*eratio1**2+a2**2*eratio2**2)
      errunco = sqrt(a1**2*sysUnc1**2 + a2**2*sysUnc2**2)
      errtheo = sqrt(a1**2*sysCor1**2 + a2**2*sysCor2**2 + 2*a1*a2*sysCor1*sysCor2)
      errsyst = sqrt(errunco**2+errtheo**2)

      print "VBTF average: %s = %.4f +- %.4f (stat.)" % (title, average, errstat) 

      print "\tVBTF systematics (1): +- %.4f (exp) +- %.4f (the)" % (errunco, errtheo) 
      print "\tVBTF systematics (2): +- %.4f (exp+the)" % (errsyst) 

######################################################################################
###  MAIN CALLS ...
######################################################################################

#############################################################
########## Wlnu total cross section
#############################################################
print "\n>>>>>>>>>>>>>>>"
vbtfXSectionCheck("W -> munu cross section",Wmunu,absStatWmunu,absUncWmunu,absCorrWmunu,relSysLumi)
vbtfXSectionCheck("W -> enu cross section",Wenu,absStatWenu,absUncWenu,absCorrWenu,relSysLumi)
vbtfXSectionAverage("W -> lnu cross section",Wmunu,Wenu,absStatWmunu,absStatWenu,absUncWmunu,absUncWenu,absCorrWmunu,absCorrWenu,relSysLumi)

#############################################################
########## Wplus -> lnu cross section
#############################################################
print "\n>>>>>>>>>>>>>>>"
vbtfXSectionCheck("W+ -> munu cross section",Wplusmunu,absStatWplusmunu,absUncWplusmunu,absCorrWplusmunu,relSysLumi)
vbtfXSectionCheck("W+ -> enu cross section",Wplusenu,absStatWplusenu,absUncWplusenu,absCorrWplusenu,relSysLumi)
vbtfXSectionAverage("W+ -> lnu cross section",Wplusmunu,Wplusenu,absStatWplusmunu,absStatWplusenu,absUncWplusmunu,absUncWplusenu,absCorrWplusmunu,absCorrWplusenu,relSysLumi)

#############################################################
########## Wminus -> lnu cross section
#############################################################
print "\n>>>>>>>>>>>>>>>"
vbtfXSectionCheck("W- -> munu cross section",Wminusmunu,absStatWminusmunu,absUncWminusmunu,absCorrWminusmunu,relSysLumi)
vbtfXSectionCheck("W- -> enu cross section",Wminusenu,absStatWminusenu,absUncWminusenu,absCorrWminusenu,relSysLumi)
vbtfXSectionAverage("W- -> lnu cross section",Wminusmunu,Wminusenu,absStatWminusmunu,absStatWminusenu,absUncWminusmunu,absUncWminusenu,absCorrWminusmunu,absCorrWminusenu,relSysLumi)

#############################################################
########## W+/W- ratio
#############################################################
#
print "\n>>>>>>>>>>>>>>>"
vbtfRatioCheck("W+ / W- cross section ratio, muon channel",Ratiomunu,absStatRatiomunu,absUncRatiomunu,absCorrRatiomunu)
vbtfRatioCheck("W+ / W- cross section ratio, electron channel",Ratioenu,absStatRatioenu,absUncRatioenu,absCorrRatioenu)
vbtfRatioAverage("W+ / W- cross section ratio",Ratiomunu,Ratioenu,absStatRatiomunu,absStatRatioenu,absUncRatiomunu,absUncRatioenu,absCorrRatiomunu,absCorrRatioenu)

#############################################################
########## Z > ll cross section (in 60 < Mll < 120 GeV)
#############################################################
#
print "\n>>>>>>>>>>>>>>>"
vbtfXSectionCheck("Z -> mumu cross section",Zmumu,absStatZmumu,absUncZmumu,absCorrZmumu,relSysLumi)
vbtfXSectionCheck("Z -> ee cross section",Zee,absStatZee,absUncZee,absCorrZee,relSysLumi)
vbtfXSectionAverage("Z -> ll cross section",Zmumu,Zee,absStatZmumu,absStatZee,absUncZmumu,absUncZee,absCorrZmumu,absCorrZee,relSysLumi)

#############################################################
########## W/Z ratio
#############################################################
#
print "\n>>>>>>>>>>>>>>>"
vbtfRatioCheck("W/Z ratio muons",WZmu,absStatWZmu,absUncWZmu,absCorrWZmu)
vbtfRatioCheck("W/Z ratio electrons",WZe,absStatWZe,absUncWZe,absCorrWZe)
vbtfRatioAverage("W/Z ratio",WZmu,WZe,absStatWZmu,absStatWZe,absUncWZmu,absUncWZe,absCorrWZmu,absCorrWZe)
