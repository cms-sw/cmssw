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

OPTION= "StatisticalAverage"
#OPTION = "StatisticalPlusSystematicAverage"

######################################################################################
###  INPUTS FOLLOW (they are not the final version)
######################################################################################

# Relative luminosity error
relSysLumi = 11e-2

# Electron inputs
Wenu = 9.801
absStatWenu = 0.112
absCorrWenu = Wenu*sqrt(0.008**2+0.0116**2) # theory uncertainty
absUncWenu = sqrt(0.495**2-absCorrWenu**2)

Wplusenu = 5.802
absStatWplusenu = 0.079
absCorrWplusenu = Wplusenu*sqrt(0.009**2+0.0133**2) # theory uncertainty
absUncWplusenu = sqrt(0.306**2-absCorrWplusenu**2)

Wminusenu = 3.945
absStatWminusenu = 0.067
absCorrWminusenu = Wminusenu*sqrt(0.015**2+0.0090**2) # theory uncertainty
absUncWminusenu = sqrt(0.224**2-absCorrWminusenu**2)

Zee = 1.006323
absStatZee = 0.031914
absCorrZee = Zee*sqrt(0.011**2+0.0134**2) # theory uncertainty
absUncZee = 0.035613

Ratioenu = Wplusenu/Wminusenu
absStatRatioenu = 0.030
absCorrRatioenu = Ratioenu*sqrt(0.017**2+0.0127**2) # theory uncertainty
absUncRatioenu = sqrt(0.071**2-absCorrRatioenu**2)

WZe = Wenu/Zee
absStatWZe = WZe*sqrt((absStatWenu/Wenu)**2+(absStatZee/Zee)**2)
absCorrWZe = WZe*sqrt(0.009**2+0.0103**2) # theory uncertainty
absUncWZe = WZe*sqrt((absUncWenu/Wenu)**2 + (absUncZee/Zee)**2)

# Muon inputs 
Wmunu = 9.9686
absStatWmunu = 0.0902
absCorrWmunu = Wmunu*sqrt(0.011**2+0.0136**2) # theory uncertainty
absUncWmunu = sqrt(0.3090**2-absCorrWmunu**2)

Wplusmunu = 5.8677
absStatWplusmunu = 0.0680
absCorrWplusmunu = Wplusmunu*sqrt(0.013**2+0.0142**2) # theory uncertainty
absUncWplusmunu = sqrt(0.1773**2-absCorrWplusmunu**2)

Wminusmunu = 4.1008
absStatWminusmunu = 0.0586
absCorrWminusmunu = Wminusmunu*sqrt(0.019**2+0.0126**2) # theory uncertainty
absUncWminusmunu = sqrt(0.1480**2-absCorrWminusmunu**2)

Zmumu = 0.9153
absStatZmumu = 0.0307
absCorrZmumu = Zmumu*sqrt(0.012**2+0.0158**2) # theory uncertainty
absUncZmumu = sqrt(0.0210**2-absCorrZmumu**2)

Ratiomunu = Wplusmunu/Wminusmunu
absStatRatiomunu = 0.0262
absCorrRatiomunu = Ratiomunu*sqrt(0.021**2+0.0119**2) # theory uncertainty
absUncRatiomunu = sqrt(0.0539**2-absCorrRatiomunu**2)

WZmu = Wmunu/Zmumu
absStatWZmu = 0.3740
absCorrWZmu = WZmu*sqrt(0.011**2+0.0135**2) # theory uncertainty
absUncWZmu = sqrt(0.3279**2-absCorrWZmu**2)

######################################################################################
###  Utility functions
######################################################################################

###
def vbtfXSectionCheck(title, xsection, exsection, sysUnc, sysCor, relSysLumi):
      absSysLumi = xsection*relSysLumi

      print "VBTF inputs: %s = %.4f +- %.4f (stat.) +- %.4f (syst.) +- %.4f (lumi) [nb]" % (title, xsection, exsection, sqrt(sysUnc**2+sysCor**2), absSysLumi) 

###
def vbtfXSectionAverage(title, xsection1, xsection2, exsection1, exsection2, sysUnc1, sysUnc2, sysCor1, sysCor2, relSysLumi):
      if OPTION== "StatisticalAverage":
            V11 = exsection1**2
            V22 = exsection2**2
            V12 = 0
      else:
            V11 = exsection1**2 + sysUnc1**2 + sysCor1**2
            V22 = exsection2**2 + sysUnc2**2 + sysCor2**2
            # Correlation is assumed to be 100% positive (no exception for the moment)
            rho12 = +1.0
            V12 = rho12 * sysCor1*sysCor2  

      a1 = (V22-V12)/(V11+V22-2*V12)
      a2 = (V11-V12)/(V11+V22-2*V12)
      average = a1*xsection1 + a2*xsection2
      errstat = sqrt(a1**2*exsection1**2+a2**2*exsection2**2)
      errsyst = sqrt(a1**2*(sysUnc1**2+sysCor1**2) + a2**2*(sysUnc2**2+sysCor2**2) + 2*a1*a2*sysCor1*sysCor2)

      print "VBTF average: %s = %.4f +- %.4f (stat.) [nb]" % (title, average, errstat) 

      absSysLumi = average*relSysLumi
      print "\tVBTF systematics: +- %.4f (det+the) +- %.4f (lumi) [nb]" % (errsyst, absSysLumi) 

###
def vbtfRatioCheck(title, ratio, absStat, sysUnc, sysCor):
      print "VBTF inputs: %s = %.4f +- %.4f (stat.) +- %.4f (syst.)" % (title, ratio, absStat, sqrt(sysUnc**2+sysCor**2)) 

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
      errsyst = sqrt(a1**2*(sysUnc1**2+sysCor1**2) + a2**2*(sysUnc2**2+sysCor2**2) + 2*a1*a2*sysCor1*sysCor2)

      print "VBTF average: %s = %.4f +- %.4f (stat.)" % (title, average, errstat) 

      print "\tVBTF systematics: +- %.4f (det+the)" % (errsyst) 

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
