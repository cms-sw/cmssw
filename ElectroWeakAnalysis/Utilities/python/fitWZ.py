#!/usr/bin/env python

from __future__ import division
from math import *
from ROOT import *
from array import array

gROOT.SetBatch(True)

# Wmunu inputs
wxsection = 9.922
ewxsection = 0.090
zxsection = 0.924
ezxsection = 0.031
sysEff = 0.013
corrZEff = -0.236

# FCN definition
def MyChi2 (npar, g, chi2, x, iflag):
      # Inverse of (Zyield,sysEff) covariance matrix
      s11 = 1./ezxsection**2/(1.-corrZEff**2)
      s22 = 1./sysEff**2/(1.-corrZEff**2)
      s12 = -corrZEff/(1.-corrZEff**2)/ezxsection/sysEff

      # Build chi2
      wxsFit = x[0]*x[1]
      zxsFit = x[1]
      effFit = x[2]
      chi2[0] = (wxsFit*(1.+effFit)-wxsection)*(wxsFit*(1.+effFit)-wxsection)/ewxsection/ewxsection
      chi2[0] += (zxsFit-zxsection)*(zxsFit-zxsection) *s11
      chi2[0] += effFit*effFit * s22
      chi2[0] += 2*(zxsFit-zxsection)*effFit * s12

# Minimation, main program
gMinuit = TMinuit()
gMinuit.SetPrintLevel(-1)
gMinuit.SetFCN(MyChi2)
arglist = array('d', 10*[0.])
ier = Long(0)

gMinuit.mnparm(0, "W/Z ratio   Mu", 10.0, 1.e-3, 0, 0, ier)
gMinuit.mnparm(1, "Zsigma [nb] Mu", 1.0, 1.e-3, 0, 0, ier)
gMinuit.mnparm(2, "DeltaEffRel Mu", 0.0, 1.e-4, 0, 0, ier)

arglist[0] = 1000.; arglist[1] = 0.1
gMinuit.mnexcm("MINIMIZE", arglist, 2, ier)

finalChi2 = 0.0
gMinuit.mnprin(3,finalChi2)

par0 = Double(0.0)
errpar0 = Double(0.0)
gMinuit.GetParameter(0,par0,errpar0)

# Extract statistical uncertainty on W/Z including efficiency uncertainties
print "\n*** Statistical uncertainty on W/Z (including eff. unc.): %.4f" % (errpar0)
