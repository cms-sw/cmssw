import FWCore.ParameterSet.Config as cms

# Parameter to be used for:
# Efficiency = 99% and Fake Rate = 1%
# For deviations bigger then 0.1*mu0 and 2*mu0

f_low     = cms.double(0.1)                               #factor f : mu1=f*mu0
f_up      = cms.double(2.0)                               #factor f : mu1=f*mu0
mu0_x0p1  = cms.vdouble(2.19664,1.94546,-99.3263,19.388)  #parameters for determination of mu0_min
mu0_x2    = cms.vdouble(13.7655,184.742,50735.3,-97.6793) #parameters for determination of mu0_min
chi2_x0p1 = cms.vdouble(4.11095e-05,0.577451,-10.378)     #parameters for determination of chi2-threshold
chi2_x2   = cms.vdouble(5.45058e-05,0.268756,-11.7515)    #parameters for determination of chi2-threshold
