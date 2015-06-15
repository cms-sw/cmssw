import copy
import math
from CMGTools.H2TauTau.proto.plotter.H2TauTauDataMC import H2TauTauDataMC
from CMGTools.RootTools.Style import *
from ROOT import kPink, Double, gPad

def sqsum (numa, numb) :
    return math.sqrt(numa * numa + numb * numb)


def buildPlot( var, anaDir,
               comps, weights, nbins, xmin, xmax,
               cut, weight,
               embed, shift=None, treeName=None ):
    pl = H2TauTauDataMC(var, anaDir,
                        comps, weights, nbins, xmin, xmax,
                        str(cut), weight,
                        embed, shift, treeName )
    return pl
    
    
    

def hist( var, anaDir,
          comp, weights, nbins, xmin, xmax,
          cut, weight,
          embed, shift=None, treeName=None ):
    pl = buildPlot( var, anaDir,
                    {comp.name:comp}, weights, nbins, xmin, xmax,
                    cut, weight,
                    embed, shift, treeName )
    histo = copy.deepcopy( pl.Hist(comp.name) )    
    return histo
    
    
    

def shape( var, anaDir,
           comp, weights, nbins, xmin, xmax,
           cut, weight,
           embed, shift=None, treeName=None):
    shape = hist( var, anaDir,
                  comp, weights, nbins, xmin, xmax,
                  cut, weight,
                  embed, shift, treeName )
    shape.Normalize()
    return shape
    
    
    

def shape_and_yield( var, anaDir,
                     comp, weights, nbins, xmin, xmax,
                     cut, weight,
                     embed, treeName=None ):
    shape = hist( var, anaDir,
                  comp, weights, nbins, xmin, xmax,
                  cut, weight,
                  embed, treeName )
    yi = shape.Integral()
    shape.Normalize()
    return shape, yi
    
    
    

def addQCD( plot, dataName, VVgroup):
    # import pdb; pdb.set_trace()
    plotWithQCD = copy.deepcopy( plot )
    qcd = copy.deepcopy(plotWithQCD.Hist(dataName))
    qcd.Add(plotWithQCD.Hist('Ztt'), -1)
    qcd.Add(plotWithQCD.Hist('Ztt_TL'), -1)
    qcd.Add(plotWithQCD.Hist('Ztt_ZL'), -1)
    qcd.Add(plotWithQCD.Hist('Ztt_ZJ'), -1)  
    qcd.Add(plotWithQCD.Hist('TTJets'), -1)
    qcd.Add(plotWithQCD.Hist('WJets'), -1)
    if VVgroup:
        qcd.Add(plotWithQCD.Hist('VV'), -1)

    # adding the QCD data-driven estimation to the  plot
    plotWithQCD.AddHistogram( 'QCD', qcd.weighted, 888)
    plotWithQCD.Hist('QCD').stack = True
    plotWithQCD.Hist('QCD').SetStyle( sHTT_QCD )
    return plotWithQCD
    
    
    

def getQCD( plotSS, plotOS, dataName, VVgroup=None, scale=1.06, subtractBGForShape=True ):

    # use SS data as a control region
    # to get the expected QCD shape and yield
    plotSSWithQCD = addQCD( plotSS, dataName, VVgroup)

    # extrapolate the expected QCD shape and yield to the
    # signal region
    qcd_yield = plotSSWithQCD.Hist('QCD').Integral()
    qcd_yield *= scale
    
    plotOSWithQCD = copy.deepcopy( plotOS )

    if subtractBGForShape:
        qcdOS = copy.deepcopy( plotSSWithQCD.Hist('QCD') )
    else:
        qcdOS = copy.deepcopy( plotSSWithQCD.Hist('Data') )
        
    # qcdOS.RemoveNegativeValues()
    print 'QCD yield', qcd_yield

    qcdOS.Scale( qcd_yield / qcdOS.Integral() )

    plotOSWithQCD.AddHistogram('QCD', qcdOS.weighted, 1030)
    plotOSWithQCD.Hist('QCD').layer=1.5
    plotOSWithQCD.Hist('QCD').SetStyle(sHTT_QCD)

    return plotSSWithQCD, plotOSWithQCD
    
    

def fW(mtplot, dataName, xmin, xmax, VVgroup=None, channel = 'TauMu'):

    wjet = copy.deepcopy(mtplot.Hist(dataName))
    oldIntegral = wjet.Integral(True, xmin, xmax)
    error_data = math.sqrt(oldIntegral)
    wjet.Add(mtplot.Hist('Ztt'), -1)
    wjet.Add(mtplot.Hist('Ztt_ZL'), -1)
    wjet.Add(mtplot.Hist('Ztt_ZJ'), -1)
    wjet.Add(mtplot.Hist('Ztt_TL'), -1)
    wjet.Add(mtplot.Hist('TTJets'), -1)
    if VVgroup:
        wjet.Add(mtplot.Hist('VV'), -1)

    subtrIntegral = wjet.Integral(True, xmin, xmax)

    print 'Subtracted BG', oldIntegral - subtrIntegral

    relSysError = 0.1 * (oldIntegral - subtrIntegral)/subtrIntegral
    print 'W+Jets, high MT: Relative error due to BG subtraction', relSysError

    relSysError = 0.05 * mtplot.Hist('Ztt').Integral(True, xmin, xmax)

    # print 'Error due to Ztt subtraction', relSysError/subtrIntegral

    relSysError = math.sqrt(relSysError**2 + 0.05*0.05*(mtplot.Hist('Ztt_ZJ').Integral(True, xmin, xmax)**2))

    # print 'Plus ZJ subtraction', relSysError/subtrIntegral

    relSysError = math.sqrt(relSysError**2 + 0.2*0.2*(mtplot.Hist('Ztt_ZL').Integral(True, xmin, xmax)**2))

    # print 'Plus ZL subtraction', relSysError/subtrIntegral

    relSysError = math.sqrt(relSysError**2 + 0.2*0.2*(mtplot.Hist('Ztt_TL').Integral(True, xmin, xmax)**2))

    # print 'Plus TL subtraction', relSysError/subtrIntegral

    relSysError = math.sqrt(relSysError**2 + 0.1*0.1 * (mtplot.Hist('TTJets').Integral(True, xmin, xmax)**2))
    # print 'Plus TT subtraction', relSysError/subtrIntegral
    if VVgroup:
      relSysError = math.sqrt(relSysError**2 + 0.3*0.3*mtplot.Hist('VV').Integral(True, xmin, xmax)**2)
      # print 'Plus VV subtraction', relSysError/subtrIntegral
    print 'W+Jets, high MT: Absolute error due to BG subtraction with smaller DY uncertainties', relSysError
    relSysError = relSysError/subtrIntegral
    print 'W+Jets, high MT: Relative error due to BG subtraction with smaller DY uncertainties', relSysError
    # print 'W+Jets, high MT: Contribution from ttbar', 0.1*mtplot.Hist('TTJets').Integral(True, xmin, xmax)/subtrIntegral

    mtplot.AddHistogram( 'Data-DY-TT-VV', wjet.weighted, 1010)
    mtplot.Hist('Data-DY-TT-VV').stack = False
    # with a nice pink color
    pink = kPink+7
    sPinkHollow = Style( lineColor=pink, markerColor=pink, markerStyle=4)
    mtplot.Hist('Data-DY-TT-VV').SetStyle( sPinkHollow )

    data_integral = mtplot.Hist('Data-DY-TT-VV').Integral(True, xmin, xmax)

    error_tmp = Double(0.)
    error_mc = Double(0.)
    data_integral_jan = mtplot.Hist('Data-DY-TT-VV').weighted.IntegralAndError(mtplot.Hist('Data-DY-TT-VV').weighted.FindFixBin(xmin), mtplot.Hist('Data-DY-TT-VV').weighted.FindFixBin(xmax)-1, error_tmp)

    if data_integral_jan != data_integral:
        print 'WARNING, not the same integral in w+jets estimation'

    print 'Adding relative error due to data error', error_data/data_integral
    relSysError = math.sqrt(error_data**2/data_integral**2 + relSysError**2)
    print 'TOTAL ERROR on high mass SF', relSysError
    mc_integral = mtplot.Hist('WJets').Integral(True, xmin, xmax)
    # Do not double-count W+jets uncertainty (is in high-low ratio)
    return data_integral, mc_integral


    

def w_lowHighMTRatio( var, anaDir,
                      comp, weights, 
                      cut, weight, lowMTMax, highMTMin, highMTMax, chargeRequirement, treeName = None):
    cutWithChargeReq = ' && '.join([cut, chargeRequirement]) 
    max = 5000
    mt = shape(var, anaDir,
               comp, weights, max, 0, max,
               cutWithChargeReq, weight,
               None, None, treeName = treeName)

    mt_low = mt.Integral(True, 0, lowMTMax)
    mt_high = mt.Integral(True, highMTMin, highMTMax)
    mt_ratio = mt_low / mt_high 

    print 'W MT ratio, mt ranges', lowMTMax, highMTMin, highMTMax

    error_low = Double(0.)
    error_high = Double(0.)
    mt_low_jan = mt.weighted.IntegralAndError(1, mt.weighted.FindBin(lowMTMax)-1, error_low)
    mt_high_jan = mt.weighted.IntegralAndError(mt.weighted.FindBin(highMTMin), mt.weighted.FindBin(highMTMax), error_high)

    print mt.weighted.FindBin(lowMTMax)-1, mt.weighted.FindBin(highMTMin), mt.weighted.FindBin(highMTMax)

    if mt_high > 0. and mt_low > 0.:
        print 'MT ratio', mt_ratio, '+-', math.sqrt(error_low**2/mt_low**2 + error_high**2/mt_high**2) * mt_ratio
        # print 'MT ratio', mt_low_jan/mt_high_jan, '+-', math.sqrt(error_low**2/mt_low**2 + error_high**2/mt_high**2) * mt_ratio
        print 'Integrals: low', mt_low, '+-', error_low, 'high', mt_high, '+-', error_high
        # print 'Integrals: low', mt_low_jan, '+-', error_low, 'high', mt_high_jan, '+-', error_high
        print 'Relative errors: low', error_low/mt_low, 'high', error_high/mt_high
        print 'TOTAL RELATIVE ERROR:', math.sqrt(error_low**2/mt_low**2 + error_high**2/mt_high**2), '\n'
    return mt_ratio
    
    
    

def plot_W(anaDir, comps, weights, 
           nbins, xmin, xmax,
           cut, weight,
           embed, VVgroup=None, TTgroup=None, treeName=None):

    # get WJet scaling factor for same sign
    var = 'mt'
    sscut = '{cut} && mt>{min} && mt<{max} && diTau_charge!=0'.format(
        cut = cut, 
        min=xmin,
        max=xmax
        )
    oscut = sscut.replace('diTau_charge!=0', 'diTau_charge==0')


    # get WJet scaling factor for opposite sign
    print 'extracting WJets data/MC factor in high mt region, OS'
    print oscut
    mtOS = H2TauTauDataMC( var, anaDir, comps, weights,
                           nbins, xmin, xmax, 
                           cut = oscut, weight=weight,
                           embed=embed, treeName=treeName)
    if VVgroup != None :
        mtOS.Group ('VV',VVgroup)
    if TTgroup != None:
        mtOS.Group('TTJets', TTgroup)

    print TTgroup, mtOS
    
    print 'W SCALE FACTOR OS:'
    data_OS, mc_OS = fW( mtOS, 'Data', xmin, xmax, VVgroup)
    fW_OS = data_OS / mc_OS


    print 'extracting WJets data/MC factor in high mt region, SS'
    print sscut 
    mtSS = H2TauTauDataMC(var, anaDir, comps, weights,
                          nbins, xmin, xmax,
                          cut = sscut, weight=weight,
                          embed=embed, treeName=treeName)
    if VVgroup != None :
        mtSS.Group ('VV',VVgroup)
    if TTgroup != None:
        mtSS.Group('TTJets', TTgroup)

    print 'W SCALE FACTOR SS:'
    data_SS, mc_SS = fW( mtSS, 'Data', xmin, xmax, VVgroup)

    fW_SS = data_SS / mc_SS if mc_SS else -1.

    
    print 'fW_SS=',fW_SS,'fW_OS=',fW_OS
 
    return fW_SS, fW_OS, mtSS, mtOS

