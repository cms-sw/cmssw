import os.path, glob, sys
import ROOT
import array
import math

# N.B.: Consult ./xeon_scripts/benchmark-cmssw-ttbar-fulldet-build.sh for info on nTHs, nVUs, and text file names

arch   = sys.argv[1] # SNB, KNL, SKL-SP
sample = sys.argv[2]
build  = sys.argv[3] # CE, FV

g = ROOT.TFile('benchmarkMEIF_'+arch+'_'+sample+'_'+build+'.root','recreate')

# Parallelization datapoints
if arch == 'KNL' :
    nvu = '16int'
    thvals = ['1','2','4','8','16','32','64','96','128','160','192','224','256']
    evvals = ['1','2','4','8','16','32','64','128']
elif arch == 'SNB' :
    nvu = '8int'
    thvals = ['1','2','4','6','8','12','16','20','24']
    evvals = ['1','2','4','8','12']
elif arch == 'SKL-SP' :
    nvu = '16int'
    thvals = ['1','2','4','8','16','32','48','64']
    evvals = ['1','2','4','8','16','32','64']
elif arch == 'LNX-G' :
    nvu = '16int'
    thvals = ['1','2','4','8','16','32','48','64']
    evvals = ['1','2','4','8','16','32','64']
elif arch == 'LNX-S' :
    nvu = '16int'
    thvals = ['1','2','4','8','16','32','48','64']
    evvals = ['1','2','4','8','16','32','64']
else :
    print arch,"is not a valid architecture! Exiting..."
    sys.exit(0)

# extra text label
text = 'MEIF'

# text for grepping
grepnEV  = '=== TOTAL for'
grepTime = 'Total event loop time'

# needed for speedups
xval0 = array.array('d',[0])
yval0 = array.array('d',[0])

# time    
for evval in evvals :
    print arch,sample,build,"nEV:",evval
    
    # define event float
    ev = float(evval)
        
    # define tgraphs vs absolute time and speedup
    g_time    = ROOT.TGraph()
    g_speedup = ROOT.TGraph()

    point = 0
    for thval in thvals :
        xval = float(thval)
        if ev > xval: continue;
            
        # extracted time
        yval = float(0)
        nev  = float(1)

        # open log file, grep for relevant lines
        with open('log_'+arch+'_'+sample+'_'+build+'_NVU'+nvu+'_NTH'+thval+'_NEV'+evval+'.txt') as f :
            for line in f :
                if grepnEV in line :
                    lsplit = line.split()                
                    nev  = float(lsplit[3])
                elif grepTime in line :
                    lsplit = line.split()                
                    yval = float(lsplit[4])

        yval /= nev

        # Printout value for good measure
        print xval,yval

        # store val
        g_time.SetPoint(point,xval,yval)
        point = point+1

    # write out the plot
    g_time.Write('g_'+build+'_'+text+'_nEV'+evval+'_time')

    # needed for speedup calculation
    if evval is '1' :
        g_time.GetPoint(0,xval0,yval0)        
        
    # speedup plots
    point = 0
    for thval in thvals :
        xval = float(thval)
        if ev > xval: continue;

        # set up inputs
        xval = array.array('d',[0])
        yval = array.array('d',[0])
        
        # get point from time
        g_time.GetPoint(point,xval,yval)

        speedup  = 0.
        if yval[0] > 0. : 
            speedup  = yval0[0]/yval[0]
                
        # store in speedup plot
        g_speedup.SetPoint(point,xval[0],speedup)
        point = point+1

    # always write out speedup
    g_speedup.Write('g_'+build+'_'+text+'_nEV'+evval+'_speedup')

g.Write()
g.Close()
