import os.path, glob, sys
import ROOT
import array
import math

# N.B.: Consult ./xeon_scripts/benchmark-cmssw-ttbar-fulldet-build.sh for info on nTHs, nVUs, and text file names

def run():
    # command line input
    arch   = sys.argv[1] # SNB, KNL, SKL-SP
    sample = sys.argv[2] 
    build  = sys.argv[3] # BH, STD, CE, FV
    isVU   = sys.argv[4] # 'true' or 'false': if no argument passed, will not do VU plots
    isTH   = sys.argv[5] # 'true' or 'false': if no argument passed, will not do TH plots

    # reopen file for writing
    g = ROOT.TFile('benchmark_'+arch+'_'+sample+'.root','update')

    # Vectorization data points
    vuvals = ['1','2','4','8']
    nth = '1'
    
    if  arch == 'KNL' or arch == 'SKL-SP' or arch == 'LNX-G' or arch == 'LNX-S':
        vuvals.append('16')
        vuvals.append('16int')
    elif arch == 'SNB' :
        vuvals.append('8int')
    else :
        print arch,'is not a valid architecture! Exiting...'
        sys.exit(0)

    # call the make plots function
    if isVU == 'true' :
        makeplots(arch,sample,build,vuvals,nth,'VU')

    # Parallelization datapoints
    if arch == 'KNL' :
        nvu = '16int'
        thvals = ['1','2','4','8','16','32','64','96','128','160','192','224','256']
    elif arch == 'SNB' :
        nvu = '8int'
        thvals = ['1','2','4','6','8','12','16','20','24']
    elif arch == 'SKL-SP' :
        nvu = '16int'
        thvals = ['1','2','4','8','16','32','48','64']
    elif arch == 'LNX-G' :
        nvu = '16int'
        thvals = ['1','2','4','8','16','32','48','64']
    elif arch == 'LNX-S' :
        nvu = '16int'
        thvals = ['1','2','4','8','16','32','48','64']
    else :
        print arch,'is not a valid architecture! Exiting...'
        sys.exit(0)
    
    # call the make plots function
    if isTH == 'true' :
        makeplots(arch,sample,build,thvals,nvu,'TH')

    g.Write()
    g.Close()

def makeplots(arch,sample,build,vals,nC,text):
    # position in logs
    if   build == 'BH'  : pos = 8  
    elif build == 'STD' : pos = 11  
    elif build == 'CE'  : pos = 14 
    elif build == 'FV'  : pos = 17
    else :
        print build,'is not a valid test! Exiting...'
        sys.exit(0)

    # time    
    print arch,sample,build,text

    # define tgraphs vs absolute time and speedup
    g_time    = ROOT.TGraphErrors(len(vals)-1)
    g_speedup = ROOT.TGraphErrors(len(vals)-1)

    # make separate plot for intrinsics measurement
    if text is 'VU' :
        g_time_int    = ROOT.TGraphErrors(1)
        g_speedup_int = ROOT.TGraphErrors(1)

    point = 0
    for val in vals :
        if    val is '16int': xval = 16.0
        elif  val is '8int' : xval = 8.0
        else                : xval = float(val)

        # array of time values
        yvals = array.array('d');

        # always skip the first event
        firstFound = False

        # open the correct log file, store times into temp file
        if   text is 'VU' : os.system('grep Matriplex log_'+arch+'_'+sample+'_'+build+'_NVU'+val+'_NTH'+nC +'.txt >& log_'+arch+'_'+sample+'_'+build+'_'+text+'.txt')
        elif text is 'TH' : os.system('grep Matriplex log_'+arch+'_'+sample+'_'+build+'_NVU'+nC +'_NTH'+val+'.txt >& log_'+arch+'_'+sample+'_'+build+'_'+text+'.txt')
        else :
            print 'VU or TH are the only options for extra text! Exiting...'
            exit

        # open temp file, store event times into yvals
        with open('log_'+arch+'_'+sample+'_'+build+'_'+text+'.txt') as f :
            for line in f :
                if 'Matriplex' not in line : continue
                if 'Total' in line : continue
                if not firstFound :
                    firstFound = True
                    continue
                lsplit = line.split()
                yvals.append(float(lsplit[pos]))

        # Compute mean and uncertainty on mean from yvals
        sum = 0.;
        for yval in range(0,len(yvals)):
            sum = sum + yvals[yval]
        if len(yvals) > 0 :
            mean = sum/len(yvals)
        else :
            mean = 0
        emean = 0.;
        for yval in range(0,len(yvals)):
            emean = emean + ((yvals[yval] - mean) * (yvals[yval] - mean))
        if len(yvals) > 1 :
            emean = math.sqrt(emean / (len(yvals) - 1))
            emean = emean/math.sqrt(len(yvals))
        else :
            emean = 0

        # Printout value for good measure
        print val,mean,'+/-',emean

        # store intrinsics val into separate plot
        if 'int' not in val :
            g_time.SetPoint(point,xval,mean)
            g_time.SetPointError(point,0,emean)
            point = point+1
        else :
            g_time_int.SetPoint(0,xval,mean)
            g_time_int.SetPointError(0,0,emean)

    # always write out the standard plot
    g_time.Write('g_'+build+'_'+text+'_time')

    # write out separate intrinsics plot
    if text is 'VU' :
        g_time_int.Write('g_'+build+'_'+text+'_time_int')

    # Speedup calculation
    xval0 = array.array('d',[0])
    yval0 = array.array('d',[0])
    yerr0 = array.array('d',[0])

    # Get first point to divide by
    g_time.GetPoint(0,xval0,yval0)
    yerr0.append(g_time.GetErrorY(0))

    point = 0
    for val in vals :
        # set up inputs
        xval = array.array('d',[0])
        yval = array.array('d',[0])
        yerr = array.array('d',[0])

        # get standard plots from standard plot
        if 'int' not in val :
            g_time.GetPoint(point,xval,yval)
            yerr.append(g_time.GetErrorY(point))
        else :
            g_time_int.GetPoint(0,xval,yval)
            yerr.append(g_time_int.GetErrorY(0))

        speedup  = 0.
        espeedup = 0.
        if yval[0] > 0. and yval0[0] > 0. : 
            speedup  = yval0[0]/yval[0]
            espeedup = speedup * math.sqrt(math.pow(yerr0[0]/yval0[0],2) + math.pow(yerr[0]/yval[0],2))

        # store in the correct plot
        if 'int' not in val :
            g_speedup.SetPoint(point,xval[0],speedup)
            g_speedup.SetPointError(point,0,espeedup)
            point = point+1
        else :
            g_speedup_int.SetPoint(0,xval[0],speedup)
            g_speedup_int.SetPointError(0,0,espeedup)

    # always write out the standard plot
    g_speedup.Write('g_'+build+'_'+text+'_speedup')

    # write out separate intrinsics plot
    if text is 'VU' :
        g_speedup_int.Write('g_'+build+'_'+text+'_speedup_int')

    # all done
    return

if __name__ == "__main__":
    run()
