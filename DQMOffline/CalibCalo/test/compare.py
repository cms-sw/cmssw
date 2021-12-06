#! /usr/bin/env python3

import os
import sys
import errno

#print error/help message and exit
def help_message():
    print "Usage:\n\
compare [folder_name] [options] -v versions_to_compare -f files_to_compare\n\
Versions and files must be whitespace separated.\n\
If no folder is specified the pwd will be used.\n\
folder_name, if specified, must be the first argument.\n\
Options:\n\
-h   prints this message.\n\
--no_exec   does not show graphs (batch mode).\n\
--outfile filename.root    writes the root output in the specified file.\n\
--html   produce html output\n\
--canvas   with --html option and without --no_exec option produces output directly in png, if not specified output is produced in eps and then converted in png format with an external program (but can be used in batch mode).\n\
Example:\n\
./compare.py myDir -v CMSSW_X_Y_Z CMSSW_J_K_W -f file1.root file2.root --no_exec --outfile out.root"
    sys.exit()

#run command in the command line with specified environment
def runcmd(envir,program,*args):
    pid=os.fork()
    if not pid:
        os.execvpe(program,(program,)+args,envir)
    return os.wait()[0]

def runcmd2(envir,program,args):
    pid=os.fork()
    if not pid:
        os.execvpe(program,(program,)+args,envir)
    return os.wait()[0]

#recursive function to search the graphs in the root file
#unfortunately different versions/files produce different paths
wdir=0
ptf=""
def srcpath(fldr):
    tmp=[]
    wdir.cd(fldr)
    for key in wdir.GetListOfKeys():
        if key.IsFolder():
            tmp=srcpath(key.GetName())
            if tmp[0]:
                return [True,key.GetName()+"/"+tmp[1]]
        else:
            if key.GetName()=="eventEnergyEB":
                pth=wdir.pwd()
                return [True,""]
    wdir.cd("../")
    return [False,""]

#print the help message
if "-h" in sys.argv or "-help" in sys.argv or "--help" in sys.argv:
    help_message()

#move to the working diretory (if specified)
cwd=""
if len(sys.argv)>1:
    if not sys.argv[1][0]=="-":
        name=sys.argv[1]        
        try:
            cwd=os.getcwd()
            os.chdir(name)
        except OSError, inst:
            if inst.errno==errno.ENOENT:
                print "Error: the specified working folder does not exist"
                help_message()
else: help_message()

#read and parse the input
state="n"
root_out="compare_out.root"
execute=True
html=False
cnv=False

ver=[]
fil=[]
for arg in sys.argv:
    if arg=="-v" or arg=="-V":
        state="v"
    elif arg=="-f" or arg=="-F":
        state="f"
    elif arg=="--no_exec":
        execute=False
    elif arg=="--outfile":
        state="r"
    elif arg=="--html":
        html=True
    elif arg=="--canvas":
        cnv=True
    elif state=="v":
        ver.append(arg)
    elif state=="f":
        fil.append(arg)
    elif state=="r":
        root_out=arg

#disable window output if --no_exec argument is passed
if not execute:
    sys.argv.append("-b")

#try to use pyROOT directly from here, if environment variables were not set
#this will run again the program with environment variables of the highest version of CMSSW specified in -v
try:
    from ROOT import *
except ImportError:
    print "Warning: environment variables not set, proceeding anyway running this script with the environment variables of the higher version of CMSSW specified in -v"
    #storing cmsenv environment variables
    os.chdir(max(ver))
    env=os.popen("scramv1 runtime -sh","r")
    environment=os.environ
    for l in env.readlines():
        variable,value=l[7:len(l)-3].strip().split("=",1)
        environment[variable]=value[1:]
    env.close()
    if cwd=="":
        os.chdir("../")
    else:
        os.chdir("../../")
    #running again the program
    if execute:
        runcmd2(environment,"./compare.py",tuple(sys.argv[1:]))#works only if compare.py is located in the pwd
    else:
        runcmd2(environment,"./compare.py",tuple(sys.argv[1:-1]))#works only if compare.py is located in the pwd
else:    
    gSystem.Load("libFWCoreFWLite.so")
    FWLiteEnabler::enable()
    outfile=TFile(root_out,"recreate")
    histo=[]
    canvas=[]
    legend=TLegend(.89,0.8,1,.2)
    histonames=["eventEnergyEB","eventEnergyEE","iEtaDistributionEB","iphiDistributionEB","meanEnergyEB","meanEnergyEE","nRechitsEB","nRechitsEE","rhEnergyEB","rhEnergyEE","iEtaDistributionEE"]
    colornames=[kRed,kBlue,kGreen,kCyan,kMagenta,kYellow,kOrange,kPink,kViolet,kAzure,kTeal,kSpring]
#kBlack,
    for nv,v in enumerate(ver):
        histo.append([])#a list for every version: "version list"
        for nf,f in enumerate(fil):
            if not nv:
                canvas.append(TCanvas(f[:len(f)-5],f[:len(f)-5]))
                canvas[nf].Divide(2,6)
            histo[nv].append([])#a list for every file in each version: "file list"
            histo[nv][nf].append(TFile(v+"/"+f))#the first element of the "file list" is the file itself
            histo[nv][nf].append([])#the second element of the "file list" is the "histo list"
            histo[nv][nf][0].cd()
            wdir=gDirectory
            pth=srcpath("")[1]
            for nh,h in enumerate(histonames):
                histo[nv][nf][1].append(histo[nv][nf][0].Get(pth+h))#the histo list contains histos
                canvas[nf].cd(nh+1)
                histo[nv][nf][1][nh].SetLineColor(colornames[nv%len(colornames)])
                if nv:
                    histo[nv][nf][1][nh].Draw("same")
                else:
                    histo[nv][nf][1][nh].Draw()
                if nf==0 and nh==0:
                    legend.AddEntry(histo[nv][nf][1][nh],v[6:],"l")
                if nv==(len(ver)-1):
                    legend.Draw()
            if nv==(len(ver)-1):
                outfile.cd()
                canvas[nf].Write()
    if execute:
        print "Press enter to end the program"
        os.system("read")
    if html:
        if cnv:
            if execute:
                for nf,f in enumerate(fil):
                    try:
                        os.mkdir(f[:len(f)-5])
                    except OSError,inst:
                        if inst.errno==errno.EEXIST:
                            print "Possibly overwriting images"
                    os.system("cp "+cwd+"/temp.html "+f[:len(f)-5]+"/index.html")
                    os.chdir(f[:len(f)-5])
                    canvas[nf].cd()
                    canvas[nf].SetWindowSize(1050,780)
                    #os.system("sleep 2")
                    for nh,h in enumerate(histonames):
                        canvas[nf].cd(nh+1).SetPad(0,0,1,1)
                        canvas[nf].cd(nh+1).Print(h+".png","png")
                    #os.system("sleep 2")
                    canvas[nf].SetWindowSize(500,375)
                    #os.system("sleep 2")
                    for nh,h in enumerate(histonames):
                        canvas[nf].cd(nh+1).SetPad(0,0,1,1)
                        canvas[nf].cd(nh+1).Print(h+"_s.png","png")
                    #os.system("sleep 2")
                    os.chdir("../")
            else:
                print "Warning:to use --canvas option do not use --no_exec option. Rerun without --canvas option."
        else:
            for nf,f in enumerate(fil):
                try:
                    os.mkdir(f[:len(f)-5])
                except OSError,inst:
                    if inst.errno==errno.EEXIST:
                        print "Possibly overwriting images"
                os.system("cp "+cwd+"temp.html "+f[:len(f)-5]+"/index.html")
                os.chdir(f[:len(f)-5])
                canvas[nf].cd()
                canvas[nf].SetWindowSize(1050,780)
                for nh,h in enumerate(histonames):
                    canvas[nf].cd(nh+1).SetPad(0,0,1,1)
                    canvas[nf].cd(nh+1).Print(h+".eps")
                    gSystem.Exec("pstopnm -ppm -xborder 0 -yborder 0 -xsize 1050 -portrait "+h+".eps");
                    gSystem.Exec("pnmtopng "+h+".eps001.ppm > "+h+".png")
                    try:
                        os.remove(h+".eps")
                    except OSError:
                        pass
                    try:
                        os.remove(h+".eps001.ppm")
                    except OSError:
                        pass
                canvas[nf].SetWindowSize(500,375)
                for nh,h in enumerate(histonames):
                    canvas[nf].cd(nh+1).SetPad(0,0,1,1)
                    canvas[nf].cd(nh+1).Print(h+"_s.eps")
                    gSystem.Exec("pstopnm -ppm -xborder 0 -yborder 0 -xsize 500 -portrait "+h+"_s.eps");
                    gSystem.Exec("pnmtopng "+h+"_s.eps001.ppm > "+h+"_s.png")
                    try:
                        os.remove(h+"_s.eps")
                    except OSError:
                        pass
                    try:
                        os.remove(h+"_s.eps001.ppm")
                    except OSError:
                        pass
                os.chdir("../")
