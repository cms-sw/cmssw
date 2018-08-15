#! /usr/bin/env python

#################################################################
# timingPdfMaker.py - python script
# Created by Alejandro Gomez
# For more info, please check:
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideHLTTimingSummary
# or by email: alejandro.gomez@cern.ch

# This python script creates managebles pdf for HLT Timing Studies. 
# It saves some.png files from a former root file, creates a tex file,
# compiles it twice for better results and remove all the temp files.
# Please check the usage function to see how it works.
###############################################################

#############################################
## Imports
############################################

from __future__ import print_function
import os, glob
import os.path
import operator
import subprocess 
import six
import sys, getopt
sys.argv.append('-b')
from ROOT import *
import cStringIO 


gROOT.SetStyle("Plain")
gStyle.SetOptStat(111111)
gStyle.SetHistFillColor(kBlue)

###############################################################
def usage():
###############################################################
    print("\nThis is the usage function\n")
    print('Usage: '+sys.argv[0]+' -i <file1> [option]')
    print('e.g.:  '+sys.argv[0]+' -i outputTiming.root -t')
    print('e.g.:  '+sys.argv[0]+' -i outputTiming.root -s HLT_Jet300_v5\n')

    print('\n-----Options-----')
    print(' -i		 	Input File')
    print(' -o 			Output File (optional)')
    print(' -t 			For only main time info per event. It will take less than 1 min.')
    print(' -p			For path time info. It will take approx 25 min.')
    print(' -m 			For module time info. It will take approx 25 min.')
    print(' -s Path_Name            (For an specific path). Ti will take less than 1 min.')
    print('\n For -p or -m option, the process needs like 200 Mb in disk space,')
    print(' but please dont be afraid. Before the process ends, all the temporal files')
    print(' will be erased.')


###############################################################
def maininfo(infile, outfile):
    ''' Creates main info tex file'''
##############################################################
    texpreamble = ['\documentclass[10pt,a5paper,landscape]{report}\n',
                   '\\usepackage{graphicx}\n',
                    '\\usepackage[a5paper,vmargin={5mm,2mm},hmargin={5mm,5mm}]{geometry}\n',
                    '\\usepackage[linktocpage]{hyperref}\n',
                    '\hypersetup{backref, colorlinks=true}\n',
                    '\\title{ \\textbf{\Huge{HLT Timing Summary}} \\footnote{\large{Documentation at \\url{https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideHLTTimingSummary}}} \\\\ Main Info }\n',
                    '\\author{\Large{CMS Experiment}}\n',
                    '\date{\\today}\n',
                    '\\begin{document}\n',
                    '\maketitle\n',
                    '\\newpage\n',
                    '\clearpage\n'
                    '\\tableofcontents\n',
                    '\\newpage\n',
                    '\\newpage \chapter{Total time for all modules per event} \\newpage \centering \includegraphics[scale=0.6]{totalTimetemp.png}\n']

    names1 = {}
    names4 = {}
    specific = {}
    file1 = TFile(infile,'read')
    for k in file1.GetListOfKeys():
        allnames = k.ReadObj().GetName()
        if 'pathTime_' in allnames:
            pathname = '_'.join(allnames.split('_')[1:])
            if not pathname in names1:
                names1[pathname] = k.ReadObj().GetMean()
        elif 'incPathTime_' in allnames:
            pathname = '_'.join(allnames.split('_')[1:])
            if not pathname in names4:
                names4[pathname] = k.ReadObj().GetMean()

    names2 = dict(sorted(six.iteritems(names1), key=operator.itemgetter(1),reverse=True)[:10])
    names3 = sorted(names2, key=names2.get, reverse=True)
    names5 = dict(sorted(six.iteritems(names4), key=operator.itemgetter(1),reverse=True)[:10])
    names6 = sorted(names5, key=names5.get, reverse=True)

    texfile = open(outfile+'-main.tex', 'w')
    texfile.writelines(texpreamble)
    if os.path.exists(infile.replace('.root','')+'-summary.txt'):
        excludefile = open(infile.replace('.root','')+'-summary.txt', 'r')
        texfile.write('\\newpage \section{Summary} \n \\begin{verbatim} \n')
        for line in excludefile.readlines():
            texfile.write(line+'\n')
        excludefile.close()
        texfile.write('\end{verbatim}')
    texfile.write('\\newpage \\chapter{10 Slowest Paths}\n')
    texfile.write('\section{Average module (in path) time}\n')
    for path in names3:
        texfile.write('\\newpage \subsection{'+ path.replace('_','\_') +'} \centering \includegraphics[scale=0.35]{moduleInPathTimeSummary'+ path.replace('_','') +'temp.png}\n')
        get_plot2(infile,'moduleInPathTimeSummary_'+path)
    texfile.write('\section{Average module (in path) running time}\n')
    for path in names3:
        texfile.write('\\newpage \subsection{'+ path.replace('_','\_') +'} \centering \includegraphics[scale=0.35]{moduleInPathScaledTimeSummary'+ path.replace('_','') +'temp.png}\n')
        get_plot2(infile,'moduleInPathScaledTimeSummary_'+path)
    texfile.write('\section{Per event time for path}\n')
    for path in names3:
        texfile.write('\\newpage \subsection{'+ path.replace('_','\_') +'} \centering \includegraphics[scale=0.6]{pathTime'+ path.replace('_','') +'temp.png}\n')
        get_plot1(infile,'pathTime_'+path)
    texfile.write('\section{Per event incremental time for path}\n')
    for path in names6:
        texfile.write('\\newpage \subsection{'+ path.replace('_','\_') +'} \centering \includegraphics[scale=0.6]{incPathTime'+ path.replace('_','') +'temp.png}\n')
        get_plot1(infile,'incPathTime_'+path)
    texfile.write('\end{document}')
    texfile.close()

    texfile.close()

    get_plot1(infile,'totalTime')



#################################################
def pathsinfo(infile,outfile):
    '''Create the paths info tex file'''
################################################

    texpreamble = ['\documentclass[10pt,a5paper,landscape]{book}\n',
                    '\\usepackage{graphicx}\n',
                    '\\usepackage[a5paper,vmargin={5mm,2mm},hmargin={5mm,5mm}]{geometry}\n',
                    '\\usepackage[linktocpage]{hyperref}\n',
                    '\\usepackage[titles]{tocloft}\n'
                    '\hypersetup{backref, colorlinks=true}\n',
                    '\setlength{\cftsecnumwidth}{4em}\n'
                    '\\title{ \\textbf{\Huge{HLT Timing Summary}} \\footnote{\large{Documentation at \\url{https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideHLTTimingSummary}}} \\\\ Paths Info }\n',
                    '\\author{\Large{CMS Experiment}}\n',
                    '\date{\\today}\n',
                    '\\begin{document}\n',
                    '\maketitle\n',
                    '\\newpage\n',
                    '\\tableofcontents\n',
                    '\\newpage\n']

    names1 = {}
    file = TFile(infile,'read')
    for k in file.GetListOfKeys():
        allnames= k.ReadObj().GetName()
        mean = 1
        if 'moduleInPathScaledTime_' in allnames:
            pathname = '_'.join(allnames.split('_')[1:-1])
            if not pathname in names1:
                names1[pathname] = mean
    names = sorted(names1.keys())

    texfile = open(outfile+'-paths.tex', 'w')
    texfile.writelines(texpreamble)

    texfile.write('\\chapter{Average module (in path) time}\n')
    for path in names:
        texfile.write('\\newpage \section{'+ path.replace('_','\_') +'} \centering \includegraphics[scale=0.35]{moduleInPathTimeSummary'+ path.replace('_','') +'temp.png}\n')	
        get_plot2(infile,'moduleInPathTimeSummary_'+path)
    texfile.write('\\chapter{Average module (in path) running time}\n')
    for path in names:
        texfile.write('\\newpage \section{'+ path.replace('_','\_') +'} \centering \includegraphics[scale=0.35]{moduleInPathScaledTimeSummary'+ path.replace('_','') +'temp.png}\n')	
        get_plot2(infile,'moduleInPathScaledTimeSummary_'+path)
    texfile.write('\\chapter{Failing module (by path)}')
    for path in names:
        texfile.write('\\newpage \section{'+ path.replace('_','\_') +'} \centering \includegraphics[scale=0.35]{failedModule'+ path.replace('_','') +'temp.png}\n')
        get_plot2(infile,'failedModule_'+path)
    texfile.write('\\chapter{Per event time for path}\n')
    for path in names:
        texfile.write('\\newpage \section{'+ path.replace('_','\_') +'} \centering \includegraphics[scale=0.6]{pathTime'+ path.replace('_','') +'temp.png}\n')
        get_plot1(infile,'pathTime_'+path)
    texfile.write('\\chapter{Per event incremental time for path}\n')
    for path in names:
        texfile.write('\\newpage \section{'+ path.replace('_','\_') +'} \centering \includegraphics[scale=0.6]{incPathTime'+ path.replace('_','') +'temp.png}\n')
        get_plot1(infile,'incPathTime_'+path)

    texfile.write('\end{document}')
    texfile.close()


#################################################
def moduleinfo(infile,outfile):
    '''Create the paths info tex file'''
################################################

    texpreamble = ['\documentclass[10pt,a5paper,landscape]{report}\n',
                    '\\usepackage{graphicx}\n',
                    '\\usepackage[a5paper,vmargin={5mm,2mm},hmargin={5mm,5mm}]{geometry}\n',
                    '\\usepackage[linktocpage]{hyperref}\n',
                    '\hypersetup{backref, colorlinks=true}\n',
                    '\\usepackage[titles]{tocloft}\n'
                    '\setlength{\cftsecnumwidth}{4em}\n'
                    '\\title{ \\textbf{\Huge{HLT Timing Summary}} \\footnote{\large{Documentation at \\url{https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideHLTTimingSummary}}} \\\\ Modules Info }\n',
                    '\\author{\Large{CMS Experiment}}\n',
                    '\date{\\today}\n',
                    '\\begin{document}\n',
                    '\maketitle\n',
                    '\\newpage\n',
                    '\\tableofcontents\n',
                    '\\newpage\n']

    names1 = {}
    file = TFile(infile,'read')
    for k in file.GetListOfKeys():
        allnames = k.ReadObj().GetName()
        mean = 1
        if 'moduleTime_' in allnames:
            modname = ''.join(allnames.split('_')[1:])
            if not (('!' in modname) or ('-' in modname)): 
                if not modname in names1:
                    names1[modname] = mean
    names = sorted(names1.keys())

    texfile1 = open(outfile+'-modules.tex', 'w')
    texfile1.writelines(texpreamble)

    texfile1.write('\\chapter{Time per event for module} \n \\newpage')
    for modules in names:
        texfile1.write('\section{'+modules+'}')
        texfile1.write('\centering \includegraphics[scale=0.4]{moduleTime'+ modules +'temp.png}\n')
        get_plot1(infile,'moduleTime_'+modules)
    texfile1.write('\end{document}')
    texfile1.close()

    texfile2 = open(outfile+'-runningModules.tex', 'w')
    texfile2.writelines(texpreamble)
    texfile2.write('\\chapter{Running Time per event for module} \n \\newpage')
    for modules in names:
        texfile2.write('\section{'+modules+'}')
        texfile2.write('\centering \includegraphics[scale=0.45]{moduleScaledTime'+modules+'temp.png}\n')    
        get_plot1(infile,'moduleScaledTime_'+ modules)

    texfile2.write('\end{document}')
    texfile2.close()

###############################################################
def specificpathinfo(infile, outfile, path):
    ''' Creates an specific path info tex file'''
##############################################################
    texpreamble = ['\documentclass[10pt,a5paper,landscape]{report}\n',
                    '\\usepackage{graphicx}\n',
                    '\\usepackage[a5paper,vmargin={5mm,2mm},hmargin={5mm,5mm}]{geometry}\n',
                    '\\usepackage[linktocpage]{hyperref}\n',
                    '\\usepackage[titles]{tocloft}\n'
                    '\hypersetup{backref, colorlinks=true}\n',
                    '\setlength{\cftsubsecnumwidth}{4em}\n'
                    '\\title{ \\textbf{\Huge{HLT Timing Summary}} \\footnote{\large{Documentation at \\url{https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideHLTTimingSummary}}} \\\\ Main Info + ' + path.replace('_','\_') +' info }\n',
                    '\\author{\Large{CMS Experiment}}\n',
                    '\date{\\today}\n',
                    '\\begin{document}\n',
                    '\maketitle\n',
                    '\\tableofcontents \n'
                    '\\newpage\n \\chapter{Main Info} \n',
                    '\\newpage \centering \section{Total time for all modules per event} \includegraphics[scale=0.6]{totalTimetemp.png}\n']

    texfile = open(outfile+'-'+path+'.tex', 'w')
    texfile.writelines(texpreamble)
    get_plot1(infile,'totalTime')

    names = {}
    file1 = TFile(infile,'read')
    for k in file1.GetListOfKeys():
        allnames = k.ReadObj().GetName()
        if 'moduleInPathScaledTime_' in allnames:
            pathname = '_'.join(allnames.split('_')[1:-1])
            if not pathname in names:
                names[pathname] = {}
    for pathnames in names.keys():
        histo = file1.Get('moduleInPathTimeSummary_'+pathnames)
        for bin in range(histo.GetNbinsX()):
            label = histo.GetXaxis().GetBinLabel(bin+1)
            names[pathnames][bin+1] = label

    for pathname in names:
        if path in pathname:
            texfile.write('\chapter{' + path.replace('_','\_')+ ' Info} \n')
            texfile.write('\\newpage \section{Average module in '+ path.replace('_','\_') +' time} \centering \includegraphics[scale=0.35]{moduleInPathTimeSummary'+ path.replace('_','') +'temp.png}\n')
            get_plot2(infile,'moduleInPathTimeSummary_'+path)
            texfile.write('\\newpage \section{Average module in '+ path.replace('_','\_') +' running time} \centering \includegraphics[scale=0.35]{moduleInPathScaledTimeSummary'+ path.replace('_','') +'temp.png}\n')
            get_plot2(infile,'moduleInPathScaledTimeSummary_'+path)
            texfile.write('\\newpage \section{Per event time for '+ path.replace('_','\_') +'} \centering \includegraphics[scale=0.6]{pathTime'+ path.replace('_','') +'temp.png}\n')
            get_plot1(infile,'pathTime_'+path)
            texfile.write('\\newpage \section{Per event incremental time for '+ path.replace('_','\_') +'} \centering \includegraphics[scale=0.6]{incPathTime'+ path.replace('_','') +'temp.png}\n')
            get_plot1(infile,'incPathTime_'+path)
            texfile.write('\section{Running time per event for '+path.replace('_','\_')+'}')	
            for modules in names[path].values():
                texfile.write('\subsection{'+ modules +'} \centering \includegraphics[scale=0.6]{moduleInPathScaledTime'+ path.replace('_','') + modules +'temp.png}\n')
                get_plot1(infile,'moduleInPathScaledTime_'+ path +'_'+ modules)
    texfile.write('\\end{document}')
    texfile.close()




###################################################################
def get_plot1(file,allnames):
    ''' Function to create the plot and save it as.png file '''
###################################################################
    file = TFile(file,'read')
    histo = file.Get(allnames)
    can = TCanvas('can', '', 800,600)
    can.cd()
    histo.UseCurrentStyle()
    histo.Draw()
    can.SetBorderMode(0)
    can.SetFillColor(kWhite)
    can.SaveAs(allnames.replace('_','')+'temp.png')
    del can



###################################################################
def get_plot2(infile,allnames):
    ''' Function to create the plot and save it as.png file '''
###################################################################
    file1 = TFile(infile,'read')
    histo = file1.Get(allnames)
    can = TCanvas('can', '', 1600,1000)
    can.cd()
    histo.UseCurrentStyle()
    histo.Draw()
    if histo.GetNbinsX() > 50:
        histo.GetXaxis().SetLabelSize(0.02)
    else:
        histo.GetXaxis().SetLabelSize(0.03)
    can.SetBorderMode(0)
    can.SetBorderSize(0)
    can.SetFillColor(kWhite)
    can.SetBottomMargin(0.4)
    can.SaveAs(allnames.replace('_','')+'temp.png')
    del can

###############################################################
def main(argv):
###############################################################
    print("\nPython script that creates Timing Summary pdf files")
    print("For more info, please contact Alejandro Gomez")
    print("email: alejandro.gomez@cern.ch\n")

    infile = None
    outfile = None
    path = None
    call_maininfo = False
    call_pathsinfo = False
    call_modulesinfo = False
    call_specificinfo = False
    try:
        opts, args = getopt.getopt(argv, 'hi:o:tbpms:', ['help', 'input=', 'output='])
        if not opts:
            print('No options supplied')
            usage()
    except getopt.GetoptError as e:
        print(e)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('h', '--help'):
            usage()
            sys.exit(2)
        elif opt == '-b':
            print('Running in batch mode') 
        elif opt in ('-i', '--input'):
            infile = arg
            outfile = infile.replace('.root','')
        elif opt in ('-o', '--output'):
            outfile = arg
        elif opt == '-t':
            call_maininfo = True
        elif opt == '-p':
            call_pathsinfo = True
        elif opt == '-m':
            call_modulesinfo = True
        elif opt == '-s':
            path = arg
            call_specificinfo = True
        else:
            usage()
            sys.exit(2)


    if call_maininfo:
        print('Creating the Main Info Timing Summary pdf')
        print('Creating plots...')
        maininfo(infile,outfile)
        print('Compiling tex file......')
        subprocess.call(['pdflatex', '-interaction=batchmode', outfile+'-main.tex'])
        print('Verifing......') 
        subprocess.call(['pdflatex', '-interaction=batchmode', outfile+'-main.tex'])    #twice for better compilation
        print('Removing temp files.........')
        os.remove(outfile+'-main.aux')
        os.remove(outfile+'-main.log')
        os.remove(outfile+'-main.out')
        os.remove(outfile+'-main.tex')
        os.remove(outfile+'-main.toc')
        for filename in glob.glob('*temp.png'):
            os.remove(filename)
        print('{0}-main.pdf is done'.format(outfile))

    if call_pathsinfo:
        print('Creating the Paths Info Timing Summary pdf')
        print('This process takes awhile... please be patient')
        print('Creating plots...')
        pathsinfo(infile,outfile)
        print('Compiling tex file......')
        subprocess.call(['pdflatex', '-interaction=batchmode', outfile+'-paths.tex'])
        print('Verifing......') 
        subprocess.call(['pdflatex', '-interaction=batchmode', outfile+'-paths.tex'])    #twice for better compilation
        print('Removing temp files.........')
        os.remove(outfile+'-paths.aux')
        os.remove(outfile+'-paths.log')
        os.remove(outfile+'-paths.out')
        os.remove(outfile+'-paths.tex')
        os.remove(outfile+'-paths.toc')
        for filename in glob.glob('*temp.png'):
            os.remove(filename)
        print('{0}-paths.pdf is done'.format(outfile))

    if call_modulesinfo:
        print('Creating the Modules Info Timing Summary pdf')
        print('This process takes awhile... please be patient')
        print('Creating plots...')
        moduleinfo(infile,outfile)
        print('Compiling tex file......')
        subprocess.call(['pdflatex', '-interaction=batchmode', outfile+'-modules.tex'])
        subprocess.call(['pdflatex', '-interaction=batchmode', outfile+'-runningModules.tex'])
        print('Verifing......')
        subprocess.call(['pdflatex', '-interaction=batchmode', outfile+'-modules.tex'])    #twice for better compilation
        subprocess.call(['pdflatex', '-interaction=batchmode', outfile+'-runningModules.tex'])    #twice for better compilation
        print('Removing temp files.........')
        os.remove(outfile+'-modules.aux')
        os.remove(outfile+'-runningModules.aux')
        os.remove(outfile+'-modules.log') 
        os.remove(outfile+'-runningModules.log') 
        os.remove(outfile+'-modules.out')
        os.remove(outfile+'-runningModules.out')
        os.remove(outfile+'-modules.tex')
        os.remove(outfile+'-runningModules.tex')
        os.remove(outfile+'-modules.toc')
        os.remove(outfile+'-runningModules.toc')
        for filename in glob.glob('*temp.png'):
            os.remove(filename) 
        print('{0}-modules.pdf is done'.format(outfile)) 
        print('{0}-runningModules.pdf is done'.format(outfile)) 

    if call_specificinfo:
        print('Creating the Main Info + '+ path +' Timing Summary pdf')
        print('This process takes awhile... please be patient')
        print('Creating plots...')
        specificpathinfo(infile,outfile,path)
        print('Compiling tex file......')
        subprocess.call(['pdflatex', '-interaction=batchmode', outfile+'-'+path+'.tex'])
        print('Verifing......')
        subprocess.call(['pdflatex', '-interaction=batchmode', outfile+'-'+path+'.tex'])    #twice for better compilation
        print('Removing temp files.........')
        os.remove(outfile+'-'+path+'.aux')
        os.remove(outfile+'-'+path+'.log')
        os.remove(outfile+'-'+path+'.out')
        os.remove(outfile+'-'+path+'.tex')
        os.remove(outfile+'-'+path+'.toc')
        for filename in glob.glob('*temp.png'):
            os.remove(filename)
        print('{0}-'.format(outfile)+path+'.pdf is done')


#######################################################
if __name__ =='__main__':
#######################################################
    main(sys.argv[1:])
