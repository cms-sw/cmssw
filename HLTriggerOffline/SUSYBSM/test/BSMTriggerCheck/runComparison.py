#! /usr/bin/env python
import os
os.system("make clean; make; \\rm *.log log.list")


############################################

#dir1='TriggerValidation_223_HLT'
#dir2='TriggerValidation_224_HLT'
#out='223_vs_224'

#samples=['LM1']
#prefix1 = "histo_"
#prefix2 = "histo_"
#sufix1 = "_IDEALV11"
#sufix2 = "_IDEALV11_v1"
#label1 = "LM1_223"
#label2 = "LM1_224"

############################################

#dir1='TriggerValidation_224_HLT'
#dir2='TriggerValidation_300pre2_HLT'
#out='224_vs_300pre2'

#samples=['LM1']
#prefix1 = "histo_"
#prefix2 = "histo_"
#sufix1 = "_IDEALV11_v1"
#sufix2 = "_IDEALV9"
#label1 = "LM1_223"
#label2 = "LM1_300pre2"

############################################

#dir1='TriggerValidation_224_HLT'
#dir2='TriggerValidation_300pre6_HLT'
#out='224_vs_300pre6'

#samples=['LM1']
#prefix1 = "histo_"
#prefix2 = "histo_"
#sufix1 = "_IDEALV11_v1"
#sufix2 = "_IDEAL_30x_v1"
#label1 = "LM1_223"
#label2 = "LM1_300pre6"

############################################

dir1='/afs/cern.ch/user/c/chiorbo/scratch0/SUSY_2007/TriggerValidation/TriggerValidation_DQM_312_commit_V00-06-00/src/HLTriggerOffline/SUSYBSM/test'
dir2='/afs/cern.ch/user/c/chiorbo/scratch0/SUSY_2007/TriggerValidation/TriggerValidation_DQM_312_commit_V00-06-00/src/HLTriggerOffline/SUSYBSM/test'
out='mc1_vs_mc2'

samples=['_HLT']
prefix1 = "DQM_V0001"
prefix2 = "DQM_V0001"
sufix1 = "_R000000001"
sufix2 = "_R000000001_2"
label1 = "HLT"
label2 = "HLT"

############################################

os.system('mkdir html/'+out)
#create html index page
os.system('cp html/template/index.html html/'+out+'/index.html')


#create the cover page
inputhtml  = open('html/template/beginning.html')
outputhtml = open('html/'+out+'/cover.html','w')
for line in inputhtml:
    # remove .root
    if line.find('<!-- Here python will write the name of first release -->') != -1: outputhtml.write(dir1) 
    # remove .root
    elif line.find('<!-- Here python will write the name of second release -->') != -1: outputhtml.write(dir2)
    else: outputhtml.write(line)
    continue
inputhtml.close()
outputhtml.close()

#create the menu page
os.system('cp html/template/menu_beginning.html html/'+out+'/menu.html')
for sample in samples:
    tmp1 = open('tmp.html','w')
    tmp2 = open('html/template/menu_body.html')
    for line in tmp2:
        if line.find('thissample') != -1:
            newline = line.replace('thissample',sample)
            tmp1.write(newline)
        else: tmp1.write(line)
        continue
    tmp1.close()
    tmp2.close()
    os.system('more tmp.html >> html/'+out+'/menu.html')
    os.system('rm tmp.html')
    continue
os.system('more html/template/menu_end.html >> html/'+out+'/menu.html')

#run the code for each sample
for sample in samples:
    file1   = dir1+'/'+prefix1+sample+sufix1+'.root'
    file2   = dir2+'/'+prefix2+sample+sufix2+'.root'
    outputfile =  'outputfile.root'

    #create html page for this sample
    inputhtml  = open('html/template/comp_beginning.html')
    os.system('mkdir html/'+out+'/'+sample)
    outputhtml = open('html/'+out+'/'+sample+'/comparison.html','w')

    # add right version names in the html
    for line in inputhtml:
        if line.find('<!-- Here python will write the name of first release -->') != -1: outputhtml.write(dir1) 
        elif line.find('<!-- Here python will write the name of second release -->') != -1: outputhtml.write(dir2) 
        elif line.find('<!-- Here python will write the name of the model -->') != -1: outputhtml.write(sample)

        elif line.find('thissample') != -1:
            newline = line.replace('thissample',sample)
            outputhtml.write(newline)
        else: outputhtml.write(line)
        continue
    inputhtml.close()
    outputhtml.close()
    
    # run the comparison 
    os.system('./triggerComparison.x  -File1='+file1+' -File2='+file2+' -OutputFile='+outputfile+' -label1='+label1+' -label2='+label2)
    # for old names
    #    os.system('./triggerComparison.x  --oldL1names -File1='+file1+' -File2='+file2+' -OutputFile='+outputfile+' -label1='+label1+' -label2='+label2)
    os.system('mv HLTcomparison.log html/'+out+'/'+sample)
    os.system('mv L1comparison.log html/'+out+'/'+sample)
    
    # mv root file to the html directory
    os.system('mv '+outputfile+' html/'+out+'/'+sample)

    # add eff and residual pulls to the html
    os.system('more html/template/comp.html >> html/'+out+'/'+sample+'/comparison.html')

    # link the compatibility maps
    os.system('more compatibility.html >> html/'+out+'/'+sample+'/comparison.html')

    # create jpg files
    os.system("ls *eps > listeps.log")
    listeps = open("listeps.log")
    for epsfile in listeps: os.system("convert \""+epsfile[:-1]+"\" \""+epsfile[:-4]+"jpg\"")
    thefile = open('html/'+out+'/'+sample+'/comparison.html',"r+")
    # link HLT files    

    #thefile.seek(0,2)
    #thefile.write('<tr><td><center><table>\n')
    #listeps.seek(0)
    #for epsfile in listeps:
    #    if(epsfile.find('HLT') != -1):   #this is a plot of a trigger path
    #        tmp1 = open('html/template/addplot.html')
    #        for line in tmp1:
    #            newline = line.replace('triggerpath',epsfile[:-5])
    #            thefile.write(newline+'\n')
    #            continue
    #        continue
    #    continue
    #thefile.write('</table></center></td>\n')

    # link L1 files 
    #thefile.write('<td><center><table>\n')
    #listeps.seek(0)
    #for epsfile in listeps:
    #  if(epsfile.find('L1') != -1):   #this is a plot of a trigger path
    #    if(epsfile.find('A_') != -1):   #this is a plot of a trigger path
    #        tmp1 = open('html/template/addplot.html')
    #        for line in tmp1:
    #            newline = line.replace('triggerpath',epsfile[:-5])
    #            thefile.write(newline+'\n')
    #            continue
    #        continue
    #    continue
    #thefile.write('</table></center></td></tr>\n')
    #thefile.close()

    # write end of the comparison web page 
    os.system('more html/template/end.html >> html/'+out+'/'+sample+'/comparison.html')
    #  move all eps and jpg files in the proper directory
    os.system('mv *jpg html/'+out+'/'+sample+'/')
    os.system('mv *eps html/'+out+'/'+sample+'/')
    continue
os.system('\\rm listeps.log')
