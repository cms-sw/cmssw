#! /usr/bin/env python
# Author:       Jake Herman
# Date:         27 June 2011
#Purpose: Script reads in a list of .pkl files from sys.argv and combines them to intialize an HVMapNoise object and saves pickles of HV assignment dictionaries for diff ration on and off. Makes plots of diff and ratios. Quick and dirty for testing purposes.
#TO DO: Add boolean trips to script to filter out methods not chosen by user 

import pickle, os, sys

sys.path.append('Classes/')

from optparse import OptionParser

#intialize and fill command line interface
parser = OptionParser(usage = '''Usage: ./MakeAssignments.py [Options] [Option Arguments] [NoiseDictionaries]
Example:
        ./MakeAssignments.py -d .5 PedestalRuns/*
           Can use bash wild card expansions or individual file names''')

parser.add_option('-p', dest ='name', default = 'Results',  help = "The name of the directory in 'Output/<dataset>' to which results will be saved, the directory name defaults to 'Results'" )
parser.add_option('-D',action = 'store', type = 'float', dest = 'StrictDiffCut', help = "Tells the strict difference method to be done and the sets the cut level passed by the user")
parser.add_option('-d', action = 'store', type = 'float', dest = 'RelaxDiffCut', help = "Tells the relax difference method to be done to augment the strict difference method and sets the cut level to that passed by the user")
parser.add_option('-x', action = 'store', type = 'float', nargs = 2, dest = 'RatioOffCuts', help = "Tells the off ratio method to be done and sets the cuts to that passed by the user (HV1/OFF  first), takes two arguments will raise an error if two are not given")
parser.add_option('-o', action = 'store', type = 'float', nargs = 2, dest = 'RatioOnCuts', help = "Tells the on ratio method to be done and sets the cuts to that passed by the user (ON/HV1 first), takes two arguments will raise an error if two are not givene")
parser.add_option('-f', action = 'store', type = 'float', dest = 'OnOffCut',  help = "Tells the Off - On diff method to be done for further classification of undetermined modules and sets the cut to that passed by user")
(Commands, args) = parser.parse_args()

from HVMapToolsClasses import HVMapNoise, HVAnalysis

#Make list of methods to be done for intialization of HVAnalysis
AnaList = []
if Commands.StrictDiffCut is not None:
   AnaList.append('DIFF')
if Commands.RelaxDiffCut is not None:
   AnaList.append('DIFF')
if Commands.RatioOffCuts is not None:
   AnaList.append('ROFF')
if Commands.RatioOnCuts is not None:
   AnaList.append('RON')
if Commands.OnOffCut is not None:
   AnaList.append('OF')

#intialize HVMapNoise Class
Noise = HVMapNoise(Commands.name)

#default for dataset
dataset = 'No_Data'

for path in args:
   try:
      Noise.AddFromPickle(path)

      if dataset != 'No_Data' and dataset != path.split('/')[1]:
         dataset = path.split('/')[1]
         print "Warning you are using multiple datasets"

      else:
         dataset = path.split('/')[1]
         
   except:
      print path,"not added"


Analysis = HVAnalysis(Commands.name, Noise, AnaList)

#make directory structure and intialize results directory
savepath = 'Output/'+dataset +'/' + Commands.name

if dataset != 'No_Data':
   if Commands.name not in os.listdir('Output/' + dataset):
      os.system('mkdir %s'%savepath)

#Keep track of the analyses done
Methods = []

#Make strict diff assignments
if Commands.StrictDiffCut is not None:
   Methods.append("Strict Diff")
   
   Assignments  = Analysis.StrictDiffMethod(Commands.StrictDiffCut)
   
   #save assignments
   file = open(savepath + '/StrictDiffMethod_' + str(Commands.StrictDiffCut) +'.pkl','wb')
   pickle.dump(Assignments,file)
   file.close()
   Analysis.MkList(Assignments, savepath +'/StrictDiffList_' + str(Commands.StrictDiffCut))
   Analysis.Mktxt(Assignments, savepath +'/StrictDiffTkParse_' + str(Commands.StrictDiffCut))

   Analysis.PrintResults(Assignments,"Strict Difference Method")


#Relaxed Diff -------------------------------
if Commands.RelaxDiffCut is not None:
   Methods.append("Relaxed Diff")
   
   UndetCate = Analysis.DiffMethod(Commands.RelaxDiffCut)
   
   #Keep only the undetermined modules for categorization
   for detID in UndetCate.keys():
      if UndetCate[detID] == 'HV1' or UndetCate[detID] == 'HV2' or UndetCate[detID] == 'Masked':
         UndetCate.pop(detID,'')
  
   #Flag misconfigured  and turned off detIDs for 'HV1' and 'HV2'
   for RunT in Noise.Noise.keys():
      if RunT == 'HV1' or RunT == 'HV2':
         for SubD in Noise.Noise[RunT].keys():
            for detID in Noise.Noise[RunT][SubD].keys():
               if detID in UndetCate.keys():
                  n = 0
                  for APVID in Noise.Noise[RunT][SubD][detID].keys():
                     if Noise.Noise[RunT][SubD][detID][APVID] < 1:
                        if n == 0:
                           if Noise.Noise[RunT][SubD][detID][APVID] != 0:
                              if 'HV1' not in UndetCate[detID] and 'HV2' not in UndetCate[detID]:
                                 UndetCate.update({detID : 'Misconfigured'})
                                 n += 1
                              elif 'misconfigured' not in UndetCate[detID]:
                                 UndetCate.update({detID : UndetCate[detID].partition('unresponsive')[0] + 'misconfigured APVs'})
                                 n += 1
                           else:
                              if 'HV1' not in UndetCate[detID] and 'HV2' not in UndetCate[detID]:
                                 UndetCate.update({detID : 'Turned off'})
                                 n += 1
                              elif 'off' not in UndetCate[detID]:
                                 UndetCate.update({detID : UndetCate[detID].partition('unresponsive')[0] + 'turned off APVs'})
                                 n += 1
  
   #Flag Modules which were misconfigure or turned off in HVON or HVOFF runs and are unresponsive and can therefore not be tagged as cross talking or no HV!!! Need to protect from insufficient data
   for RunT in Noise.Noise.keys():
      if RunT == 'HV1' or RunT == 'HV2':
         for SubD in Noise.Noise[RunT].keys():
            for detID in Noise.Noise[RunT][SubD].keys():
               if detID in UndetCate.keys():
                  for APVID in Noise.Noise[RunT][SubD][detID].keys():
                     if Noise.Noise[RunT][SubD][detID][APVID] < 1:
                        if UndetCate[detID] == 'Unresponsive':
                           if Noise.Noise[RunT][SubD][detID][APVID] == 0:
                              UndetCate.update({detID:'Unresponsive with turned off APVs in HVON or HVOFF run'})
                           else:
                              UndetCate.update({detID:'Unresponsive with misconfigured APVs in HVON or HVOFF run'})

   
  
   #On Off diff method-------------------------
   if Commands.OnOffCut is not None:
      Methods.append("On Off Diff")
      
      UnresDict = Analysis.OnOffDiffMethod(Commands.OnOffCut)
   
      for detID in UndetCate.keys():
         if UndetCate[detID] =='Unresponsive':
            UndetCate.update({detID: UnresDict[detID]})
      
   #Compile category statistics              
   CateStat = {}

   for detID in UndetCate.keys():
      if UndetCate[detID] not in CateStat.keys():
          CateStat.update({UndetCate[detID] : 1})
      else:
         CateStat.update({UndetCate[detID] : CateStat[UndetCate[detID]] +1})

   print '---------------------Undetermined module catagorization---------------------'
   for cate in CateStat.keys():
      print cate,':\t', CateStat[cate]
      
   
   #save------------------------------------------------------
   file = open(savepath + '/UndeterminedCategories_' + str(Commands.RelaxDiffCut) + '_'+ str(Commands.OnOffCut)+'.pkl','wb')
   pickle.dump(UndetCate,file)
   file.close()

   
   Analysis.MkList(UndetCate, savepath +'/UndeterminedCategoriesList_' + str(Commands.RelaxDiffCut) + '_' + str(Commands.OnOffCut))

   for category in CateStat.keys():
      Analysis.Mktxt(UndetCate, savepath +'/'+ category.replace(' ','_')+'_TkParse_'+ str(Commands.RelaxDiffCut)+'_'+str(Commands.OnOffCut) , category)
      Analysis.MkSList(UndetCate, category, savepath + '/' +category.replace(' ','_') + '_AliasList_'+ str(Commands.RelaxDiffCut)+'_'+str(Commands.OnOffCut), 'A')
      


#Ratio ON---------------------------
if Commands.RatioOnCuts is not None:
   Methods.append("On Ratio")
   
   RatON = Analysis.RatioMethod('ON',Commands.RatioOnCuts[0], Commands.RatioOnCuts[1] )
   Analysis.PrintResults(RatON,"HV ON Ratio Method")

   file = open(savepath + '/RatioOn_' + str(Commands.RatioOnCuts[0]) + '_' + str(Commands.RatioOnCuts[1]) +'.pkl','wb')
   pickle.dump(RatON,file)
   file.close()

   Analysis.Mktxt(RatON, savepath + '/RatioOnTKParse_' + str(Commands.RatioOnCuts[0]) + '_' + str(Commands.RatioOnCuts[1]))
   Analysis.MkList(RatON,savepath + '/RatioOnList_' + str(Commands.RatioOnCuts[0]) + '_' + str(Commands.RatioOnCuts[1]))



#Ratio OFF---------------------------
if Commands.RatioOffCuts is not None:
   Methods.append("Off Ratio")
   
   RatOFF = Analysis.RatioMethod('OFF',Commands.RatioOffCuts[0], Commands.RatioOffCuts[1])
   Analysis.PrintResults(RatOFF,"HV OFF Ratio method")

   file = open(savepath + '/RatioOff_' + str(Commands.RatioOffCuts[0]) + '_' + str(Commands.RatioOffCuts[1])+ '.pkl','wb')
   pickle.dump(RatOFF,file)
   file.close()

   Analysis.Mktxt(RatOFF, savepath + '/RatioOffTKParse_' + str(Commands.RatioOffCuts[0])+ '_' + str(Commands.RatioOffCuts[1]))
   Analysis.MkList(RatOFF,savepath + '/RatioOffList_' + str(Commands.RatioOffCuts[0]) + '_' + str(Commands.RatioOffCuts[1]))


#report methods
if 'args' in dir():
   print "-----------------Analyses Performed-------------------"

   for method in Methods:
      print method
