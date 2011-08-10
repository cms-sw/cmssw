#! /usr/bin/env python
#Author: Jacob Herman
#Date 7/24/2011
#Purpose: This script is for combining HV map assignment dictionaries created by the 'MakeAssignments' script. The assignments can either be combined constructively where it must only be assigned positively (including weak assignments such as 'HV1 with..') in one dictionary to take a positive assignment in the combined dictionary. Or destructively where assignments must be consistant in both dictionaries for a positive assignment in the combined. The default is constructive.

from optparse import OptionParser
import pickle, os, sys

sys.path.append('Classes/')


#intialize command line interface
parser = OptionParser()
parser = OptionParser(usage='''Usage: ./CombineAssignments.py [options] [Assignment Dictionaries]
Examples:
./CombineAssignments.py YourAssignmentDictionary1.pkl YourAssignmentDictionary2.pkl ...
   No options selected in this example'''
)                      
parser.add_option('-d', action = 'store_false', default = True, dest = 'constructive', help = 'Tells the combination process to be done destructively in which an assignment must be consistant between both input dictionaries to be positively assigned in the combined dictionary, the default is constructive assignments')
parser.add_option('-n', default = "Combined", dest = 'name', help = "Sets the name of the pickle file to which the combined assignments dictionary will be dumped, defaults to 'Combined'")
parser.add_option('-p', default = 'Output' , dest = 'path', help = "Sets path that you want to save the resulting combined dictionary pickle file and flipped detIDs list to under 'Output/', defaults to 'Output/'")


(Commands, args) = parser.parse_args()

from HVMapToolsClasses import HVAnalysis
Analysis = HVAnalysis('printer',{},[],False)

#Make sure the user passed arguments
if 'args' in dir():

    if Commands.path != 'Output':

        Commands.path = 'Output/' + Commands.path
        
        if not Commands.path.endswith('/'):
            if Commands.path not in os.listdir('Output'):
                os.system("mkdir %s"%("Output/" + Commands.path))
            Commands.path = Commands.path + '/'
        
        
        elif Commands.path.replace('/','') not in os.listdir('Output'):
                os.system("mkdir %s"%("Output/" + Commands.path))
    else:
        Commands.path = Commands.path +'/'

    
    
    #intialize Output file and combined dictionary
    Outfile = open(Commands.path + Commands.name  + '.pkl','wb')
    Combined = {}
    
    #Constructive Combination
    if Commands.constructive:

        flipdets = {}
        
        counter = 0
        for path in args:
            counter += 1
            
            try:
           
                file = open(path)
                Assigned = pickle.load(file)

                for detID in Assigned.keys():

                    if detID not in Combined.keys():
                        Combined.update({detID:Assigned[detID]})
                    
                    elif 'HV1' in Assigned[detID]:
                        if 'HV2' in Combined[detID]:
                            print "Notice:", detID, "has conflicting assignments. Reassigned from", Combined[detID], "to",Assigned[detID]
                            if detID not in flipdets.keys():
                                flipdets.update({detID:1})
                            else:
                                flipdets.update({detID: flipdets[detID] + 1})
                        Combined.update({detID:Assigned[detID]})

                    elif 'HV2' in Assigned[detID]:
                        if 'HV1' in Combined[detID]:
                            print "Notice:", detID, "has conflicting assignments. Reassigned from", Combined[detID], "to",Assigned[detID]
                            if detID not in flipdets.keys():
                                flipdets.update({detID:1})
                            else:
                                flipdets.update({detID: flipdets[detID] + 1})
                                
                        Combined.update({detID:Assigned[detID]})

                if len(flipdets.keys()) > 0:
                    print len(flipdets.keys()), "total"

                    Analysis.Mktxt(flipdets, Commands.path+'/FlipDets_tkparse', '1')
                    file = open(Commands.path + "FlippedIDsList.txt","wb")
                    afile = open('data/StripDetIDAlias.pkl','rb')
                    alias = pickle.load(afile)
                    file.write("|detID|\tAlias|\t# of times flipped|\n")
            
                    for detID in flipdets.keys():
                        file.write('|'+str(detID) +'|\t' +str(alias[int(detID)]).replace('set([','').replace('])','')+ '|\t' +str(flipdets[detID])+'|\n')
                    Analysis.Mktxt(flipdets,Commands.path + "FlipDets_TkParse.txt", 1)
                    
            except:
                print path, 'could not be added'

        if counter < 2:
            print 'Warning: less than 2 dictionaries successfully combined'

        pickle.dump(Combined,Outfile)
        print counter,'dictionaries have been constructively combined and stored in ', Commands.path + Commands.name  + '.pkl'


    #destructive Combination
    else:

        counter = 0
        for path in args:
            counter += 1
            
            try:
                file = open(path)
                Assigned = pickle.load(file)

                for detID in Assigned.keys():

                    if detID not in Combined.keys():
                        Combined.update({detID:Assigned[detID]})
                    
                    elif 'HV1' in Assigned[detID]:
                        if Combined[detID] == Assigned[detID]:
                            Combined.update({detID:Assigned[detID]})
                        else:
                            Combined.update({detID: 'Undetermined'})

                    elif 'HV2' in Assigned[detID]:
                        if Combined[detID] == Assigned[detID]:
                            Combined.update({detID:Assigned[detID]})
                        else:
                            Combined.update({detID: 'Undetermined'})

            except:
                print path, 'could not be added'

        if counter < 2:
            print 'Warning: less than 2 dictionaries successfully combined'

        pickle.dump(Combined,Outfile)
        print counter,'dictionaries have been destructively combined and stored in: ', Commands.path + Commands.name  + '.pkl'


else:
    print 'Insuffcient arguments passed'


Analysis.Mktxt(Combined,Commands.path + 'Combined_Tkparse')
if Commands.constructive:
    Analysis.PrintResults(Combined, str(counter) + " Constructively Combined Results")
else:
    Analysis.PrintResults(Combined, str(counter) + " Detructively Combined Results")
