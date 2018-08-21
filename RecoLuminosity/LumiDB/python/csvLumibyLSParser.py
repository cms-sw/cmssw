from __future__ import print_function
# Note: this is specifically to parse a .csv file generated from a command like
# pixelLumiCalc.py lumibyls -i json_DCSONLY_pp.txt --hltpath "HLT_Photon75_CaloIdVL_IsoL*" -o myHLTtest.out
# format: Run,LS,HLTpath,L1bit,HLTpresc,L1presc,Recorded(/ub),Effective(/ub)
import csv
import re
def is_intstr(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
class csvLumibyLSParser(object):
    def __init__(self,filename,RunX):
        self.__result={}
        self.__strresult={}
        self.__filename=filename
        self.__RunX=RunX
        csvReader=csv.reader(open(filename),delimiter=',')
        oldRun=0
        runnumber=0
        ldict = {}
        llist = []
        NonValidLumi = 0
        lastLumi = 0
        for row in csvReader:
            field0=str(row[0]).strip()
            fieldsplit=re.split(':',field0)
            runstring = fieldsplit[0]
            try:
                field1=str(row[1]).strip()
                fieldsplit=re.split(':',field1)
                lsstring = fieldsplit[0]
            except Exception as e:
                lsstring='1' # for list with run number only, fake lsnum
            if not is_intstr(runstring) or not  is_intstr(lsstring):
                continue
            runnumber=int(runstring)
            lsnumber=int(lsstring)

            if runnumber != oldRun:
                if oldRun>0:
                    self.__result[oldRun]=ldict
                    ldict = {}
                    oldRun = runnumber
                    lastLumi = 0
                    NonValidLumi = 0
                else:
                    oldRun = runnumber

            try:
                if RunX=='Run2':
                    delivered, recorded = float( row[4] ), float( row[5] )
                if RunX=='Run1':
                    delivered, recorded = float( row[5] ), float( row[6] )
            except:
                print('Record not parsed, Run = %d, LS = %d' % (runnumber, lsnumber))                

# Commented out... If there is no value, there is no interpolation now...
#            if recorded>0 :
#                lastLumi = recorded
#                if NonValidLumi>0:
#                    # have to put real values in lumi list
#                    for lnum in llist:
#                        elems = [delivered, recorded]
#                        ldict[lnum] = elems
#                    NonValidLumi=0
#                    llist = []
#            else:
#                if lastLumi>0:
#                    recorded = lastLumi
#                else:
#                   # have to save lumi sections to fill once we get a non-zero lumi value
#                   llist.append(lsnumber)
#                   NonValidLumi=1

            elems = [ delivered,recorded ]
            ldict[lsnumber]=elems

        self.__result[runnumber]=ldict #catch the last one

    def runs(self):
        return self.__result.keys()
    def runsandls(self):
        '''return {run:lslist}
        '''
        return self.__result
#    def runsandlsStr(self):
#        '''return {'run':lslist}
#        '''
#        return self.__strresult
    def numruns(self):
        return len(self.__result)
    def numls(self,run):
        return len(self.__result[run])

if __name__ == '__main__':
    result={}
    #filename='../test/lumi_by_LS_all.csv'
    filename='test.csv'
    s=csvLumibyLSParser(filename)
    print('runs : ',s.runs())
    print('full result : ',s.runsandls())
    #print 'str result : ',s.runsandlsStr()
    print('num runs : ',s.numruns())
    #print 'numls in run : ',s.numls(135175)

