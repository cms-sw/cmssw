import csv
def is_intstr(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
class csvSelectionParser(object):
    def __init__(self,filename):
        self.__result={}
        self.__strresult={}
        self.__filename=filename
        csvReader=csv.reader(open(filename),delimiter=',')
        for row in csvReader:
            field0=str(row[0]).strip()
            try:
                field1=str(row[1]).strip()
            except Exception,e:
                field1='1' # for list with run number only, fake lsnum
            if not is_intstr(field0) or not  is_intstr(field1):
                continue
            runnumber=int(field0)
            lsnumber=int(field1)
            if self.__result.has_key(runnumber):
                self.__result[runnumber].append(lsnumber)
            else:
                self.__result[runnumber]=[lsnumber]            
        for k,lsvalues in self.__result.items():
            lsvalues.sort()
            self.__strresult[k]=[str(x) for x in lsvalues]
    def runs(self):
        return self.__result.keys()
    def runsandls(self):
        '''return {run:lslist}
        '''
        return self.__result
    def runsandlsStr(self):
        '''return {'run':lslist}
        '''
        return self.__strresult
    def numruns(self):
        return len(self.__result.keys())
    def numls(self,run):
        return len(self.__result[run])
        
if __name__ == '__main__':
    result={}
    #filename='../test/lumi_by_LS_all.csv'
    filename='../test/newruns.csv'
    s=csvSelectionParser(filename)
    print 'runs : ',s.runs()
    print 'full result : ',s.runsandls()
    print 'str result : ',s.runsandlsStr()
    print 'num runs : ',s.numruns()
    #print 'numls in run : ',s.numls(135175)

