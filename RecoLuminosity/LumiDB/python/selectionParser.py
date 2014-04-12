import json
class selectionParser(object):
    def __init__(self,selectStr):
        self.__result={}
        self.__strresult={}
        strresult=json.loads(selectStr)
        for k,v in strresult.items():
            expandedvalues=[]
            for w in v:
                if len(w)==0:
                    self.__result[int(k)]=expandedvalues
                    self.__strresult[k]=[]
                    continue
            ###weed out [10]-like stuff just in case they exist
                elif len(w)==1:
                    expandedvalues.append(w[0])
            ##weed out [10,10]-like stuff
                elif len(w)==2 and w[0]==w[1]:
                    expandedvalues.append(w[0])
                else:
                    for i in range(w[0],w[1]+1):
                        expandedvalues.append(i)
            self.__result[int(k)]=expandedvalues
            self.__strresult[k]=[str(x) for x in expandedvalues]
    def runs(self):
        return self.__result.keys()
    def runsandls(self):
        '''return expanded {run:lslist}
        '''
        return self.__result
    def runsandlsStr(self):
        '''return expanded {'run':lslist}
        '''
        return self.__strresult
    def numruns(self):
        return len(self.__result.keys())
    def numls(self,run):
        return len(self.__result[run])
if __name__ == "__main__":
    s=selectionParser('{"1":[[3,45]],"2":[[4,8],[10,10]],"3":[[]]}')
    print 'runs : ',s.runs()
    print 'full result : ',s.runsandls()
    print 'str result : ',s.runsandlsStr()
    print 'num runs : ',s.numruns()
    print 'numls in run : ',s.numls(1)
