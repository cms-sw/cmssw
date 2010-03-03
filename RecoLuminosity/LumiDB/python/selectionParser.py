import json
class selectionParser(object):
    def __init__(self,selectStr):
        self.__result={}
        strresult=json.loads(selectStr)
        for k,v in strresult.items():
            expandedvalues=[]
            for w in v:
            ###weed out [10]-like stuff just in case they exist
                if len(w)==1:
                    expandedvalues.append(w[0])
            ##weed out [10,10]-like stuff
                elif len(w)==2 and w[0]==w[1]:
                    expandedvalues.append(w[0])
                else:
                    for i in range(w[0],w[1]+1):
                        expandedvalues.append(i)
            self.__result[int(k)]=expandedvalues
    def runs(self):
        return self.__result.keys()
    def runsandls(self):
        '''return expanded {run:lslist}
        '''
        return self.__result
    def numruns(self):
        return len(self.__result.keys())
    def numls(self,run):
        return len(self.__result[run])
if __name__ == "__main__":
    s=selectionParser('{"1":[[3,45]],"2":[[4,8],[10,10]]}')
    print 'runs : ',s.runs()
    print 'full result : ',s.runsandls()
    print 'num runs : ',s.numruns()
    print 'numls in run : ',s.numls(1)
