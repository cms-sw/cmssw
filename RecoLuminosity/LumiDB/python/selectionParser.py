import json
class selectionParser(object):
    def __init__(self,selectStr):
        self.__result={}
        strresult=json.loads(selectStr)
        for k,v in strresult.items():
            self.__result[int(k)]=v
    def runs(self):
        return self.__result.keys()
    def runsandls(self):
        return self.__result
    def numruns(self):
        return len(self.__result.keys())
    def numls(self):
        pass
if __name__ == "__main__":
    s=selectionParser('{"1":[[3,45]],"2":[[4,8],[10,10]]}')
    print 'runs : ',s.runs()
    print 'full result : ',s.runsandls()
    print 'num runs : ',s.numruns()
