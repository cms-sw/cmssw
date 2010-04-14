import csv
class csvReporter(object):
    def __init__(self,filename,fieldnames,dialect='excel',delimiter=',',quoting=csv.QUOTE_NONNUMERIC):
        self.__filename=filename
        self.__file=open(self.__filename,'wb')
        self.__fieldnames=fieldnames
        self.__writer=csv.DictWriter(self.__file,fieldnames,dialect=dialect,delimiter=delimiter,quoting=quoting)
        
    def writeHeader(self):
        try:
            self.__writer.writerow(dict( (n,n) for n in self.__fieldnames))
        except csv.Error,e:
            sys.exit('file %s: %s'%(self.__filename,e))
    def writeRow(self,rowDict):
        try:
            self.__writer.writerow(rowDict)
        except csv.Error,e:
            sys.exit('file %s: %s'%(self.__filename,e))   
    def close(self):
        self.__file.close()


#class screenReporter(object):
#    def __init__(self):
#        pass
 
    
if __name__ == '__main__':
    filename='testwrite.csv'
    fieldnames=('t1','t2','t3')
    r=csvReporter(filename,fieldnames,dialect='excel',delimiter=',')
    r.writeHeader()
    data={'t1':1,'t2':2,'t3':'a'}
    r.writeRow(data)
    data={'t1':11,'t2':22,'t3':'aa'}
    r.writeRow(data)
    #print csv.list_dialects()
    r.writeRow(data)
    r.close()
