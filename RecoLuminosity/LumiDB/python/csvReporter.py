import csv
class csvReporter(object):
    def __init__(self,filename,dialect='excel',delimiter=',',quoting=csv.QUOTE_NONNUMERIC):
        """input params:
        filename : output csv path/name
        """
        self.__filename=filename
        self.__file=open(self.__filename,'wb')
        self.__writer=csv.writer(self.__file)
    def writeRow(self,row):
        try:
            self.__writer.writerow(row)
        except csv.Error,e:
            sys.exit('file %s: %s'%(self.__filename,e))
    def writeRows(self,rows):
        try:
            self.__writer.writerows(rows)
        except csv.Error,e:
            sys.exit('file %s: %s'%(self.__filename,e))
    def close(self):
        self.__file.close()

if __name__ == '__main__':
    filename='testwrite.csv'
    r=csvReporter(filename,dialect='excel',delimiter=',')
    fieldnames=['t1','t2','t3']
    r.writeRow(fieldnames)
    data=[1,2,'a']
    r.writeRow(data)
    data=[11,22,'aa']
    r.writeRow(data)
    #print csv.list_dialects()
    r.writeRow(data)
    delivered=[['run','deliveredls','delivered'],[1234,23445,123.456],[2348,20,8893.0]]
    r.writeRows(delivered)
    r.close()
