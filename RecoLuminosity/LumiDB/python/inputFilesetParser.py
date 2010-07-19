import os,csv
from RecoLuminosity.LumiDB import csvSelectionParser,selectionParser
class inputFilesetParser(object):
    def __init__(self,inputfilename):
        filelist=inputfilename.split('+')
        self.__inputresultfiles=filelist[0:-1]
        self.__inputselectionfile=filelist[-1]
        self.__inputResultHeader=''
        self.__inputResult=[]
        self.__inputSelectionFileparsingResult=None
        if len(self.__inputselectionfile)!=0:
            basename,extension=os.path.splitext(self.__inputselectionfile)
            if extension=='.csv':#if file ends with .csv,use csv parser,else parse as json file
                self.__inputfileparsingResult=csvSelectionParser.csvSelectionParser(self.__inputselectionfile)
            else:
                selectf=open(self.__inputselectionfile,'r')
                inputfilecontent=selectf.read()
                self.__inputSelectionFileparsingResult=selectionParser.selectionParser(inputfilecontent)
        if len(self.__inputresultfiles)!=0:
            header=''
            for f in self.__inputresultfiles:
                ifile=open(f)
                hasHeader=csv.Sniffer().has_header(ifile.read(1024))
                ifile.seek(0)
                csvReader=csv.reader(ifile,delimiter=',')
                irow=0
                for row in csvReader:
                    if hasHeader and irow==0:
                        print 'header row ',row
                        self.__inputResultHeader=str(row).strip()
                    else:
                        self.__inputResult.append(row)
                    irow=irow+1
                ifile.close()
    def selectionfilename(self):
        '''return the input selection file name
        '''
        return self.__inputselectionfile
    def mergeResultOnly(self):
        '''if empty input selection filename give, I assume you only need to merge pieces of output result files into one 
        '''
        return len(self.__inputselectionfile)==0
    def resultfiles(self):
        return self.__inputresultfiles
    def resultHeader(self):
        return self.__inputResultHeader
    def resultInput(self):
        return self.__inputResult   
    def runs(self):
        if not self.__inputSelectionFileparsingResult:
            return None
        return self.__inputSelectionFileparsingResult.runs()
    def runsandls(self):
        if not self.__inputSelectionFileparsingResult:
            return None
        return self.__inputSelectionFileparsingResult.runsandls()
    def runsandlsStr(self):
        if not self.__inputSelectionFileparsingResult:
            return None
        return self.__inputSelectionFileparsingResult.runsandlsStr()
    
if __name__ == '__main__':
    result={}
    #filename='../test/overview.csv+../test/overview-140381.csv+../test/Cert_132440-139103_7TeV_StreamExpress_Collisions10_JSON.txt'
    filename='../test/overview.csv+../test/overview-140381.csv+'
    
    p=inputFilesetParser(filename)
    print p.selectionfilename()
    print p.resultfiles()
    print p.resultHeader()
    print p.resultInput()
    print p.runs()
    print p.mergeResultOnly()
    #print p.runsandls()
    #print p.runsandlsStr()
