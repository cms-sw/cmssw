import os,csv
from RecoLuminosity.LumiDB import csvSelectionParser,selectionParser,CommonUtil
class inputFilesetParser(object):
    def __init__(self,inputfilename):
        filelist=inputfilename.split('+')
        self.__inputresultfiles=filelist[0:-1]
        self.__inputselectionfile=filelist[-1]
        self.__inputResultHeader=[]
        self.__inputResult=[]
        self.__inputSelectionFileparsingResult=None
        if len(self.__inputselectionfile)!=0:
            basename,extension=os.path.splitext(self.__inputselectionfile)
            if extension=='.csv':#if file ends with .csv,use csv parser,else parse as json file
                self.__inputSelectionFileparsingResult=csvSelectionParser.csvSelectionParser(self.__inputselectionfile)
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
                        self.__inputResultHeader=row
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
        '''
        output [headerfields]
        '''
        return self.__inputResultHeader
    def resultInput(self):
        '''
        output [valuefields]
        '''
        return self.__inputResult
    def fieldvalues(self,fieldname,fieldtype):
        '''
        given the input result field name and typem return the list of values
        '''
        fieldidx=None
        result=[]
        try:
            fieldidx=self.__inputResultHeader.index(fieldname)
        except:
            print 'field ',fieldname,' not found'
            raise
        for r in self.__inputResult:
            stringvalue=r[fieldidx]
            if fieldtype in ['int','unsigned int']:
                if not CommonUtil.is_intstr(stringvalue):
                    print 'field ',fieldname,' is not integer type'
                    raise
                else:
                    result.append(int(stringvalue))
                    continue
            elif fieldtype in ['float']:
                if not CommonUtil.is_floatstr(stringvalue):
                    print 'field ',fieldname,' is not float type'
                    raise
                else:
                    result.append(float(stringvalue))
                    contine
            elif  fieldtype in ['string','str']:
                result.append(stringvalue)
            else:
                raise 'unsupported type ',fieldtype
        return result
    def fieldtotal(self,fieldname,fieldtype):
        '''
        given the input result field name and type, return the total
        '''
        fieldidx=None
        result=0
        try:
            fieldidx=self.__inputResultHeader.index(fieldname)
        except:
            print 'field ',fieldname,' not found'
            raise
        for r in self.__inputResult:
            stringvalue=r[fieldidx]
            if fieldtype in ['int','unsigned int']:
                if not CommonUtil.is_intstr(stringvalue):
                    print 'field ',fieldname,' is not integer type'
                    raise
                else:
                    result=int(result)+int(stringvalue)
                    continue
            elif fieldtype in ['float'] :
                if not CommonUtil.is_floatstr(stringvalue):
                    print 'field ',fieldname,' is not float type'
                    raise
                else:
                    result=float(result)+float(stringvalue)
                    continue
            else:
                raise 'cannot sum types other than int ,float'
        return result
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
    filename='../test/overview.csv+../test/overview-140381.csv+../test/Cert_132440-139103_7TeV_StreamExpress_Collisions10_JSON.txt'
    #filename='../test/overview.csv+../test/overview-140381.csv+'
    p=inputFilesetParser(filename)
    print p.selectionfilename()
    print p.resultfiles()
    #print p.runs()
    #print p.runsandls()
    print 'do I only need to merge the results? ',p.mergeResultOnly()
    resultheader=p.resultHeader()
    result=p.resultInput()
    alreadyprocessedRuns=p.fieldvalues('run','int')
    print 'runs already have results ', alreadyprocessedRuns
    print 'total delivered ',p.fieldtotal('delivered','float')
    print 'total recorded ',p.fieldtotal('recorded','float')
    newrunsandls={}
    for run,cmslslist in p.runsandls().items():
        if run in alreadyprocessedRuns:
            continue
        else:
            newrunsandls[run]=cmslslist
    print 'runs and ls still need to be processed', newrunsandls
    filename='../test/lumi_900_output.json'
    p2=inputFilesetParser(filename)
    print 'result 2: ',p2.runs()
