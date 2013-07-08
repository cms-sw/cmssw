import ConfigParser,os.path

#################################
#norm file format spec
#lines beginning with a semicolon ';' a pound sign '#' or the letters 'REM' (uppercase or lowercase) will be ignored. 
#section uppercase
# [NORMDEFINITION] #section required only if first create
#   name=HFtest   #priority to commandline --name option if present
#   comment=
#   lumitype=
#   istypedefault=
# [NORMDATA Since] # section required
#   since= 
#   corrector=
#   norm_occ1=
#   norm_occ2=
#   amodetag=
#   egev=
#   comment=
#   ...
#################################

class normFileParser(object):
    def __init__(self,filename):
        self.__parser=ConfigParser.ConfigParser()
        self.__inifilename=filename
        self.__defsectionname='NormDefinition'
        self.__datasectionname='NormData'
    def parse(self):
        '''
        output:
           [ {defoption:value},[{dataoption:value}] ]
        '''
        if not os.path.exists(self.__inifilename) or not os.path.isfile(self.__inifilename):
            raise ValueError(self.__inifilename+' is not a file or does not exist')
        self.__parser.read(self.__inifilename)
        result=[]
        defsectionResult={}
        datasectionResult=[]      
        sections=self.__parser.sections()
        for section in sections:
            thisectionresult={}
            options=self.__parser.options(section)
            for o in options:
                try:
                    thisectionresult[o]=self.__parser.get(section,o)
                except:
                    continue
            if section==self.__defsectionname:
                defsectionResult=thisectionresult
            elif self.__datasectionname in section:
                datasectionResult.append(thisectionresult)
        return [defsectionResult,datasectionResult]
    
if __name__ == "__main__":
    s='../test/norm_HFV2.cfg'
    parser=normFileParser(s)
    print parser.parse()

