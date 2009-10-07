
import copy

class filereader:

    class Module:
        def __init__(self,label='',value=[]):
            self.label=label
            self.value=value

    def __init__(self):
        self.aList=['Module', 'ESSource']
        
    def startswith(self,line):
        "Checks if the first word of the line starts with any of the aList elements"
        for item in self.aList:
            if line.startswith(item):
                return True
        return False    

    def readfile(self,nomefile):
        "Reads the file line by line and searches for the begin and the end of each Module block"       
        aFile = open(nomefile)
        module=[]
        value=[]
        file_modules = {}
        processHistory=False
        processing=False
        insideModuleBlock = False
        insideParameterBlock = False
        nprocess=-1
        key=''
        for line in aFile.readlines():
            if line.startswith("Processing History:"):
                value=[]
                processHistory=True
            elif (not line.startswith('---------Event')) and processHistory:
                splitLine= line.split()
                if splitLine[3]=='[2]':
                    processing=True
                    value.append(line)
                elif  processing:
                    value.append(line)
            elif line.startswith('---------Event') and processing:
                file_modules['Processing']=value
                processHistory=False
                processing=False
            elif self.startswith(line):
                if  insideParameterBlock:
                    module.append(tuple(value))
                    file_modules[key].append(module)
                    insideParameterBlock = False
                    insideModuleBlock = False  ###controllare
                value=[]
                module=[]
                splitLine= line.split()
                key=splitLine[-1]
                if key not in file_modules.keys():
                    file_modules[key]=[]
                module.append(splitLine[-2])
                value.append(line[:-1])
                insideModuleBlock = True
                insideParameterBlock = False
            elif (line.startswith(' parameters')) and insideModuleBlock:
                insideParameterBlock = True
                value.append(line[:-1])
            elif line.startswith('ESModule') and insideParameterBlock:
                module.append(tuple(value))
                file_modules[key].append(module)
                insideParameterBlock = False
                insideModuleBlock = False
            #elif line=='}' and insideParameterBlock:
                #module.append(tuple(value))
                #file_modules[key].append(module)
                #insideParameterBlock = False
                #insideModuleBlock = False
            elif (insideParameterBlock):
                value.append(line[:-1])
            
        if   insideParameterBlock:
            module.append(tuple(value))
            file_modules[key].append(module)
            insideParameterBlock = False
            insideModuleBlock = False 

        
        return file_modules 
                                                                                                                        
