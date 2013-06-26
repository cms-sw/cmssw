class filereader:

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
        module = []
        source = []
        file_modules = {}
        insideModuleBlock = False
        insideParameterBlock = False
        for line in aFile.readlines():
            if self.startswith(line):
                if  insideParameterBlock:
                    file_modules[key]=module
                    insideParameterBlock = False
                    #print line[:-1]
                module=[]
                module.append(line[:-1])
                key=line[line.index(':')+2:-1]
                #print key
                insideModuleBlock = True
                insideParameterBlock = False
            elif (line.startswith(' parameters')) and insideModuleBlock:
                insideParameterBlock = True
                module.append(line[:-1])
                #print line[:-1]
            elif line.startswith('ESModule') and insideParameterBlock:
                file_modules[key]=module
                insideParameterBlock = False
                insideModuleBlock = False
            elif (insideParameterBlock):
                module.append(line[:-1])
                #print line[:-1]

        
        return file_modules 

