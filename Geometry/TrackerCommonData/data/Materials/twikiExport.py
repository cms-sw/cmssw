#!/usr/bin/env python

import optparse
import ConfigParser

class Constituent:
    def __init__(self, line, predefinedMaterials):
        if  len(line.split('"')) < 5 or len(line.split('"')[4].split()) < 3:
            raise StandardError , "not a well formed Constituent: "+constString
        self.theDescription = line.split('"')[1]
        self.theName = line.split('"')[3]
        self.theCount = float(line.split('"')[4].split()[1])
        self.theVolume = float(line.split('"')[4].split()[0]) * self.theCount
        self.theType = line.split('"')[4].split()[2]
        self.theDensity = float(predefinedMaterials[self.theName][2])
        self.theX0 = float(predefinedMaterials[self.theName][3]) * self.theDensity
        self.theL0 = float(predefinedMaterials[self.theName][4]) * self.theDensity
        self.theMass = self.theVolume * self.theDensity
#        print "%s\tX0: %.3f\trho: %.3f\tX0': %.3f"%(self.theName, float(predefinedMaterials[self.theName][3]), float(predefinedMaterials[self.theName][2]), self.theX0)        
    def __str__(self):
        return "Name: "+self.theName+" desc: "+self.theDescription+" mass "+self.theMass+" type "+self.theType+" count "+self.theCount
        
class Material:
    def __init__(self, matString, comment):
        if matString == "" or not matString[0] == "#":
            raise StandardError , "not a valid material Definition: "+matString
        line = matString[1:]

        if    len( line.split('"') ) < 5 or len( line.split('"')[4].split() ) < 2:
            raise StandardError , "not a well formed Material Definition: "+matString            
        self.theDescription = line.split('"')[1]
        self.theName = line.split('"')[3]
        self.theMcVolume = float(line.split('"')[4].split()[0])
        self.theComment = comment
        self.theConstituents = []
    
    def getRealValues(self):
        result = {}
        result["Volume"] = 0
        result["X0"] = 0
        result["L0"] = 0
        
        totMass =  self.getMass()
        invX0 = 0
        invL0 = 0
        for con in self.theConstituents:
            pWeight = con.theVolume * con.theDensity / totMass
            invX0 += pWeight / con.theX0
            invL0 += pWeight / con.theL0
            result["Volume"] += con.theVolume 
        result["Density"] = self.getMass() / result["Volume"]            
        result["X0"] = 1 / ( invX0 * result["Density"] )
        result["L0"] = 1 / ( invL0 * result["Density"] )
        
        return result
    
    def getSimValues(self):
        result = self.getRealValues()
        fraction = self.theMcVolume / result["Volume"]
        result["Volume"] = self.theMcVolume
        result["Density"] = self.getMass() / self.theMcVolume
        result["X0"] *= fraction
        result["L0"] *= fraction
    
        return result
   
    def getMass(self):
        result = 0
        for con in self.theConstituents:
            result += con.theMass * con.theCount
        return result
        
    def addConstituent(self, constString, predefinedMaterials):
        if constString  == "" or not constString[0] == "*":
            raise StandardError , "not a valid Constituent: "+constString
        line = constString[1:]
        self.theConstituents.append( Constituent(line, predefinedMaterials) )

        number = int( line.split('"')[0].split()[0] )
        if not len(self.theConstituents) == number:
            raise StandardError, "Constituent Number mismatch for "+str(len(self.theConstituents))+" in: "+line
    
    def __str__(self):
        result = "[ "+self.theName+" Description: "+self.theDescription+" MC-Volume:"+str(self.theMcVolume)+"\n"
        result += "Comments:\n"+self.theComment
        return result

#parses the .in File
def parseInFile(inFile, predefinedMaterials, config):
    comment = ""
    materials = []
    i = 0
    for line in inFile.split("\n"):
        i=i+1
        if i < int(config["parsing"]["intro"]):
            continue
        if not line == "" and line[0] == "#":
            material = Material( line , comment )
            if not material.theName == "END":
                materials.append( material )
            comment = ""

            #print "Name: "+name+" Description: "+description+" mcVolume: "+str(mcVolume)
        elif not line == "" and line[0] == "*":
            
                materials[-1].addConstituent(line, predefinedMaterials)
        else:
            ignore = False
            for char in config["parsing"]["ignorelines"]:
                if len(line.strip()) == line.strip().count(char) and not len(line) == 0:
                    ignore = True
            if not ignore:
                comment += line+"\n"
    return materials

def parseComment(comment):
    result = ""
    if comment.find("<twiki>") >= 0:
        result = comment.split("twiki>")[1][:-2]
    else:
        result += "<verbatim>\n"
        result += comment
        result += "</verbatim>\n"        
    return result

def getTwiki(materials,config):
    result = config["twiki"]["pagename"]+"\n"
    result += "%TOC%\n\n"
    for mat in materials:
        result += config["twiki"]["heading"]+mat.theName+"\n"
        result += "short Description: *"+mat.theDescription+"* <br>\n"
        result += "Mass: %.3fg\n" % mat.getMass()
        result += config["twiki"]["tableformat"]+'\n'
        result += "| ** | *Volume* | *Density* |  *X<sub>0</sub>* | *&lambda;<sub>0</sub>* |\n"
        result += "| ** | *cms<sup>3</sup>* | *g cm<sup>-3</sup>* |  *cm* | *cm* |\n"
        result +=  "| Real | %(Volume).3f | %(Density).3f | %(X0).3f | %(L0).3f |\n" % mat.getRealValues()
        result +=  "| Simulation | %(Volume).3f | %(Density).3f | %(X0).3f | %(L0).3f |\n" % mat.getSimValues()
        result += "\n"+config["twiki"]["subheading"]+"Comments\n"
        result += parseComment(mat.theComment)
        result += "\n---+++!!Material\n"
        result += config["twiki"]["tableformat"]+'\n'
        result += "  | *Description* | *Material Name* | *Volume* | *Mass* | *Count* | *Type* | *X<sub>0</sub>* | *&lambda;<sub>0</sub>* |\n"
        result += '  | ** | ** | *cm<sup>3</sup>* | *g* | ** | ** | *g cm<sup>-2</sup>* | *g cm<sup>-2</sup>* |\n'
        for const in mat.theConstituents:
            result += "  | =%s= | =%s= | %.2e | %.2e | %.2f | %s | %.2f | %.2f |\n" % ( const.theDescription , const.theName , const.theVolume , const.theMass , const.theCount , const.theType , const.theX0 , const.theL0 )
#            result += "  | ="+const.theDescription+"= | ="+const.theName+"= | "+str(const.theVolume)+" | "+str(const.theMass)+" | "+str(const.theCount)+" | "+const.theType+" | "+str(const.theX0)+" | "+str(const.theL0)+" |\n"
        result += "\n"

    return result

#reads mixed_materials.input and pure_materials.input
def readMaterialFile(fileName):
    file = open(fileName,"r")
    result = {}
    for line in file:
        if not line == "\n":
            name = line.split('"')[1]
            content = line.split('"')[2].split()
            result[name] = content
    return result

#reads back one [section] of [config]
def readSection(config,section):
    result = {}
    for option in config.options(section):
        result[option] = config.get(section,option)
    return result

# reads the twikiConfig.ini
def readConfig(fileName):
    config = ConfigParser.ConfigParser()   
    config.read(fileName)
    result = {}
#    result["general"] = readSection("general")
    result["parsing"] = readSection(config,"parsing")
    result["twiki"] = readSection(config,"twiki")

    return result


#main
def main():
    optParser = optparse.OptionParser()
    optParser.add_option("-i", "--input", dest="inFile",
                  help="the .in material description that is the input", metavar="INPUT")
    optParser.add_option("-o", "--output", dest="output",
                  help="the file to put the twiki formated output in (default: <.in-Filename>.twiki)", metavar="TWIKIOUT")
    optParser.add_option("-c", "--config", dest="config",
                  help="configuration to use (default twikiConfig.ini)", metavar="CONFIG")

    (options, args) = optParser.parse_args()

    if options.inFile == None:
        raise StandardError, "no .in File given!"
    if options.output == None:
        options.output = options.inFile.replace(".in",".twiki")
    if options.config == None:
        options.config = "twikiConfig.ini"
    
    config = readConfig(options.config)

    predefinedMaterials = {}
    predefinedMaterials.update( readMaterialFile("pure_materials.input") )
    predefinedMaterials.update( readMaterialFile("mixed_materials.input") )

    inFile = open(options.inFile,"r")
    inFileContent = inFile.read()
    inFile.close()

    materials = parseInFile(inFileContent, predefinedMaterials, config)
    twikiString = getTwiki(materials, config)

    outFile = open(options.output,"w")
    outFile.write(twikiString)
    outFile.close()

main()
