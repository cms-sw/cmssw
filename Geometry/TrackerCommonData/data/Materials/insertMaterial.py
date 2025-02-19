#!/usr/bin/env python
import sys
import xml.dom.minidom
import math
import optparse

maxDist = 0.0001

#parse .tiles-File
def readFractions(fileName, materialMap):
    result = {}
    file = open(fileName,"r")
    name = None
    dens = None
    fractions = {}
    for line in file:
        line = line.strip("\n")
        if len(line.split("\""))==3:
            contentName = line.split("\"")[1]
            content = line.split("\"")[2].split()
            if len(content) == 2:
                if not name == None:
                    result[name] = dens,fractions
                name = contentName
                dens = content[1]
                fractions = {}
            elif len(content) == 1:
    #            print "  "+contentName+" "+str(float(content[0][1:])*0.01)
                fractions[getMaterial(contentName,materialMap)] = float(content[0][1:])*0.01
    
    if not name == None:
        result[name] = dens,fractions
                        
    for material in result:
        sum = 0
        for fraction in result[material][1]:
            sum+= result[material][1][fraction]
        if math.fabs(sum - 1.0) > maxDist:
            raise StandardError, "Material Fractions do not add up to 100%: "+ material+" "+str(sum)
    return result
        
#get a source:material from the [material] only
def getMaterial(material, fileName):
    materialMap ={}
    file = open(fileName,"r")
    for line in file:
        line = line.strip("\n")
        content = line.split()
        if len(content) == 2:
            materialMap[content[0]] = content[1] 
    if material in materialMap:
        result = materialMap[material]+":"+material
    else:
        result =  "materials:"+material
    return result

#parse XML-File
def readXML(fileName):
    file = open(fileName,'r')
    dom = xml.dom.minidom.parse(file)
    file.close()
    return dom

#get the last subnode named [name] of [rootNode]
def getSection(rootNode, name):
    result = None
    for node in rootNode.childNodes:
        if node.nodeName == name:
            result = node
    if result == None:
        raise StandardError, "Could not find: \""+name+"\" in childnodes of the rootNode!"
    return result          

#returns a map of [name] nodes by their names. stating from rootNode
def getNodes(rootNode, name):
    result = {}
    for node in rootNode.childNodes:
        if node.nodeName == name and node.nodeType == node.ELEMENT_NODE:
            for i in range(0,node.attributes.length):
                if node.attributes.item(i).name == "name":
                    result[node.attributes.item(i).nodeValue] = node
    return result
#returns a pretty printed string of [dom] withot whitespace-only-lines
def prettierprint(dom):
    result = ""
    output = dom.toprettyxml()
    for line in output.splitlines():
        if not line.strip(" \t") == "":
            result+= line+"\n"
    return result

#gets a map of attributes and their values of [node] 
#(not used but interessting for debug)
def getAttributes(node):
    result = {};
    for i in range(0,node.attributes.length):
#        print "  "+node.attributes.item(i).name+" = "+node.attributes.item(i).nodeValue
        result[node.attributes.item(i).name] = node.attributes.item(i).nodeValue    
    return result

#print information on all subnodes of [domina]
#(not used but interessting for debug)
def dokument(domina):
    for node in domina.childNodes:
        print "NodeName:", node.nodeName,
        if node.nodeType == node.ELEMENT_NODE:
            print "Typ ELEMENT_NODE"
            print getAttributes(node)
        elif node.nodeType == node.TEXT_NODE:
            print "Typ TEXT_NODE, Content: ", node.nodeValue.strip()
        elif node.nodeType == node.COMMENT_NODE:
            print "Typ COMMENT_NODE, "
        #dokument(node)

#prints all CompositeMaterials beneeth [rootNode]
#(not used but interessting for debug)
def printMaterials(rootNode):
    matNodes = getNodes(rootNode,"CompositeMaterial")
    for name in matNodes:
        print "  "+name+" (dens = "+getAttributes(matNodes[name])["density"]+")"
        for fractionNode in matNodes[name].childNodes:
            if fractionNode.nodeName == "MaterialFraction":
                fractionString = getAttributes(fractionNode)["fraction"]
                for materialNode in fractionNode.childNodes:
                    if materialNode.nodeName == "rMaterial":
                        fractionString += "\tof "+getAttributes(materialNode)["name"].split(":")[1]
                        fractionString += "\tfrom "+getAttributes(materialNode)["name"].split(":")[0]
                print "   |-- "+fractionString
    
#returns the Material Section doe of a DDD Material xmlfile
def getMaterialSection(rootNode):
    dddef = getSection(rootNode,'DDDefinition')
    matSec = getSection(dddef,'MaterialSection')
    return matSec

#creates a CompositeMaterial with [name] [method] [density] and [symbol] beneeth [rootNode]. 
#fractions is a map of material Names containing the fractions
#NOTE: if an material of that name allready exists it will be overridden. 
def createCompositeMaterial(doc,rootNode,name, density,fractions,method="mixture by weight", symbol=" "):
    newMaterial = doc.createElement("CompositeMaterial")
    newMaterial.setAttribute("name",name)
    newMaterial.setAttribute("density",density)
    newMaterial.setAttribute("method",method)
    newMaterial.setAttribute("symbol",symbol)    

    for fracMaterialName in fractions:
        fraction = doc.createElement("MaterialFraction")
        fraction.setAttribute("fraction",str(fractions[fracMaterialName]))
        newMaterial.appendChild(fraction)
        fracMaterial = doc.createElement("rMaterial")
        fracMaterial.setAttribute("name",fracMaterialName)
        fraction.appendChild(fracMaterial)

    exMaterials = getNodes(rootNode,"CompositeMaterial")
    if name in exMaterials:
        rootNode.replaceChild(newMaterial,exMaterials[name])
    else:
        rootNode.appendChild(newMaterial)

#main
def main():
    optParser = optparse.OptionParser()
    optParser.add_option("-t", "--titles", dest="titlesFile",
                  help="the .titles file to parse (as generated by mixture)", metavar="TITLES")
    optParser.add_option("-x", "--xml", dest="xmlFile",
                  help="the .xml file to parse (must be DDD complient)", metavar="XML")
    optParser.add_option("-o", "--output", dest="output",
                  help="the file to write the new materials into default: materialOutput.xml", metavar="XMLOUT")
    optParser.add_option("-m", "--materialMap", dest="materialMap",
                  help="file containing map of materials not defined in materials.xml. default: material.map", metavar="MMAP")

    (options, args) = optParser.parse_args()

    if options.titlesFile == None:
        raise StandardError, "no .titles File given!"
    if options.xmlFile == None:
        raise StandardError, "no .xml File given!"
    if options.output == None:
        options.output = "materialOutput.xml"
    if options.materialMap == None:
        options.materialMap = "material.map"
    theOptions = options

    materials = readFractions(options.titlesFile, options.materialMap)
    
    dom = readXML(options.xmlFile)
    matSec = getMaterialSection(dom)

 #   print "before:"
 #   printMaterials(matSec)

    for material in materials:
        createCompositeMaterial(dom,matSec,material,str(materials[material][0])+"*g/cm3",materials[material][1])

#    print "after:"
#    printMaterials(matSec)
    outFile = open(options.output,"w")
    outFile.write(prettierprint(dom))
    outFile.close()
    
main()

