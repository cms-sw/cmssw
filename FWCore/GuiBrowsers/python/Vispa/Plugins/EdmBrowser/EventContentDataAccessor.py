import sys
import logging
import os.path

from Vispa.Share.BasicDataAccessor import BasicDataAccessor
from Vispa.Plugins.EdmBrowser.EdmDataAccessor import EdmDataAccessor

class EventContentDataAccessor(BasicDataAccessor):
    def __init__(self):
        self._eventContents = []
        self._configAccessors = {}
        
    def children(self, object):
        return []
    
    def isContainer(self, object):
        return False
        
    def label(self, object):
        return "_".join(object)

    def eventContentsList(self):
        """ return top level objects from file, e.g. the event.
        """
        return self._eventContents
    
    def topLevelObjects(self):
        objects = []
        for name, content, input, comment in self._eventContents:
            objects += list(content)
        return objects

    def addContents(self,content,content2):
        for object in content2:
            if not self.inContent(object,content):
                content+=[object]

    def properties(self, object):
        """ Make list of all properties """
        properties = []
        properties += [("Category", "Object info", "")]
        properties += [("String", "Label", object)]
        return properties

    def addConfig(self, accessor):
        if not accessor:
            return
        configName=os.path.splitext(os.path.basename(accessor.configFile()))[0]
        name = "Input: " + configName
        inputContent=accessor.inputEventContent()
        self._eventContents += [(name, inputContent[0], True, inputContent[1])]
        output_content=[]
        if len(self._eventContents)>1:
            self.addContents(output_content,self._eventContents[-2][1])
        output_content=self.applyCommands(output_content,accessor.inputCommands())
        self.addContents(output_content,accessor.outputEventContent())
        output_content=self.applyCommands(output_content,accessor.outputCommands())
        name = "Output: " + configName
        self._eventContents += [(name, output_content, False, {})]

    def addContentFile(self, filename):
        accessor=EdmDataAccessor()
        accessor.open(filename)
        branches=[branch[0].split("_") for branch in accessor.filteredBranches()]
        name = os.path.splitext(os.path.basename(filename))[0]
        self.addBranches(name,branches)
    
    def addBranches(self,name,branches):
        content = []
        for branch in branches:
            type = branch[0]
            label = branch[1]
            product = branch[2]
            process = branch[3]
            content += [(type,label,product,process)]
        self._eventContents += [(name, content, False, {})]

    def compareEntry(self, entry1, entry2):
        result=True
        for i in range(4):
            result=result and (entry1[i]==entry2[i] or entry1[i]=="*" or entry2[i]=="*")
        return result

    def inContent(self, entry, content):
        return True in [self.compareEntry(entry,c) for c in content]

    def applyCommands(self, content, outputCommands):
        keep = {}
        if len(outputCommands)>0 and outputCommands[0]!="keep *":
            for object in content:
                keep[object] = False
        else:
            for object in content:
                keep[object] = True
        for o in outputCommands:
            command, filter = o.split(" ")
            if len(filter.split("_")) > 1:
                module = filter.split("_")[1]
                product = filter.split("_")[2]
                process = filter.split("_")[3]
            else:
                module = filter
                product = "*"
                process = "*"
            for object in content:
                if "*" in module:
                    match = module.strip("*") in object[1]
                else:
                    match = module == object[1]
                if "*" in product:
                    match = match and product.strip("*") in object[2]
                else:
                    match = match and product == object[2]
                if "*" in process:
                    match = match and process.strip("*") in object[3]
                else:
                    match = match and process == object[3]
                if match:
                    keep[object] = command == "keep"
        return [object for object in content if keep[object]]

    def clear(self):
        self._eventContents=[]
        